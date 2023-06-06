import glob
import json
from enum import Enum
from pathlib import Path
from typing import List, Optional

import attrs
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
import logging
import random
import re

import cli
from models.blip import blip_decoder
import random


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s -  %(name)s: %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# These regexes parse the raw model output into a structured format.
# We need all of these because the order of the output can be
# changed based on the training.
question_regex = r"question[^:]*:(?P<question>[^?]*\?)"
question_at_start_regex = r"^[^\w]*(?P<question>[^?]*\?)"
answer_regex = r"answer[^:]*:(?P<answer>[^.]*)"
rationale_regex = r"rationale[^:]*:(?P<rationale>.*$)"
rationale_at_start_regex = r"rationale[^:]*:(?P<rationale>.*?(?=question))"
lazy_take = r".*?"
ques_answer_rationale = (
    question_at_start_regex + lazy_take + answer_regex + lazy_take + rationale_regex
)
rationale_ques_answer = (
    rationale_at_start_regex + lazy_take + question_regex + lazy_take + answer_regex
)
question_answer = question_at_start_regex + lazy_take + answer_regex


class VQADatasetOrigin(Enum):
    visual_genome = "vg"
    vqa_v2 = "vqa"


def collate_safe(batch):
    """
    Drop any `None`-valued elements before batching.
    This requires that the dataset's __getitem__ method returns
    None when the data is corrupt or broken.
    """
    non_null_samples = list(filter(lambda x: x is not None, batch))
    reconstituted_batch = []
    for sample in batch:
        if sample is None:
            reconstituted_batch.append(random.choice(non_null_samples))
        else:
            reconstituted_batch.append(sample)
    return torch.utils.data.dataloader.default_collate(reconstituted_batch)


@attrs.define
class VQARecord:
    question_id: int
    question: str
    answer: List[str]
    image: str
    dataset: str
    rationale: Optional[str] = None

    @classmethod
    def build_from_raw_model_output(
        cls,
        model_output,
        image_path,
        dataset_origin=VQADatasetOrigin.visual_genome,
        question_id: int = 0,
        parse_rationale: bool = False,
    ):

        if parse_rationale:
            parsed = re.search(ques_answer_rationale, model_output)
            if parsed is None:
                parsed = re.search(rationale_ques_answer, model_output)

            question = parsed.group("question").strip()
            answer = parsed.group("answer").strip()
            rationale = parsed.group("rationale").strip()
        else:
            parsed = re.search(question_answer, model_output)
            question = parsed.group("question").strip()
            answer = parsed.group("answer").strip()
            rationale = None

        if answer.startswith(":"):
            answer = answer.replace(":", "").strip()

        answer = [_.strip() for _ in answer.split(",")]
        if "yes" in answer and "no" in answer:
            raise AmbiguousBooleanAnswerError(
                f"Yes and no are both in the answer: {answer}"
            )

        # Keep the name of the image and the parent folder,
        # discard the rest. This matches the format BLIP uses.
        image_path = Path(image_path)
        truncated_path = f"{image_path.parent.name}/{image_path.name}"
        return cls(
            question_id=question_id,
            question=question,
            answer=answer,
            image=truncated_path,
            dataset=dataset_origin.value,
            rationale=rationale,
        )


class AmbiguousBooleanAnswerError(Exception):
    pass


class ImagesForGenerationDS(Dataset):
    def __init__(
        self,
        image_root,
        transform=None,
        truncate_to: int = None,
        annotations_fname=None,
    ):
        """Create a dataset of images for generation.

        The dataset can be truncated to a number of images, or only load images specified by
        an annotations file.

        Args:
            image_root (str): Absolute path to the folder containing the images. Will
                be globbed (single level) to find all images. Alternatively, a list of
                images within this folder can be specified by the annotations file.
            transform (optional): Transform to apply to each image. Defaults to None.
            truncate_to (int, optional): Number of images to truncate the dataset to.
                Defaults to None.
            annotations_fname (str, optional): The absolute path to a JSON file file
                containing image paths within the image folder to use. Should be a
                list of dictionaries, each having an `image` key, which locates an image
                when combined with the image folder path. Defaults to None.
        """
        self.image_root = Path(image_root)
        self.transform = transform
        self.truncate_to = truncate_to
        self.annotations_fname = annotations_fname

        assert self.image_root.exists()

        # The reason we have two ways of loading images is because sometimes we want to
        # generate questions for images that only belong to a specific dataset (e.g. A-OKVQA)
        # but the images that the dataset uses are sourced from a larger image dataset (e.g. COCO)
        # which is used by a bunch of other datasets and stored in its entirety on disk.
        # We could manually pick out the used subset and save it elsewhere, but
        # wee don't want to have a bunch of duplicates
        # of a subset of the images in the larger dataset lying around, so we only load the images
        # that are specified in the annotations file.

        # Discover image paths by globbing the image root.
        if self.annotations_fname is None:
            logger.info("Globbing images from %s", self.image_root)
            self.image_paths = []
            for idx, image_path in enumerate(glob.iglob(f"{self.image_root}/*.jpg")):
                if self.truncate_to is not None and idx >= self.truncate_to:
                    break

                self.image_paths.append(image_path)

        # Discover image paths by reading the annotations file.
        else:
            logger.info(
                "Reading images from annotations file %s", self.annotations_fname
            )
            with open(self.annotations_fname, "r") as f:
                annotations = json.load(f)
            self.image_paths = []
            for idx, annotation in enumerate(annotations):
                if self.truncate_to is not None and idx >= self.truncate_to:
                    break
                image_path = annotation["image"]
                self.image_paths.append(str(self.image_root / image_path))

    def __getitem__(self, index: int):
        try:
            image = Image.open(self.image_paths[index]).convert("RGB")
        except Exception as e:
            logger.warning(
                f"Failed to load image {self.image_paths[index]} at index %d", index
            )
            logger.exception(e)
            return None
        if self.transform:
            image = self.transform(image)
        return image, self.image_paths[index]

    def __len__(self):
        return len(self.image_paths)


def build_model_from_config(config):
    model = blip_decoder(
        med_config=config.multimodal_encoder_decoder_config,
        pretrained=config.pretrained,
        image_size=config.image_size,
        vit=config.vit,
        prompt=config.prompt,
    )
    model.eval()
    return model


def build_dataset_from_config(config):
    transform = transforms.Compose(
        [
            transforms.Resize(
                (config.image_size, config.image_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    return ImagesForGenerationDS(
        config.image_folder,
        transform=transform,
        truncate_to=config.truncate_to,
        annotations_fname=config.annotations,
    )


def main(args, config):
    logger.info("Instantiating model from %s", config.pretrained)
    model = build_model_from_config(config)
    logger.info("Building dataset")
    ds = build_dataset_from_config(config)
    logger.info("Dataset built with %d images", len(ds))
    loader = DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_safe,
    )

    model.to(args.device)

    if config.dry_run:
        logger.info("Dry run, not generating any questions")

    all_records = []
    successful_parses = 0
    failed_parses = 0
    for batch_idx, batch in tqdm(enumerate(loader), total=len(loader)):
        images, image_paths = batch

        if config.dry_run:
            continue

        images = images.to(args.device)
        # Repeat the generation config.questions_per_image times.
        # This will result in multiple questions per image, because we
        # are repeating the image and generating a new question each time.
        for _ in range(config.questions_per_image):
            with torch.no_grad():
                outputs = model.generate(
                    images,
                    sample=True,
                    top_p=config.top_p,
                    max_length=config.max_length,
                    min_length=config.min_length,
                )

            for idx, (model_output, image_path) in enumerate(zip(outputs, image_paths)):
                try:
                    record = VQARecord.build_from_raw_model_output(
                        model_output,
                        image_path,
                        dataset_origin=VQADatasetOrigin(config.vqa_dataset_origin),
                        # We don't want a mix of rationale / non rationale questions.
                        # If the config says we want rationales, every question
                        # should have one. So we tell the code explicitly to
                        # parse the rationale, and it will throw an error if it
                        # couldn't parse one.
                        parse_rationale=config.parse_rationale,
                    )
                    record.question_id = idx
                except Exception as e:
                    if isinstance(e, KeyboardInterrupt):
                        raise e
                    logger.exception(e)
                    logger.warning(
                        f"Failed to parse output %s into question-answer pair for image %s",
                        model_output,
                        image_path,
                    )
                    failed_parses += 1
                    continue
                else:
                    all_records.append(attrs.asdict(record))
                    successful_parses += 1

        if len(all_records) >= config.truncate_to_strict:
            break

        logger.info(
            "Sucessfully parsed %d questions, failed to parse %d questions",
            successful_parses,
            failed_parses,
        )

    # Useful when you have a small number of images and a large number of questions per image.
    # When slowly increasing the number of QA pairs being trained on, this should probably
    # be set to True to avoid missing out on a large number of images that have no questions.
    # E.g. with only 3k images and 10k questions per image, training on 3k synthetic questions
    # in order will only get you ~300 unique images.
    if config.shuffle:
        logger.info("Shuffling records")
        random.shuffle(all_records)

    logger.info("Generated %d questions", len(all_records))
    logger.info("Writing questions to %s", config.output_annotations_name)
    try:
        with open(
            Path(config.output_folder) / config.output_annotations_name, "w"
        ) as f:
            json.dump(all_records, f)
    except Exception as e:
        import ipdb

        ipdb.set_trace()
    logger.info("Successfully serialized questions")


if __name__ == "__main__":
    args, config = cli.parse_args(
        default_config_path="./configs/generate_questions_vg.yaml"
    )
    cli.setup(args, config)
    main(args, config)
