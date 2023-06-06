import cli
import json
from typing import List, Dict, Literal, Optional
import cattrs
from omegaconf import DictConfig
from pathlib import Path
import logging
from pydantic import BaseModel
import schemas
from enum import Enum
from tqdm import tqdm


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s -  %(name)s: %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


IMAGES_ROOT = Path("/net/acadia10a/data/zkhan/coco2014")
OKVQA_ROOT = Path("/net/acadia10a/data/zkhan/ok-vqa")


class SplitType(Enum):
    train = "train"
    val = "val"


class OkvqaSplit(BaseModel):
    split: SplitType
    annotations: List[schemas.VQAAnnotationRecord]
    questions: List[schemas.QuestionRecord]

    @classmethod
    def build_from_path(
        cls, split_name: str, annotation_path: Path, question_path: Path
    ):
        logger.info(
            f"Loading {split_name} split from {annotation_path} and {question_path}"
        )
        with open(annotation_path, "r") as f:
            annotations = json.load(f)["annotations"]
        with open(question_path, "r") as f:
            questions = json.load(f)["questions"]
        logger.info(
            "Structuring %d annotations and %d questions",
            len(annotations),
            len(questions),
        )
        annotations = [
            schemas.VQAAnnotationRecord.parse_obj(a) for a in tqdm(annotations)
        ]
        questions = [schemas.QuestionRecord.parse_obj(q) for q in tqdm(questions)]
        assert len(annotations) == len(questions)
        return cls(
            split=SplitType(split_name), annotations=annotations, questions=questions
        )


# OKVQA only has training and validation splits. Both of them have annotations and questions.
# This is in contrast to VQAv2, which has a test split with only questions.
# So while we could make a single function to convert both splits to `TrainingRecord`s, we
# keep them separate to be consistent with VQAv2, where the testing split is structured
# as `TestingRecord`s.


def make_train_record(
    annotation: schemas.VQAAnnotationRecord, question: schemas.QuestionRecord
) -> schemas.TrainingRecord:
    assert annotation.question_id == question.question_id
    assert question.image_id == annotation.image_id
    return schemas.TrainingRecord(
        dataset="okvqa",
        image=annotation.image_id,
        question=question.question,
        question_id=question.question_id,
        answer=[_.answer for _ in annotation.answers],
    )


def make_val_record(record: schemas.QuestionRecord) -> schemas.TestingRecord:
    return schemas.TestingRecord(
        question_id=record.question_id,
        question=record.question,
        image=record.image_id,
        dataset="okvqa",
    )


def pad_coco_id(coco_id: int) -> str:
    return f"{coco_id:0>12}"


def point_record_to_coco_image_file(
    record: schemas.TrainingRecord, split: SplitType
) -> schemas.TrainingRecord:
    coco_image_id = pad_coco_id(record.image)
    relative_path_to_image = (
        f"{split.value}2014/COCO_{split.value}2014_{coco_image_id}.jpg"
    )
    record.image = relative_path_to_image
    return record


if __name__ == "__main__":
    train = OkvqaSplit.build_from_path(
        "train",
        annotation_path=OKVQA_ROOT / "mscoco_train2014_annotations.json",
        question_path=OKVQA_ROOT / "OpenEnded_mscoco_train2014_questions.json",
    )

    val = OkvqaSplit.build_from_path(
        "val",
        annotation_path=OKVQA_ROOT / "mscoco_val2014_annotations.json",
        question_path=OKVQA_ROOT / "OpenEnded_mscoco_val2014_questions.json",
    )
    logger.info(
        "Converting %d annotations and questions into training records",
        len(train.annotations),
    )
    train_records = [
        make_train_record(a, q)
        for a, q in tqdm(zip(train.annotations, train.questions))
    ]

    logger.info("Verifying all %d images are present", len(train_records))
    train_records = [
        point_record_to_coco_image_file(r, train.split) for r in tqdm(train_records)
    ]
    for record in tqdm(train_records):
        assert (IMAGES_ROOT / record.image).exists()

    logger.info("Converting %d questions into validation records", len(val.questions))
    val_records = [make_val_record(q) for q in tqdm(val.questions)]
    logger.info("Verifying all %d images are present", len(val_records))
    val_records = [
        point_record_to_coco_image_file(r, val.split) for r in tqdm(val_records)
    ]
    for record in tqdm(val_records):
        assert (IMAGES_ROOT / record.image).exists()

    # Use these for training and generating answers, but use the
    # original annotations for scoring the answers.
    with open(OKVQA_ROOT / "train.json", "w") as f:
        json.dump([r.dict() for r in train_records], f)
    logger.info(
        "Wrote %d training records to %s", len(train_records), OKVQA_ROOT / "train.json"
    )

    with open(OKVQA_ROOT / "val.json", "w") as f:
        json.dump([r.dict() for r in val_records], f)
    logger.info(
        "Wrote %d validation records to %s", len(val_records), OKVQA_ROOT / "val.json"
    )
