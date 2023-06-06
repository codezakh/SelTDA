import os
import json
import random
from PIL import Image
from typing import Optional, Union, List

import torch
from torch.utils.data import Dataset
from data.utils import pre_question

from torchvision.datasets.utils import download_url

STANDARD_DATASETS = ('vqa', 'aokvqa', 'okvqa')


class GenericVqaDataset(Dataset):
    def __init__(
        self, transform, ann_root: str, vqa_root: str, ann_files: List[str], split: str="train", truncate_to: Optional[int] = None,
        answer_list: str = 'answer_list'
    ):
        self.split = split

        self.transform = transform
        self.vqa_root = vqa_root
        self.truncate_to = truncate_to

        self.annotation = []
        for f in ann_files:
            # download_url(urls[f], ann_root)
            self.annotation += json.load(
                open(os.path.join(ann_root, "%s.json" % f), "r")
            )

        if self.truncate_to:
            self.annotation = self.annotation[:self.truncate_to]

        if split in ('test', 'val'):
            self.answer_list = json.load(
                open(os.path.join(ann_root, f'{answer_list}.json'), "r")
            )

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.vqa_root, ann["image"])

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        if self.split in ("test", "val"):
            question = pre_question(ann["question"])
            question_id = ann["question_id"]
            return image, question, question_id

        elif self.split == "train":
            question = pre_question(ann["question"])
            answer_weight = {}
            for answer in ann["answer"]:
                if answer in answer_weight.keys():
                    answer_weight[answer] += 1 / len(ann["answer"])
                else:
                    answer_weight[answer] = 1 / len(ann["answer"])
                
            answers = list(answer_weight.keys())
            weights = list(answer_weight.values())

            return image, question, answers, weights

# TODO: Create a generic VQA dataset class and clean up the code.
class vqa_dataset(Dataset):
    def __init__(
        self, transform, ann_root, vqa_root, vg_root, train_files=[], split="train", truncate_to: Optional[int] = None,
        append_rationale_to_answer: bool = False,
        append_rationale_to_question: bool = False,
        use_rationale_as_answer: Union[bool,str] = False,
    ):
        self.split = split

        self.transform = transform
        self.vqa_root = vqa_root
        self.vg_root = vg_root
        self.truncate_to = truncate_to
        self.append_rationale_to_answer = append_rationale_to_answer
        self.append_rationale_to_question = append_rationale_to_question
        self.use_rationale_as_answer = use_rationale_as_answer

        if split == "train":
            urls = {
                "vqa_train": "https://storage.googleapis.com/sfr-vision-language-research/datasets/vqa_train.json",
                "vqa_val": "https://storage.googleapis.com/sfr-vision-language-research/datasets/vqa_val.json",
                "vg_qa": "https://storage.googleapis.com/sfr-vision-language-research/datasets/vg_qa.json",
            }

            self.annotation = []
            for f in train_files:
                # download_url(urls[f], ann_root)
                self.annotation += json.load(
                    open(os.path.join(ann_root, "%s.json" % f), "r")
                )
        else:
            download_url(
                "https://storage.googleapis.com/sfr-vision-language-research/datasets/vqa_test.json",
                ann_root,
            )
            self.annotation = json.load(
                open(os.path.join(ann_root, "vqa_val.json"), "r")
            )

            download_url(
                "https://storage.googleapis.com/sfr-vision-language-research/datasets/answer_list.json",
                ann_root,
            )
            self.answer_list = json.load(
                open(os.path.join(ann_root, "answer_list.json"), "r")
            )
        if self.truncate_to:
            self.annotation = self.annotation[:self.truncate_to]

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        ann = self.annotation[index]

        # This is a vestigial case left over from the original code.
        # The questions in visual genome are handled separately.
        if ann["dataset"] == "vg":
            image_path = os.path.join(self.vg_root, ann["image"])
        else:
            image_path = os.path.join(self.vqa_root, ann["image"])

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        if self.split in ("test", "val"):
            question = pre_question(ann["question"])
            question_id = ann["question_id"]
            # We allow adding a rationale to the question
            # during test / validation time because we explore
            # models that use generated rationales during inference.
            if self.append_rationale_to_question:
                rationale = ' '.join(ann["rationales"])
                question = f"{question}. Rationale: {rationale}"
            return image, question, question_id

        elif self.split == "train":

            question = pre_question(ann["question"])

            # The ground truth questions in the vg dataset have only
            # one answer, but the questions in the generated vg dataset
            # have multiple answers, so handle them like the vqa questions
            # which have multiple answers.
            if ann["dataset"] == "vqa" or isinstance(ann["answer"], list):
                # The two blocks below should be identically except one uses
                # the rationales as the answer. This is so we can have a 
                # model that generates rationales given a question, rather
                # than an answer. 
                if not self.use_rationale_as_answer:
                    answer_weight = {}
                    for answer in ann["answer"]:
                        if answer in answer_weight.keys():
                            answer_weight[answer] += 1 / len(ann["answer"])
                        else:
                            answer_weight[answer] = 1 / len(ann["answer"])
                elif self.use_rationale_as_answer == 'separated':
                    answer_weight = {}
                    for answer in ann["rationales"]:
                        if answer in answer_weight.keys():
                            answer_weight[answer] += 1 / len(ann["rationales"])
                        else:
                            answer_weight[answer] = 1 / len(ann["rationales"])
                elif self.use_rationale_as_answer == 'concatenated':
                    answer_weight = {}
                    answer = ' '.join(ann["rationales"])
                    answer_weight[answer] = 1
                else:
                    raise ValueError(f"Invalid value for use_rationale_as_answer: {self.use_rationale_as_answer}")
                    
                answers = list(answer_weight.keys())
                weights = list(answer_weight.values())

            elif ann["dataset"] == "vg":
                answers = [ann["answer"]]
                weights = [0.2]

            if self.append_rationale_to_answer:
                rationale = ' '.join(ann["rationales"])
                answers = [f"{answer}. Rationale: {rationale}" for answer in answers]
            if self.append_rationale_to_question:
                rationale = ' '.join(ann["rationales"])
                question = f"{question}. Rationale: {rationale}"

            return image, question, answers, weights


class AokVqaDataset(vqa_dataset):
    def __init__(
        self, transform, ann_root, vqa_root, vg_root, train_files=[], split="train",
        truncate_to: Optional[int] = None,
        append_rationale_to_answer: bool = False,
        append_rationale_to_question: bool = False,
        use_rationale_as_answer: Union[bool,str] = False,
    ):
        """
        Args:
            transform (_type_): _description_
            ann_root (_type_): _description_
            vqa_root (_type_): _description_
            vg_root (_type_): _description_
            train_files (list, optional): _description_. Defaults to [].
            split (str, optional): _description_. Defaults to "train".
            truncate_to (Optional[int], optional): _description_. Defaults to None.
            append_rationale_to_answer (bool, optional): _description_. Defaults to False.
            append_rationale_to_question (bool, optional): _description_. Defaults to False.
            use_rationale_as_answer (Union[bool,str], optional): Replaces the answers in the datasets
                with the rationales. Set it to false if you don't want to do this. Otherwise, set it 
                to 'concatenated' if you want to have a single long rationale for each question, and 
                'separated' if you want to have multiple small rationales for each question. Defaults to False.
        """
        self.split = split

        self.transform = transform
        self.vqa_root = vqa_root
        self.vg_root = vg_root
        self.truncate_to = truncate_to
        self.append_rationale_to_answer = append_rationale_to_answer
        self.append_rationale_to_question = append_rationale_to_question
        self.use_rationale_as_answer = use_rationale_as_answer

        if split == "train":
            self.annotation = []
            for f in train_files:
                self.annotation += json.load(
                    open(os.path.join(ann_root, "%s.json" % f), "r")
                )
        elif split == "val":
            self.annotation = json.load(
                open(os.path.join(ann_root, "val.json"), "r")
            )
            self.answer_list = json.load(
                open(os.path.join(ann_root, "answer_list.json"), "r")
            )
        else:
            self.annotation = json.load(
                open(os.path.join(ann_root, "test.json"), "r")
            )
            self.answer_list = json.load(
                open(os.path.join(ann_root, "answer_list.json"), "r")
            )

        if self.truncate_to:
            self.annotation = self.annotation[:self.truncate_to]


class OkVqaDataset(vqa_dataset):
    def __init__(
        self, transform, ann_root, vqa_root, train_files=[], split="train",
        truncate_to: Optional[int] = None,
    ):
        self.split = split

        self.transform = transform
        self.vqa_root = vqa_root
        self.truncate_to = truncate_to

        # We set these to False because the __getitem__ method
        # in the parent class tries to access these attributes,
        # even though they are irrelevant for this dataset.
        # TODO: Turn the parent classes's __getitem__ method into
        # a mixin and implement a getter for these that returns fault
        # by default.
        self.append_rationale_to_answer = False
        self.append_rationale_to_question = False 
        self.use_rationale_as_answer = False 

        if split == "train":
            self.annotation = []
            for f in train_files:
                self.annotation += json.load(
                    open(os.path.join(ann_root, "%s.json" % f), "r")
                )
        elif split == "val":
            self.annotation = json.load(
                open(os.path.join(ann_root, "val.json"), "r")
            )
            self.answer_list = json.load(
                open(os.path.join(ann_root, "answer_list.json"), "r")
            )
        else:
            raise ValueError("There is no test set for OKVQA")

        if self.truncate_to:
            self.annotation = self.annotation[:self.truncate_to]

class ArtVQADataset(vqa_dataset):
    def __init__(
        self, transform, ann_root, vqa_root, train_files=[], split="train",
        truncate_to: Optional[int] = None,
    ):
        self.split = split

        self.transform = transform
        self.vqa_root = vqa_root
        self.truncate_to = truncate_to

        # We set these to False because the __getitem__ method
        # in the parent class tries to access these attributes,
        # even though they are irrelevant for this dataset.
        # TODO: Turn the parent classes's __getitem__ method into
        # a mixin and implement a getter for these that returns fault
        # by default.
        self.append_rationale_to_answer = False
        self.append_rationale_to_question = False 
        self.use_rationale_as_answer = False 

        if split == "train":
            self.annotation = []
            for f in train_files:
                self.annotation += json.load(
                    open(os.path.join(ann_root, "%s.json" % f), "r")
                )
        elif split == "test":
            self.annotation = json.load(
                open(os.path.join(ann_root, "test.json"), "r")
            )
            self.answer_list = json.load(
                open(os.path.join(ann_root, "answer_list.json"), "r")
            )
        else:
            # Why? Because current train_vqa only supports using two datasets:
            # the train and the evaluation dataset. We need predictions for the
            # test set, we don't need predictions for the validation set, and there's
            # no way (yet) to programmatically configure the evaluation set.
            # TODO: Allow programaatic configuration of the evaluation set.
            raise ValueError("Not using the validation set for ArtVQA")

        if self.truncate_to:
            self.annotation = self.annotation[:self.truncate_to]

class VQARephrasingsDataset(vqa_dataset):
    def __init__(
        self, transform, ann_root, vqa_root, train_files=[], split="train",
        truncate_to: Optional[int] = None,
    ):
        self.split = split

        self.transform = transform
        self.vqa_root = vqa_root
        self.truncate_to = truncate_to

        # We set these to False because the __getitem__ method
        # in the parent class tries to access these attributes,
        # even though they are irrelevant for this dataset.
        # TODO: Turn the parent classes's __getitem__ method into
        # a mixin and implement a getter for these that returns fault
        # by default.
        self.append_rationale_to_answer = False
        self.append_rationale_to_question = False 
        self.use_rationale_as_answer = False 

        if split == "train":
            self.annotation = []
            for f in train_files:
                self.annotation += json.load(
                    open(os.path.join(ann_root, "%s.json" % f), "r")
                )
        elif split == "val":
            self.annotation = json.load(
                open(os.path.join(ann_root, "val.json"), "r")
            )
            self.answer_list = json.load(
                open(os.path.join(ann_root, "answer_list.json"), "r")
            )
        else:
            raise ValueError("There is no test set for VQA Rephrasings")

        if self.truncate_to:
            self.annotation = self.annotation[:self.truncate_to]

class PathVqaDataset(vqa_dataset):
    def __init__(
        self, transform, ann_root, vqa_root, train_files=[], split="train",
        truncate_to: Optional[int] = None,
    ):
        self.split = split

        self.transform = transform
        self.vqa_root = vqa_root
        self.truncate_to = truncate_to

        # We set these to False because the __getitem__ method
        # in the parent class tries to access these attributes,
        # even though they are irrelevant for this dataset.
        # TODO: Turn the parent classes's __getitem__ method into
        # a mixin and implement a getter for these that returns fault
        # by default.
        self.append_rationale_to_answer = False
        self.append_rationale_to_question = False 
        self.use_rationale_as_answer = False 

        if split == "train":
            self.annotation = []
            for f in train_files:
                self.annotation += json.load(
                    open(os.path.join(ann_root, "%s.json" % f), "r")
                )
        elif split == "test":
            self.annotation = json.load(
                open(os.path.join(ann_root, "test.json"), "r")
            )
            self.answer_list = json.load(
                open(os.path.join(ann_root, "answer_list.json"), "r")
            )
        else:
            # Why? Because current train_vqa only supports using two datasets:
            # the train and the evaluation dataset. We need predictions for the
            # test set, we don't need predictions for the validation set, and there's
            # no way (yet) to programmatically configure the evaluation set.
            # TODO: Allow programaatic configuration of the evaluation set.
            raise ValueError("Not using the validation set for PathVQA")

        if self.truncate_to:
            self.annotation = self.annotation[:self.truncate_to]

class RsvqaDataset(vqa_dataset): 
    def __init__(
        self, transform, ann_root, vqa_root, train_files=[], split="train",
        truncate_to: Optional[int] = None,
    ):
        self.split = split

        self.transform = transform
        self.vqa_root = vqa_root
        self.truncate_to = truncate_to

        # We set these to False because the __getitem__ method
        # in the parent class tries to access these attributes,
        # even though they are irrelevant for this dataset.
        # TODO: Turn the parent classes's __getitem__ method into
        # a mixin and implement a getter for these that returns fault
        # by default.
        self.append_rationale_to_answer = False
        self.append_rationale_to_question = False 
        self.use_rationale_as_answer = False 

        if split == "train":
            self.annotation = []
            for f in train_files:
                self.annotation += json.load(
                    open(os.path.join(ann_root, "%s.json" % f), "r")
                )
        elif split == "test":
            self.annotation = json.load(
                open(os.path.join(ann_root, "test.json"), "r")
            )
            self.answer_list = json.load(
                open(os.path.join(ann_root, "answer_list.json"), "r")
            )
        else:
            # Why? Because current train_vqa only supports using two datasets:
            # the train and the evaluation dataset. We need predictions for the
            # test set, we don't need predictions for the validation set, and there's
            # no way (yet) to programmatically configure the evaluation set.
            # TODO: Allow programaatic configuration of the evaluation set.
            raise ValueError("Not using the validation set for RSVQA")

        if self.truncate_to:
            self.annotation = self.annotation[:self.truncate_to]

def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights
        answer_list += answer
        n.append(len(answer))
    return (
        torch.stack(image_list, dim=0),
        question_list,
        answer_list,
        torch.Tensor(weight_list),
        n,
    )
