"""
This script converts the VQA-Rephrasings dataset into a format usable by `train_vqa.py`.
VQA-Rephrasings is a test set only dataset, which supplies rephrasings of the questions
in the VQAv2 validation set (the VQAv2 test set is not publically available.) The images
come from the COCO14 validation set.
"""


import json
from typing import List, Dict, Literal, Optional
from omegaconf import DictConfig
from pathlib import Path
import logging
import schemas
from enum import Enum
from tqdm import tqdm
import shutil

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s -  %(name)s: %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
    )
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


IMAGES_ROOT = Path('/net/acadia10a/data/zkhan/coco2014')
VQA_REPHRASINGS_ROOT = Path('/net/acadia4a/data/zkhan/vqa-rephrasings')
VQA_V2_ANSWER_LIST = '/net/acadia10a/data/zkhan/vqav2_annotations/answer_list.json'

def load_json(path: Path) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)

def make_val_record(record: schemas.QuestionRecord) -> schemas.TestingRecord:
    return schemas.TestingRecord(
        question_id=record.question_id,
        question=record.question,
        image=record.image_id,
        dataset='vqa-rephrasings',
    )


def pad_coco_id(coco_id: int) -> str:
    return f"{coco_id:0>12}"

def point_record_to_coco_image_file(record: schemas.TrainingRecord, split: str) -> schemas.TrainingRecord:
    coco_image_id = pad_coco_id(record.image)
    relative_path_to_image = f"{split}2014/COCO_{split}2014_{coco_image_id}.jpg"
    record.image = relative_path_to_image
    return record

if __name__ == "__main__":
    val_questions_raw =  load_json(VQA_REPHRASINGS_ROOT / 'v2_OpenEnded_mscoco_valrep2014_humans_og_questions.json')

    logger.info('Converting %d records', len(val_questions_raw['questions']))

    val_questions = [
        make_val_record(schemas.QuestionRecord.parse_obj(_)) for _ in val_questions_raw['questions']
    ]

    logger.info('Verifying all image paths are correct.') 
    val_records = [point_record_to_coco_image_file(r, 'val') for r in tqdm(val_questions)]
    for record in tqdm(val_records):
        assert (IMAGES_ROOT / record.image).exists()
    

    with open(VQA_REPHRASINGS_ROOT / 'val.json', 'w') as f:
        json.dump([r.dict() for r in val_records], f)
    logger.info('Wrote %d validation records to %s', len(val_records), VQA_REPHRASINGS_ROOT / 'val.json')
    shutil.copyfile(VQA_V2_ANSWER_LIST, VQA_REPHRASINGS_ROOT / 'answer_list.json')


    # Make a fake training file just so `train_vqa.py` doesn't complain.
    fake_training_record = schemas.TrainingRecord(
        question_id=0,
        question='fake question',
        image='/not/a/real/path.jpg',
        answer=['fake answer'],
        dataset='vqa-rephrasings',
    )
    with open(VQA_REPHRASINGS_ROOT / 'train.json', 'w') as f:
        json.dump([fake_training_record.dict()], f)