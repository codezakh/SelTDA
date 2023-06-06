"""
This script converts the AdVQA dataset into a format usable by our training code.
The images in the AdVQA training dataset come from CC3M, Fakeddit and VCR, but the
testing / validation images come only from COCO.

AdVQA makes available a training, validation and testing split. Right now, I only use
the validation split. 
"""

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
    "%(asctime)s -  %(name)s: %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
    )
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


IMAGES_ROOT = Path('/net/acadia10a/data/zkhan/coco2014')
ADVQA_ROOT = Path('/net/acadia10a/data/zkhan/advqa')


def make_val_record(record: schemas.QuestionRecord) -> schemas.TestingRecord:
    return schemas.TestingRecord(
        question_id=record.question_id,
        question=record.question,
        image=record.image_id,
        dataset='advqa',
    )


def pad_coco_id(coco_id: int) -> str:
    return f"{coco_id:0>12}"

def point_record_to_coco_image_file(record: schemas.TrainingRecord, split: str) -> schemas.TrainingRecord:
    coco_image_id = pad_coco_id(record.image)
    relative_path_to_image = f"{split}2014/COCO_{split}2014_{coco_image_id}.jpg"
    record.image = relative_path_to_image
    return record

if __name__ == "__main__":
    with open('/net/acadia10a/data/zkhan/advqa/v1_OpenEnded_mscoco_val2017_advqa_questions.json', 'r') as f:
        val_questions_raw = json.load(f)

    logger.info('Converting %d records', len(val_questions_raw['questions']))
    val_questions = [
        make_val_record(schemas.QuestionRecord.parse_obj(_)) for _ in val_questions_raw['questions']
    ]

    logger.info('Verifying all image paths are correct.') 
    val_records = [point_record_to_coco_image_file(r, 'val') for r in tqdm(val_questions)]
    for record in tqdm(val_records):
        assert (IMAGES_ROOT / record.image).exists()
    

    with open(ADVQA_ROOT / 'val.json', 'w') as f:
        json.dump([r.dict() for r in val_records], f)
    logger.info('Wrote %d validation records to %s', len(val_records), ADVQA_ROOT / 'val.json')