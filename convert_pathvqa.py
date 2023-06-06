
import json
from pathlib import Path
from omegaconf import OmegaConf
from dataclasses import dataclass
import schemas
from pydantic import BaseModel, validator
from tqdm import tqdm
import logging
from typing import Tuple, List, Union, Dict

PATHVQA_ROOT = Path('/net/acadia4a/data/zkhan/pathvqa')


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s -  %(name)s: %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
    )
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def load_json(_path: Path):
    with open(_path, 'r') as f:
        return json.load(f)

def write_json(_path: Path, data: dict):
    with open(_path, 'w') as f:
        json.dump(data, f)



# For evaluting this code, use an exact match evaluation.
# There's only one answer per question in PathVQA, so you
# can't use the VQAv2 evaluation code.
@dataclass
class OutputAnnotations:
    train: Path = PATHVQA_ROOT / 'train.json'
    val: Path = PATHVQA_ROOT / 'val.json'
    test: Path = PATHVQA_ROOT / 'test.json'
    answer_list: Path = PATHVQA_ROOT / 'answer_list.json'
    # This can be easily used to generate questions on images unused during training.
    test_val_combined: Path = PATHVQA_ROOT / 'test_val_combined.json'

@dataclass
class Config:
    raw_annotations: Path = PATHVQA_ROOT / 'all_data.json'
    output_annotations: OutputAnnotations = OutputAnnotations()
    pathvqa_images_dir: Path = Path('/net/acadia4a/data/zkhan/pathvqa/images')

class PathVQARecord_SuffixQA(BaseModel):
    image: str
    question: str
    answer: str

class PathVQARecord_SuffixVQA(BaseModel):
    answer_type: str
    img_id: str
    label: Dict[str, int]
    question_id: int
    question_type: str
    sent: str

class PathVQADump(BaseModel):
    test_qa: List[PathVQARecord_SuffixQA]
    test_vqa: List[PathVQARecord_SuffixVQA]
    train_qa: List[PathVQARecord_SuffixQA]
    train_vqa: List[PathVQARecord_SuffixVQA]
    val_qa: List[PathVQARecord_SuffixQA]
    val_vqa: List[PathVQARecord_SuffixVQA]

def convert_pathvqa_record_to_train_record(record_suffixqa: PathVQARecord_SuffixQA, record_suffixvqa: PathVQARecord_SuffixVQA) -> schemas.TrainingRecord:
    return schemas.TrainingRecord(
        question=record_suffixqa.question,
        answer=[record_suffixqa.answer],
        image=record_suffixqa.image,
        dataset='pathvqa',
        question_id=record_suffixvqa.question_id,
    )

def convert_pathvqa_record_to_evaluation_record(record_suffixqa: PathVQARecord_SuffixQA, record_suffixvqa: PathVQARecord_SuffixVQA) -> schemas.MinimalEvaluationRecord:
    return schemas.MinimalEvaluationRecord(
        question=record_suffixqa.question,
        answer=record_suffixqa.answer,
        image=record_suffixqa.image,
        dataset='pathvqa',
        question_id=record_suffixvqa.question_id,
        question_type=record_suffixvqa.question_type,
        answer_type=record_suffixvqa.answer_type,
    )


def make_training_records(pathvqa_dump: PathVQADump) -> List[schemas.TrainingRecord]:
    training_records = []
    for record_suffixqa, record_suffixvqa in zip(pathvqa_dump.train_qa, pathvqa_dump.train_vqa):
        training_records.append(convert_pathvqa_record_to_train_record(record_suffixqa, record_suffixvqa))
    return training_records

def make_validation_records(pathvqa_dump: PathVQADump) -> List[schemas.MinimalEvaluationRecord]:
    validation_records = []
    for record_suffixqa, record_suffixvqa in zip(pathvqa_dump.val_qa, pathvqa_dump.val_vqa):
        validation_records.append(convert_pathvqa_record_to_evaluation_record(record_suffixqa, record_suffixvqa))
    return validation_records

def make_testing_records(pathvqa_dump: PathVQADump) -> List[schemas.MinimalEvaluationRecord]:
    testing_records = []
    for record_suffixqa, record_suffixvqa in zip(pathvqa_dump.test_qa, pathvqa_dump.test_vqa):
        testing_records.append(convert_pathvqa_record_to_evaluation_record(record_suffixqa, record_suffixvqa))
    return testing_records


def redirect_image_and_verify(image_dir: Path, record: Union[schemas.TrainingRecord, schemas.MinimalEvaluationRecord]) -> None:
    image_name = record.image 
    split, *_ = image_name.split('_')

    path = f'{split}/{image_name}.jpg'

    record.image = path 
    try:
        assert (image_dir / record.image).exists(), f'Image {path} does not exist'
    except AssertionError:
        import ipdb; ipdb.set_trace()


def make_answer_list(test_records: List[schemas.MinimalEvaluationRecord]) -> List[str]:
    answer_list = []
    for record in test_records:
        answer_list.append(record.answer)
    return list(set(answer_list))

def filter_to_unique_images(records: List[schemas.TrainingRecord]) -> List[schemas.TrainingRecord]:
    unique_images = set()
    filtered_records = []
    for record in records:
        if record.image not in unique_images:
            unique_images.add(record.image)
            filtered_records.append(record)
    return filtered_records
    


if __name__ == '__main__':
    conf: Config = OmegaConf.structured(Config)
    pathvqa_dump = PathVQADump.parse_obj(load_json(conf.raw_annotations))
    training_records = make_training_records(pathvqa_dump)
    logger.info('Made %d training records', len(training_records))
    validation_records = make_validation_records(pathvqa_dump)
    logger.info('Made %d validation records', len(validation_records))
    testing_records = make_testing_records(pathvqa_dump)
    logger.info('Made %d testing records', len(testing_records))
    logger.info('Verifying all records')
    for record in tqdm(training_records + validation_records + testing_records):
        redirect_image_and_verify(conf.pathvqa_images_dir, record)

    answer_list = make_answer_list(testing_records)
    logger.info('Made answer list with %d answers', len(answer_list))

    write_json(conf.output_annotations.train, [record.dict() for record in training_records])
    write_json(conf.output_annotations.val, [record.dict() for record in validation_records])
    write_json(conf.output_annotations.test, [record.dict() for record in testing_records])
    write_json(conf.output_annotations.answer_list, answer_list)
    write_json(conf.output_annotations.test_val_combined, [record.dict() for record in filter_to_unique_images(testing_records + validation_records)])



