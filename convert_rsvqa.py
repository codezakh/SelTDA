import json
from pathlib import Path
from omegaconf import OmegaConf
from dataclasses import dataclass
import schemas
from pydantic import BaseModel, validator, ValidationError
from tqdm import tqdm
import logging
from typing import Tuple, List, Union, Dict, Optional

# Remote Sensing VQA is made up of two datasets. One is
# low resolution, the other is high resolution. They have different
# numbers of images, and probably different types and numbers of questions.
# We handle each separately.
RSVQA_LR_ROOT = Path('/net/acadia4a/data/zkhan/rsvqa/low_resolution')


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



@dataclass
class LowResOutputAnnotations:
    train: Path = RSVQA_LR_ROOT / 'train.json'
    val: Path = RSVQA_LR_ROOT / 'val.json'
    test: Path = RSVQA_LR_ROOT / 'test.json'
    # Write the MinimalEvaluationRecord (s) to this file for the scoring.
    test_annotations: Path = RSVQA_LR_ROOT / 'test_annotations.json'
    answer_list: Path = RSVQA_LR_ROOT / 'answer_list.json'

@dataclass
class LowResRawAnnotations:
    all_answers: Path = RSVQA_LR_ROOT / 'all_answers.json'
    all_questions: Path = RSVQA_LR_ROOT / 'all_questions.json'
    train_answers: Path = RSVQA_LR_ROOT / 'LR_split_train_answers.json'
    train_questions: Path = RSVQA_LR_ROOT / 'LR_split_train_questions.json'


@dataclass
class LowResConfig:
    raw_annotations: LowResRawAnnotations = LowResRawAnnotations()
    output_annotations: LowResOutputAnnotations = LowResOutputAnnotations()
    # The images are named like 0.tif, 1.tif, etc.
    low_res_images_dir: Path = Path('/net/acadia4a/data/zkhan/rsvqa/low_resolution/Images_LR')

class LRAnswerRecord(BaseModel):
    id: int
    date_added: float
    question_id: int
    people_id: int
    answer: str
    active: bool

class LRQuestionRecord(BaseModel):
    id: int
    date_added: float
    img_id: int
    people_id: int
    type: str
    question: str
    answers_ids: List[int]
    active: bool


class RsVqaLrDump(BaseModel):
    all_answers: List[LRAnswerRecord]
    all_questions: List[LRQuestionRecord]
    # The train answers and questions are the same length as all_questions and
    # all_answers, but have null fields for questions which are not in the train 
    # set. That is how we identify what is in the training split and what is in the
    # test split.
    train_answers: List[Dict] 
    train_questions: List[Dict] 

    @classmethod
    def from_cfg(cls, cfg: LowResConfig):
        return cls(
            all_answers=load_json(cfg.raw_annotations.all_answers)['answers'],
            all_questions=load_json(cfg.raw_annotations.all_questions)['questions'],
            train_answers=load_json(cfg.raw_annotations.train_answers)['answers'],
            train_questions=load_json(cfg.raw_annotations.train_questions)['questions']
        )

class RsVqaLrSplit(BaseModel):
    answers: List[LRAnswerRecord]
    questions: List[LRQuestionRecord]

class StandardFormatSplit(BaseModel):
    training_records: List[schemas.TrainingRecord]
    testing_records: List[schemas.MinimalEvaluationRecord]

def make_splits(rsvqa_lr_dump: RsVqaLrDump) -> Tuple[RsVqaLrSplit]:
    logger.info('Making splits...')
    indices = [i for i in range(len(rsvqa_lr_dump.all_answers))]
    is_test = []
    for a in rsvqa_lr_dump.train_answers:
        is_test.append(a.get('answer') is None)

    train_indices = [i for i, test in zip(indices, is_test) if not test]
    test_indices = [i for i, test in zip(indices, is_test) if test]

    logger.info('Number of train questions: %d', len(train_indices))
    logger.info('Number of test questions: %d', len(test_indices))

    # Pull the records belonging to the train split from all_answers and all_questions.
    train_answers = [rsvqa_lr_dump.all_answers[i] for i in train_indices]
    train_questions = [rsvqa_lr_dump.all_questions[i] for i in train_indices]

    # Pull the records belonging to the test split from all_answers and all_questions.
    test_answers = [rsvqa_lr_dump.all_answers[i] for i in test_indices]
    test_questions = [rsvqa_lr_dump.all_questions[i] for i in test_indices]

    train_split = RsVqaLrSplit(answers=train_answers, questions=train_questions)
    test_split = RsVqaLrSplit(answers=test_answers, questions=test_questions)

    # Sanity check that each answer belongs to each question.
    logger.info('Sanity checking each answer and question are paired correctly')
    for a, q in zip(train_split.answers, train_split.questions):
        assert a.answer == rsvqa_lr_dump.all_answers[q.answers_ids[0]].answer

    for a,q in zip(test_split.answers, test_split.questions):
        assert a.answer == rsvqa_lr_dump.all_answers[q.answers_ids[0]].answer
    # Check to make sure each question only has one answer. In other VQA datasets,
    # questions can have multiple answers. From my brief perusal, RSVQA does not, but
    # let's double check that assumption.
    for q in rsvqa_lr_dump.all_questions:
        assert len(q.answers_ids) == 1
    logger.info('Sanity check passed')

    return train_split, test_split


def convert_rsvqa_lr_record_tuple_to_vqa_format(question: LRQuestionRecord, answer: LRAnswerRecord) -> Tuple[schemas.TrainingRecord, schemas.MinimalEvaluationRecord]:
    training_record = schemas.TrainingRecord(
        dataset='rsvqa_lr',
        image=question.img_id,
        question_id=question.id,
        question=question.question,
        answer=[answer.answer]
    )

    try:
        evaluation_record = schemas.MinimalEvaluationRecord(
            question=question.question,
            dataset='rsvqa_lr',
            image=question.img_id,
            question_id=question.id,
            answer=answer.answer,
            question_type=question.type,
            answer_type='default'
        )
    except ValidationError:
        import ipdb; ipdb.set_trace()

    return training_record, evaluation_record


def transform_split_into_std_format(split: RsVqaLrSplit) -> StandardFormatSplit:
    training_records = []
    evaluation_records = []
    for q, a in zip(split.questions, split.answers):
        training_record, evaluation_record = convert_rsvqa_lr_record_tuple_to_vqa_format(q, a)
        training_records.append(training_record)
        evaluation_records.append(evaluation_record)

    return StandardFormatSplit(training_records=training_records, testing_records=evaluation_records)


def redirect_and_verify_image(record: schemas.TrainingRecord, cfg: LowResConfig):
    image_dir = cfg.low_res_images_dir

    record.image = f'{record.image}.tif'
    assert (image_dir / record.image).exists()


def make_answer_list(records: List[schemas.TrainingRecord]) -> List[str]:
    answers = []
    for record in records:
        answers.extend(record.answer)
    return list(set(answers))




if __name__ == "__main__":
    lr_config: LowResConfig = OmegaConf.structured(LowResConfig)
    lr_dump: RsVqaLrDump = RsVqaLrDump.from_cfg(lr_config)

    logger.info('Loaded %d answers and %d questions', len(lr_dump.all_answers), len(lr_dump.all_questions))
    logger.info('Training questions: %d', len(lr_dump.train_questions))
    logger.info('Training answers: %d', len(lr_dump.train_answers))

    lr_train, lr_test = make_splits(lr_dump)

    logger.info('Converting splits into standard format')
    lr_train = transform_split_into_std_format(lr_train)
    lr_test = transform_split_into_std_format(lr_test)

    logger.info('Train split in standard format: %d', len(lr_train.training_records))
    logger.info('Test split in standard format: %d', len(lr_test.testing_records))

    logger.info('Verifying images in training split exist')
    for record in tqdm(lr_train.training_records):
        redirect_and_verify_image(record, lr_config)

    logger.info('Verifying images in test split exist')
    for record in tqdm(lr_test.training_records):
        redirect_and_verify_image(record, lr_config)
    for record in tqdm(lr_test.testing_records):
        redirect_and_verify_image(record, lr_config)

    logger.info('Saving splits to disk')
    write_json(lr_config.output_annotations.train, [_.dict() for _ in lr_train.training_records])
    write_json(lr_config.output_annotations.test, [_.dict() for _ in lr_test.training_records])
    write_json(lr_config.output_annotations.test_annotations , [_.dict() for _ in lr_test.testing_records])


    logger.info('Making answer list')
    answer_list = make_answer_list(lr_test.training_records)
    logger.info('Answer list has %d answers', len(answer_list))
    write_json(lr_config.output_annotations.answer_list, answer_list)
