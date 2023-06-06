import json
from pathlib import Path
from omegaconf import OmegaConf
from dataclasses import dataclass
import schemas
from pydantic import BaseModel, validator
from tqdm import tqdm
import logging
from typing import Tuple, List, Union

AQUA_ROOT = Path("/net/acadia4a/data/zkhan/vqa-art")

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s -  %(name)s: %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def load_json(_path: Path):
    with open(_path, "r") as f:
        return json.load(f)


def write_json(_path: Path, data: dict):
    with open(_path, "w") as f:
        json.dump(data, f)


@dataclass
class RawAnnotations:
    train: Path = AQUA_ROOT / "raw_annotations" / "train.json"
    val: Path = AQUA_ROOT / "raw_annotations" / "val.json"
    test: Path = AQUA_ROOT / "raw_annotations" / "test.json"


@dataclass
class OutputAnnotations:
    train: Path = AQUA_ROOT / "train.json"
    train_no_external_knowledge_questions: Path = (
        AQUA_ROOT / "train_no_external_knowledge_questions.json"
    )
    val: Path = AQUA_ROOT / "val.json"
    test: Path = AQUA_ROOT / "test.json"
    val_annotations: Path = AQUA_ROOT / "val_annotations.json"
    val_questions: Path = AQUA_ROOT / "val_questions.json"
    test_annotations: Path = AQUA_ROOT / "test_annotations.json"
    test_questions: Path = AQUA_ROOT / "test_questions.json"
    answer_list: Path = AQUA_ROOT / "answer_list.json"


@dataclass
class Config:
    raw_annotations: RawAnnotations = RawAnnotations()
    output_annotations: OutputAnnotations = OutputAnnotations()
    semart_images_dir: Path = Path("/net/acadia4a/data/zkhan/SemArt/Images")


class AquaRecord(BaseModel):
    image: str
    question: str
    answer: str
    need_external_knowledge: bool

    @validator("question")
    def normalize(cls, v):
        # The original question does not have a question mark at the end, which is different from
        # every other VQA dataset. We add the question mark here to make it consistent.
        return f"{v}?"


def convert_aqua_record_to_train_record(
    record: AquaRecord, question_id: int
) -> schemas.TrainingRecord:
    return schemas.TrainingRecord(
        question=record.question,
        answer=[record.answer],
        image=record.image,
        dataset="aqua",
        question_id=question_id,
    )


def convert_aqua_record_to_test_record(
    record: AquaRecord, question_id: int
) -> schemas.TestingRecord:
    return schemas.TestingRecord(
        question=record.question,
        image=record.image,
        dataset="aqua",
        question_id=question_id,
    )


def convert_aqua_record_to_question_record(
    record: AquaRecord, question_id: int
) -> schemas.QuestionRecord:
    return schemas.QuestionRecord(
        question=record.question,
        image_id=record.image,
        question_id=question_id,
    )


def convert_aqua_record_to_annotation_record(
    record: AquaRecord, question_id: int
) -> schemas.VQAAnnotationRecord:
    return schemas.VQAAnnotationRecord(
        question_type="external knowledge"
        if record.need_external_knowledge
        else "no external knowledge",
        answers=[
            schemas.VQAAnnotationSubRecord(
                answer=record.answer, answer_confidence="yes", answer_id=0
            )
        ],
        image_id=record.image,
        answer_type="yes/no" if record.answer in ("yes", "no") else "other",
        question_id=question_id,
    )


def make_answer_list(records: List[AquaRecord]) -> List[str]:
    answers = set()
    for record in records:
        answers.add(record.answer)
    return list(answers)


def verify_image_exists_for_record(image_name: str, semart_images_dir: Path) -> bool:
    return (semart_images_dir / image_name).exists()


if __name__ == "__main__":
    conf: Config = OmegaConf.structured(Config)
    train_records_raw = [
        AquaRecord.parse_obj(_) for _ in tqdm(load_json(conf.raw_annotations.train))
    ]
    train_question_ids = [i for i in range(len(train_records_raw))]
    logger.info("Loaded %d raw train records", len(train_records_raw))
    train_records = [
        convert_aqua_record_to_train_record(_, i)
        for i, _ in enumerate(train_records_raw)
    ]
    logger.info(
        "Converted all %d raw train records to train_vqa format", len(train_records)
    )

    train_records_no_external_knowledge_questions = [
        convert_aqua_record_to_train_record(record, i)
        for i, record in zip(train_question_ids, train_records_raw)
        if not record.need_external_knowledge
    ]

    val_records_raw = [
        AquaRecord.parse_obj(_) for _ in tqdm(load_json(conf.raw_annotations.val))
    ]
    val_question_ids = [
        i + train_question_ids[-1] + 1 for i in range(len(val_records_raw))
    ]
    logger.info(
        "Loaded %d raw val records. Converting to train_vqa format",
        len(val_records_raw),
    )
    val_records = [
        convert_aqua_record_to_test_record(_, i)
        for i, _ in zip(val_question_ids, val_records_raw)
    ]

    test_records_raw = [
        AquaRecord.parse_obj(_) for _ in tqdm(load_json(conf.raw_annotations.test))
    ]
    logger.info(
        "Loaded %d raw test records. Converting to train_vqa format",
        len(test_records_raw),
    )
    test_question_ids = [
        i + val_question_ids[-1] + 1 for i in range(len(test_records_raw))
    ]
    test_records = [
        convert_aqua_record_to_test_record(_, i)
        for i, _ in zip(test_question_ids, test_records_raw)
    ]

    logger.info("Converting test/val records to vqa_eval_tools format")
    test_annotations = [
        convert_aqua_record_to_annotation_record(_, i)
        for i, _ in tqdm(zip(test_question_ids, test_records_raw))
    ]
    test_questions = [
        convert_aqua_record_to_question_record(_, i)
        for i, _ in zip(test_question_ids, test_records_raw)
    ]
    val_annotations = [
        convert_aqua_record_to_annotation_record(_, i)
        for i, _ in tqdm(zip(val_question_ids, val_records_raw))
    ]
    val_questions = [
        convert_aqua_record_to_question_record(_, i)
        for i, _ in zip(val_question_ids, val_records_raw)
    ]

    logger.info("Verifying that all images exist")
    for r in tqdm(set(_.image for _ in train_records + val_records + test_records)):
        assert verify_image_exists_for_record(r, conf.semart_images_dir)

    logger.info("Writing train/val/test records to disk")
    write_json(conf.output_annotations.train, [_.dict() for _ in train_records])
    write_json(conf.output_annotations.val, [_.dict() for _ in val_records])
    write_json(conf.output_annotations.test, [_.dict() for _ in test_records])
    write_json(
        conf.output_annotations.train_no_external_knowledge_questions,
        [_.dict() for _ in train_records_no_external_knowledge_questions],
    )

    logger.info("Writing test/val records to vqa_eval_tools format")
    write_json(
        conf.output_annotations.test_annotations,
        {"annotations": [_.dict() for _ in test_annotations]},
    )
    write_json(
        conf.output_annotations.test_questions,
        {"questions": [_.dict() for _ in test_questions]},
    )

    write_json(
        conf.output_annotations.val_annotations,
        {"annotations": [_.dict() for _ in val_annotations]},
    )
    write_json(
        conf.output_annotations.val_questions,
        {"questions": [_.dict() for _ in val_questions]},
    )

    logger.info("Making answer list")
    answer_list = make_answer_list(test_records_raw)
    logger.info("Made answer list with %d answers", len(answer_list))
    write_json(conf.output_annotations.answer_list, answer_list)
    logger.info("Done")
    logger.info("Wrote all files to %s", AQUA_ROOT)
