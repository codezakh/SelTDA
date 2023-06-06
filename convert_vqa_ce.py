"""
This script constructs the VQA-CE dataset from the VQAv2 dataset.
The VQA-CE dataset is a slice of the VQAv2 validation set which was
constructed so that models which have learned shortcuts to answer
questions will perform poorly. The dataset does not require retraining
a model, and is evaluation only.
"""

import json
from pathlib import Path
from typing import List
from pydantic import BaseModel
import schemas
from tqdm import tqdm
import logging
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


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


VQA_V2_ROOT = Path('/net/acadia10a/data/zkhan/vqav2_annotations')
PATH_RAW_ANNOTATIONS = VQA_V2_ROOT / 'raw_val_annotations_coco_val2014.json'
PATH_RAW_QUESTIONS = VQA_V2_ROOT / 'raw_val_questions_coco_val2014.json'
PATH_VAL_RECORDS_WHOLE = VQA_V2_ROOT / 'vqa_val.json'

VQA_CE_ROOT = Path('/net/acadia4a/data/zkhan/vqa-counterexamples')
HARD_SLICE_PATH = VQA_CE_ROOT / 'hard.json'
COUNTEREXAMPLE_SLICE_PATH = VQA_CE_ROOT / 'counterexamples.json'

class Slice(BaseModel):
    questions: List[schemas.QuestionRecord]
    annotations: List[schemas.VQAAnnotationRecord]
    testing_records: List[schemas.TestingRecord]

    @classmethod
    def build_from_whole_slice_given_question_ids(cls, qids: List[int], whole_slice: 'Slice') -> 'Slice':
        qids = set(qids)
        questions = [q for q in whole_slice.questions if q.question_id in qids]
        annotations = [a for a in whole_slice.annotations if a.question_id in qids]
        testing_records = [t for t in whole_slice.testing_records if t.question_id in qids] 
        import ipdb; ipdb.set_trace()
        # For some reason, one of the testing records is missing from the annotations.
        # Not going to track it down, will just drop it from the questions and annotations.
        try:
            assert len(questions) == len(annotations) == len(testing_records)
        except AssertionError:
            logger.warning(f'Number of questions, annotations, and testing records do not match.')
            logger.warning(f'Questions: {len(questions)}')
            logger.warning(f'Annotations: {len(annotations)}')
            logger.warning(f'Testing records: {len(testing_records)}')
            qids_missing_from_testing_records = set([q.question_id for q in questions]) - set([t.question_id for t in testing_records])
            logger.info(f'Dropping question IDs missing from testing records: {qids_missing_from_testing_records}')
            questions = [q for q in questions if q.question_id not in qids_missing_from_testing_records]
            annotations = [a for a in annotations if a.question_id not in qids_missing_from_testing_records]
        return cls(
            questions=questions,
            annotations=annotations,
            testing_records=testing_records,
        )

    def serialize_slice(self, prefix: str):
        with open(VQA_CE_ROOT / f'{prefix}_questions.json', 'w') as f:
            json.dump({'questions': [_.dict() for _ in tqdm(self.questions)]}, f)

        with open(VQA_CE_ROOT / f'{prefix}_annotations.json', 'w') as f:
            json.dump({'annotations': [_.dict() for _ in tqdm(self.annotations)]}, f)

        with open(VQA_CE_ROOT / f'{prefix}_testing_records.json', 'w') as f:
            json.dump([_.dict() for _ in tqdm(self.testing_records)], f)


if __name__  == "__main__":
    logger.info('Loading raw annotations and questions.')
    raw_annotations = load_json(PATH_RAW_ANNOTATIONS)
    raw_questions = load_json(PATH_RAW_QUESTIONS)
    val_records_whole = load_json(PATH_VAL_RECORDS_WHOLE)
    logger.info('Loaded %d raw annotations and %d raw questions.', len(raw_annotations), len(raw_questions))

    logger.info('Building easy slice.')
    hard_slice_qids = load_json(HARD_SLICE_PATH)
    counterexample_slice_qids = load_json(COUNTEREXAMPLE_SLICE_PATH)
    all_question_ids = set([q['question_id'] for q in raw_questions['questions']])
    easy_slice = all_question_ids - set(counterexample_slice_qids).union(set(hard_slice_qids))

    logger.info('Building whole slice.')
    whole_slice = Slice(
        questions=[schemas.QuestionRecord.parse_obj(q) for q in tqdm(raw_questions['questions'])],
        annotations=[schemas.VQAAnnotationRecord.parse_obj(a) for a in tqdm(raw_annotations['annotations'])],
        testing_records=[schemas.TestingRecord.parse_obj(t) for t in tqdm(val_records_whole)],
    )

    hard_slice = Slice.build_from_whole_slice_given_question_ids(hard_slice_qids, whole_slice)
    logger.info('Built hard slice with %d questions.', len(hard_slice.questions))
    counterexample_slice = Slice.build_from_whole_slice_given_question_ids(counterexample_slice_qids, whole_slice)
    logger.info('Built counterexample slice with %d questions.', len(counterexample_slice.questions))
    easy_slice = Slice.build_from_whole_slice_given_question_ids(easy_slice, whole_slice)
    logger.info('Built easy slice with %d questions.', len(easy_slice.questions))

    logger.info('Serializing slices.')
    hard_slice.serialize_slice('hard')
    counterexample_slice.serialize_slice('counterexamples')
    easy_slice.serialize_slice('easy')

    logger.info('Copying answer list from VQAv2 to VQA-CE.')
    shutil.copyfile(VQA_V2_ROOT / 'answer_list.json', VQA_CE_ROOT / 'answer_list.json')

    # For ease of use, we copy counterexamples_testing_records.json to val.json, because
    # as of now, all the datasets are hardcoded to load the validation split from a file
    # called val.json.
    logger.info('Copying counterexamples_testing_records.json to val.json.')
    shutil.copyfile(VQA_CE_ROOT / 'counterexamples_testing_records.json', VQA_CE_ROOT / 'val.json')