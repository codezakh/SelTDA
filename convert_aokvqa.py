import cli
import json
import attrs
from typing import List, Dict, Optional
import cattrs
from omegaconf import DictConfig
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s -  %(name)s: %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
    )
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

SPLIT_TO_ORIGINAL_ANNOTATION_NAMES = {
    'test': 'aokvqa_v1p0_test.json',
    'train': 'aokvqa_v1p0_train.json',
    'val': 'aokvqa_v1p0_val.json',
} 


SPLIT_TO_NEW_ANNOTATION_NAMES = {
    'train': 'train.json',
    'val': 'val.json',
    'test': 'test.json'
}

IMAGES_ROOT = Path('/net/acadia10a/data/zkhan/coco2017')

@attrs.define
class VQAV2Record:
    dataset: str
    image: str
    question: str
    question_id: str # Really an int, but I think we can use a string.
    answer: Optional[List[str]] = None
    rationales: Optional[List[str]] = None
        
    @classmethod
    def from_dict(cls, the_dict: Dict):
        return cattrs.structure(the_dict, cls)


@attrs.define
class AOKVQARecord:
    difficult_direct_answer: bool
    image_id: int
    question: str
    question_id: str
    split: str
    choices: Optional[List[str]] = None
    rationales: Optional[List[str]] = None
    correct_choice_idx: Optional[int] = None
    direct_answers: Optional[List[str]] = None
    
    @classmethod
    def from_dict(cls, the_dict: Dict):
        return cattrs.structure(the_dict, cls)

def convert_aokvqa_to_vqav2(record: AOKVQARecord) -> VQAV2Record:
    return VQAV2Record(
        answer=record.direct_answers,
        dataset='aokvqa',
        image=record.image_id,
        question=record.question,
        question_id=record.question_id,
        rationales=record.rationales
    )

def convert_coco_id_to_coco_name(coco_id: int, prefix='') -> str:
    return f"{prefix}{coco_id:012}.jpg"

def point_record_to_coco_image_file(record: AOKVQARecord, coco_image_root: Path, split: str) -> None:
    coco_filename = convert_coco_id_to_coco_name(record.image)
    record.image = f'coco-images/{coco_filename}'


def load_split(split: str, aokvqa_root: Path) -> List[AOKVQARecord]:
    annotation_filename = SPLIT_TO_ORIGINAL_ANNOTATION_NAMES[split]
    with open(aokvqa_root / annotation_filename, 'r') as f:
        annotations = json.load(f)
    
    records = [AOKVQARecord.from_dict(ann) for ann in annotations]
    logger.info('Loaded %d records from %s', len(records), annotation_filename)
    return records


def serialize_records(records: List[AOKVQARecord], annotation_root: Path, split: str) -> None:
    records_as_dict = [attrs.asdict(record) for record in records]
    output_filename = annotation_root / SPLIT_TO_NEW_ANNOTATION_NAMES[split]
    if output_filename.exists():
        logger.info('Overwriting existing annotations %s', output_filename)
    with open(output_filename, 'w') as f:
        json.dump(records_as_dict, f)
    logger.info('Wrote %d records to %s', len(records_as_dict), output_filename)


def save_answer_list_as_json(annotation_root: Path):
    # For multiple choice answering, methods usually use a list
    # of common answers and then select the answer to a question
    # by ranking them. This is easier than directly generating the
    # the answer. AOKVQA provides this list as specialized_vocab_train.csv
    # but our code needs it to be JSON. It's a simple transformation.
    
    with open(annotation_root / 'specialized_vocab_train.csv', 'r') as f:
        words = [_.strip() for _ in f.readlines()]
    
    with open(annotation_root / 'answer_list.json', 'w') as f:
        json.dump(words, f)





def main(config: DictConfig) -> None:
    for split in ('train', 'val', 'test'):
        logger.info('Processing split %s', split)
        records = load_split(split, Path(config.ann_root))
        records = [convert_aokvqa_to_vqav2(_) for _ in records]
        logger.info('Converted %d records to VQAv2 format', len(records))
        logger.info('Verifying images exist')
        for record in records:
            point_record_to_coco_image_file(record, Path(config.vqa_root), split)
            try:
                assert (IMAGES_ROOT / record.image).exists()
            except:
                import ipdb; ipdb.set_trace()
        serialize_records(records, Path(config.ann_root), split)
    save_answer_list_as_json(Path(config.ann_root))





if __name__ == "__main__":
    args, config = cli.parse_args(default_config_path='./configs/aokvqa.yaml')
    main(config)