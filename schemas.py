from pydantic import BaseModel
from typing import Optional, List, Dict, Union, NewType

SemArtImageName = NewType('SemArtImageName', str)

class TrainingRecord(BaseModel):
    dataset: str
    image: str
    question: str
    question_id: int
    answer: Optional[List[str]] = None
    rationales: Optional[List[str]] = None

class TestingRecord(BaseModel):
    question_id: int
    question: str
    image: str
    dataset: Optional[str]
    original_question_id: Optional[Union[int, str]]

# AOKVQA has string type question ids.
class AnswerRecord(BaseModel):
    question_id: Union[int,str]
    answer: str
    score: float

class QuestionRecord(BaseModel):
    # For datasets which use COCO images, the image_id is the COCO image id.
    # SemArt doesn't have image ids but names, so we use the names. 
    image_id: Union[int, SemArtImageName] 
    question: str
    question_id: int

class VQAAnnotationSubRecord(BaseModel):
    answer: str
    answer_confidence: str
    answer_id: int
        
class VQAAnnotationRecord(BaseModel):
    question_type: str
    answers: List[VQAAnnotationSubRecord]
    image_id: Union[int, SemArtImageName]
    answer_type: str
    question_id: int
    multiple_choice_answer: Optional[str] = None

# This can be used for VQA datasets that only have
# one ground truth answer per question, and so can't
# be used easily with the VQAv2 evaluation code.
class MinimalEvaluationRecord(BaseModel):
    question_id: int
    answer: str
    question: str
    image: Union[str, int]
    question_type: Optional[str] = None
    answer_type: Optional[str] = None
    dataset: Optional[str] = None