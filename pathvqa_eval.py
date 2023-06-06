"""
For AQUA (Art VQA), we don't use the VQAv2 evaluation code. 
That's because the VQAv2 evaluation code assumes there are
multiple answers for each question, but in AQUA, there's only
one answer for each question. We just do an exact match evaluation
following the AQUA paper.
"""
import json
from unittest import result
from tqdm import tqdm
import json
from pprint import PrettyPrinter
from vqa_eval_tools import VQA, VQAEval
from argparse import ArgumentParser
from pathlib import Path
import schemas
import pandas as pd


pp = PrettyPrinter()

annotation_file =  '/net/acadia4a/data/zkhan/pathvqa/test.json'



def exact_match_eval(annotation_file, result_file):
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    with open(result_file, 'r') as f:
        results = json.load(f)

    annotations = [schemas.MinimalEvaluationRecord.parse_obj(a) for a in annotations]

    annotation_lookup_table = {a.question_id: a for a in annotations}
    evaluation_records = []
    for answer_record in results:
        ground_truth = annotation_lookup_table[answer_record['question_id']]
        # It's a list, but there's only one answer for each VQA art question.
        # So we just take the first one and do an exact match.
        true_answer = ground_truth.answer
        is_correct = answer_record['answer'] == true_answer
        question_type = ground_truth.question_type
        evaluation_records.append({
            'question_id': answer_record['question_id'],
            'answer': answer_record['answer'],
            'question_type': question_type,
            'is_correct': is_correct,
            'true_answer': true_answer,
            'answer_type': ground_truth.answer_type,
        })
    
    frame = pd.DataFrame(evaluation_records)
    answertype_groupby = frame.groupby('answer_type').apply(lambda s: s['is_correct'].sum() / len(s)).to_frame()
    accuracies = {
        'overall': frame['is_correct'].sum() / len(frame),
    }
    accuracies = {**accuracies, **{_: float(answertype_groupby.loc[_]) for _ in answertype_groupby.index}}
    accuracies = {k: round(v, 4) for k, v in accuracies.items()}
    return accuracies


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("result_file", help="Path to a JSON result file generated by an evaluation.")
    args = parser.parse_args()

    results_file = args.result_file

    accuracies = exact_match_eval(annotation_file, results_file)

    pp.pprint(accuracies)
    with open(Path(results_file).parent / 'pathvqa_eval.json', 'w') as f:
        json.dump(accuracies, f)