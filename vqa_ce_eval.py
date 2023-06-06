import json
from tqdm.notebook import tqdm
import json
from pprint import PrettyPrinter
from vqa_eval_tools import VQA, VQAEval
from argparse import ArgumentParser
from pathlib import Path


pp = PrettyPrinter()

annotation_file = '/net/acadia4a/data/zkhan/vqa-counterexamples/counterexamples_annotations.json'
question_file = '/net/acadia4a/data/zkhan/vqa-counterexamples/counterexamples_questions.json'


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("result_file", help="Path to a JSON result file generated by an evaluation.")
    args = parser.parse_args()

    results_file = args.result_file

    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    advqa_obj = VQA(
    annotation_file=annotation_file,
    question_file=question_file
    )

    # We have to convert the question_id field to be an integer >.<
    with open(results_file, 'r') as f:
        predicted = json.load(f)

    for element in predicted:
        element['question_id'] = int(element['question_id'])
        
    with open(results_file, 'w') as f:
        json.dump(predicted, f)

    result_obj = advqa_obj.loadRes(
    resFile=results_file,
    quesFile=question_file
    )

    advqa_eval = VQAEval(advqa_obj, result_obj, n=2)
    advqa_eval.evaluate()
    print(f"Completed evaluation of {results_file}")
    with open(Path(results_file).parent / 'vqa_ce-eval.json', 'w') as f:
        json.dump(advqa_eval.accuracy, f)
    advqa_eval.accuracy.pop('perQuestionType')
    pp.pprint(advqa_eval.accuracy)
