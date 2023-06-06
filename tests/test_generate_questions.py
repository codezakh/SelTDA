from generate_questions import VQARecord, AmbiguousBooleanAnswerError
import pytest


def test_parsing_rationale():
    raw_model_output = ": what does this animal live in?. answer : trees, forest, savanna, park, forest, woods, grassland, zoo, zoo, zoo. rationale : the animals in this photo are found in the savannah. there is a small group of animals that appear to have some unique features. the animal on the left seems to be an elephant or tiger."
    record = VQARecord.build_from_raw_model_output(
        raw_model_output, "/fake/path/to/image.jpg", parse_rationale=True
    )
    assert record.answer == [
        _.strip()
        for _ in "trees, forest, savanna, park, forest, woods, grassland, zoo, zoo, zoo".split(
            ","
        )
    ]
    assert record.question == "what does this animal live in?"
    assert (
        record.rationale
        == "the animals in this photo are found in the savannah. there is a small group of animals that appear to have some unique features. the animal on the left seems to be an elephant or tiger."
    )


def test_parsing_output_with_no_rationale():
    raw_model_output = ": what does this animal live in?. answer : trees, forest, savanna, park, forest, woods, grassland, zoo, zoo, zoo."
    record = VQARecord.build_from_raw_model_output(
        raw_model_output, "/fake/path/to/image.jpg", parse_rationale=False
    )
    assert record.rationale is None


def test_parsing_when_rationale_comes_first():
    """Because we have models that generate the rationale first."""
    raw_model_output = "rationale : the animals in this photo are found in the savannah. there is a small group of animals that appear to have some unique features. the animal on the left seems to be an elephant or tiger. question: what does this animal live in?. answer : trees, forest, savanna, park, forest, woods, grassland, zoo, zoo, zoo."
    record = VQARecord.build_from_raw_model_output(
        raw_model_output, "/fake/path/to/image.jpg", parse_rationale=True
    )
    assert record.answer == [
        _.strip()
        for _ in "trees, forest, savanna, park, forest, woods, grassland, zoo, zoo, zoo".split(
            ","
        )
    ]
    assert record.question == "what does this animal live in?"
    assert (
        record.rationale
        == "the animals in this photo are found in the savannah. there is a small group of animals that appear to have some unique features. the animal on the left seems to be an elephant or tiger."
    )


def test_parsing_question_when_no_colon_first():
    # Models which are very good _do not_ generate an extraneous colon at the start of generated
    # questions.
    raw_model_output = " what might the person next to the suitcase be doing?. answer : waiting, waiting, waiting, waiting, waiting, waiting, waiting, waiting, waiting, waiting"
    record = VQARecord.build_from_raw_model_output(
        raw_model_output, "/fake/path/to/image.jpg", parse_rationale=False
    )
    assert record.answer == [
        _.strip()
        for _ in "waiting, waiting, waiting, waiting, waiting, waiting, waiting, waiting, waiting, waiting".split(
            ","
        )
    ]
    assert record.question == "what might the person next to the suitcase be doing?"


def test_parsing_question_when_ambiguous_boolean_answer():
    raw_model_output = "is the person blue? answer: yes, no"
    with pytest.raises(AmbiguousBooleanAnswerError):
        record = VQARecord.build_from_raw_model_output(
            raw_model_output, "/fake/path/to/image.jpg", parse_rationale=False
        )
