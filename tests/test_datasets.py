import pytest


@pytest.mark.slow
def test_vqa_dataset(request):
    from data import create_dataset
    from pathlib import Path
    from omegaconf import OmegaConf

    config = OmegaConf.load(Path(request.config.rootdir) / "configs" / "vqa.yaml")
    train, test = create_dataset("vqa", config)
    # Attempt to access an element of the dataset.
    train[0]


@pytest.mark.slow
def test_vqg_dataset(request):
    from data import create_dataset
    from pathlib import Path
    from omegaconf import OmegaConf

    config = OmegaConf.load(Path(request.config.rootdir) / "configs" / "vqg.yaml")
    train, _ = create_dataset("vqg", config)
    _, qa_pair, _ = train[0]
    assert "question" in qa_pair.lower() and "answer" in qa_pair.lower()


def test_aokvqa_dataset(request):
    from data import create_dataset
    from pathlib import Path
    from omegaconf import OmegaConf

    config = OmegaConf.load(Path(request.config.rootdir) / "configs" / "aokvqa.yaml")
    train, test = create_dataset("aokvqa", config)
    image, question, answer, weights = train[0]
    image, question, question_id = test[0]


def test_aokvqa_use_eval_as_test(request):
    from data import create_dataset
    from pathlib import Path
    from omegaconf import OmegaConf

    config = OmegaConf.load(Path(request.config.rootdir) / "configs" / "aokvqa.yaml")
    config.use_validation_set_as_test_set = True
    _, test = create_dataset("aokvqa", config)
    assert test.annotation[0]["answer"] is not None


def test_aokvqa_as_vqg_dataset(request):
    from data import create_dataset
    from pathlib import Path
    from omegaconf import OmegaConf

    config = OmegaConf.load(Path(request.config.rootdir) / "configs" / "aokvqg.yaml")
    train, test = create_dataset("vqg", config)
    # Just check that the getitem works.
    train[0]
    test[0]


def test_truncating_aokvqa_train_dataset(request):
    from data import create_dataset
    from pathlib import Path
    from omegaconf import OmegaConf

    config = OmegaConf.load(Path(request.config.rootdir) / "configs" / "aokvqa.yaml")
    config.truncate_train_dataset_to = 10
    train, _ = create_dataset("aokvqa", config)
    assert len(train) == 10


def test_truncating_vqg_dataset(request):
    from data import create_dataset
    from pathlib import Path
    from omegaconf import OmegaConf

    config = OmegaConf.load(Path(request.config.rootdir) / "configs" / "vqg.yaml")
    config.truncate_train_dataset_to = 10
    train, _ = create_dataset("vqg", config)
    assert len(train) == 10


def test_aokvqg_using_rationale(request):
    from data import create_dataset
    from pathlib import Path
    from omegaconf import OmegaConf

    config = OmegaConf.load(Path(request.config.rootdir) / "configs" / "aokvqg.yaml")
    config.use_rationale = True
    train, _ = create_dataset("vqg", config)

    first_train_sample = train[0]
    _, target, *_ = first_train_sample

    assert "rationale" in target.lower()


def test_aokvqg_using_rationale_reversed(request):
    from data import create_dataset
    from pathlib import Path
    from omegaconf import OmegaConf

    config = OmegaConf.load(Path(request.config.rootdir) / "configs" / "aokvqg.yaml")
    config.use_rationale = True
    config.generate_rationale_first = True
    train, _ = create_dataset("vqg", config)

    first_train_sample = train[0]
    _, target, *_ = first_train_sample

    assert target.lower().startswith("rationale")


def test_aokvqa_add_rationale_to_answer(request):
    from data import create_dataset
    from pathlib import Path
    from omegaconf import OmegaConf

    config = OmegaConf.load(Path(request.config.rootdir) / "configs" / "aokvqa.yaml")
    config.append_rationale_to_answer = True
    train, _ = create_dataset("aokvqa", config)

    first_train_sample = train[0]
    image, question, answer, weights = first_train_sample
    # Check that the rationale is in the answer.
    rationales = train.annotation[0]["rationales"]
    for rationale in rationales:
        # We append the all the rationales to each answer, but we'll just
        # check the first one.
        assert rationale in answer[0]


def test_aokvqa_add_rationale_to_question(request):
    """
    Useful for using rationales at test time.
    """
    from data import create_dataset
    from pathlib import Path
    from omegaconf import OmegaConf

    config = OmegaConf.load(Path(request.config.rootdir) / "configs" / "aokvqa.yaml")
    config.append_rationale_to_question = True
    train, _ = create_dataset("aokvqa", config)

    first_train_sample = train[0]
    image, question, answer, weights = first_train_sample
    # Check that the rationale is in the answer.
    rationales = train.annotation[0]["rationales"]
    for rationale in rationales:
        assert rationale in question


def test_aokvqa_use_rationale_as_answer(request):
    """
    This can be used for the rationale-generating model.
    """
    from data import create_dataset
    from pathlib import Path
    from omegaconf import OmegaConf

    config = OmegaConf.load(Path(request.config.rootdir) / "configs" / "aokvqa.yaml")
    config.use_rationale_as_answer = "separated"
    train, _ = create_dataset("aokvqa", config)

    first_train_sample = train[0]
    image, question, answer, weights = first_train_sample
    # We expect the answer to be a list of rationales, one
    # list item for each rationale.
    rationales = train.annotation[0]["rationales"]
    for rationale in rationales:
        assert rationale in answer


def test_aokvqa_use_concatenated_rationale_as_answer(request):
    from data import create_dataset
    from pathlib import Path
    from omegaconf import OmegaConf

    config = OmegaConf.load(Path(request.config.rootdir) / "configs" / "aokvqa.yaml")
    config.use_rationale_as_answer = "concatenated"
    train, _ = create_dataset("aokvqa", config)

    first_train_sample = train[0]
    image, question, answer, weights = first_train_sample
    # Check that the rationales ARE the answer.
    rationales = train.annotation[0]["rationales"]
    # Check that the "answer" is just a single item list, because
    # we concatenate the rationales together.'
    assert len(answer) == 1
    # Then check that the string contains all the rationales.
    for rationale in rationales:
        assert rationale in answer[0]


def test_okvqa_dataset(request):
    from data import create_dataset
    from pathlib import Path
    from omegaconf import OmegaConf

    config = OmegaConf.load(Path(request.config.rootdir) / "configs" / "okvqa.yaml")
    train, test = create_dataset("okvqa", config)
    # Just check that the getitem works.
    train[0]
    test[0]
    assert train.annotation[0]["dataset"] == "okvqa"
    assert test.annotation[0]["dataset"] == "okvqa"


def test_vqg_on_okvqa(request):
    from data import create_dataset
    from pathlib import Path
    from omegaconf import OmegaConf

    config = OmegaConf.load(Path(request.config.rootdir) / "configs" / "okvqg.yaml")
    train, _ = create_dataset("vqg", config)

    first_train_sample = train[0]
    _, target, *_ = first_train_sample


def test_artvqa_dataset(request):
    from data import create_dataset
    from pathlib import Path
    from omegaconf import OmegaConf

    config = OmegaConf.load(Path(request.config.rootdir) / "configs" / "artvqa.yaml")
    train, test = create_dataset("artvqa", config)
    # Just check that the getitem works.
    train[0]
    test[0]
    assert train.annotation[0]["dataset"] == "aqua"
    assert test.annotation[0]["dataset"] == "aqua"


def test_pathvqa_dataset(request):
    from data import create_dataset
    from pathlib import Path
    from omegaconf import OmegaConf

    config = OmegaConf.load(Path(request.config.rootdir) / "configs" / "pathvqa.yaml")
    train, test = create_dataset("pathvqa", config)
    # Just check that the getitem works.
    train[0]
    test[0]
    assert train.annotation[0]["dataset"] == "pathvqa"
    assert test.annotation[0]["dataset"] == "pathvqa"


def test_vqg_on_artvqa(request):
    from data import create_dataset
    from pathlib import Path
    from omegaconf import OmegaConf

    config = OmegaConf.load(Path(request.config.rootdir) / "configs" / "artvqg.yaml")
    train, _ = create_dataset("vqg", config)

    first_train_sample = train[0]
    _, target, *_ = first_train_sample


def test_vqa_rephrasings_dataset(request):
    from data import create_dataset
    from pathlib import Path
    from omegaconf import OmegaConf

    config = OmegaConf.load(
        Path(request.config.rootdir) / "configs" / "vqa_rephrasings.yaml"
    )
    train, test = create_dataset("vqa_rephrasings", config)
    # Just check that the getitem works.
    test[0]
    assert test.annotation[0]["dataset"] == "vqa-rephrasings"


def test_vqg_on_pathvqa(request):
    from data import create_dataset
    from pathlib import Path
    from omegaconf import OmegaConf

    config = OmegaConf.load(Path(request.config.rootdir) / "configs" / "pathvqg.yaml")
    train, _ = create_dataset("vqg", config)

    first_train_sample = train[0]
    _, target, *_ = first_train_sample


def test_rsvqa_lr_dataset(request):
    from data import create_dataset
    from pathlib import Path
    from omegaconf import OmegaConf

    config = OmegaConf.load(Path(request.config.rootdir) / "configs" / "rsvqa_lr.yaml")
    train, test = create_dataset("rsvqa", config)
    # Just check that the getitem works.
    train[0]
    test[0]
    assert train.annotation[0]["dataset"] == "rsvqa_lr"
    assert test.annotation[0]["dataset"] == "rsvqa_lr"


def test_specifying_validation_file_for_generic_ds(request):
    from data import create_dataset
    from pathlib import Path
    from omegaconf import OmegaConf

    config = OmegaConf.load(Path(request.config.rootdir) / "configs" / "generic.yaml")
    train, test = create_dataset("generic_vqa", config)
    # Just check that the getitem works.
    train[0]
    test[0]

    with pytest.raises(FileNotFoundError):
        config = OmegaConf.load(
            Path(request.config.rootdir) / "configs" / "generic.yaml"
        )
        config.val_file = "mhmmm jebyit heniop"
        train, test = create_dataset("generic_vqa", config)


def test_specifying_answer_list_for_generic_ds(request):
    from data import create_dataset
    from pathlib import Path
    from omegaconf import OmegaConf

    config = OmegaConf.load(Path(request.config.rootdir) / "configs" / "generic.yaml")
    config.answer_list = "answer_list"
    train, test = create_dataset("generic_vqa", config)
    # Just check that the getitem works.
    train[0]
    test[0]

    # Now change the answer list to some file that doesn't exist and make sure it fails.
    config = OmegaConf.load(Path(request.config.rootdir) / "configs" / "generic.yaml")
    config.answer_list = "mhmmm jebyit heniop"
    with pytest.raises(FileNotFoundError):
        train, test = create_dataset("generic_vqa", config)
