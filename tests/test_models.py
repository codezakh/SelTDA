from pathlib import Path
from omegaconf import OmegaConf
import torch
from models.blip import decoder_from_config
import pytest


@pytest.mark.slow
def test_decoder_forward_pass(request):
    config = OmegaConf.load(Path(request.config.rootdir) / "configs" / "vqg.yaml")
    # Don't load a checkpoint, this just makes it faster.
    config.pretrained = None
    torch.hub.set_dir(config.torch_home)
    model = decoder_from_config(config)
    # Create a fake image.
    image = torch.rand(1, 3, config["image_size"], config["image_size"])
    # Create a fake text.
    text = ["Do you like horses?"]
    model(image, text)


@pytest.mark.slow
def test_tokenization_length_respected(request):
    config = OmegaConf.load(Path(request.config.rootdir) / "configs" / "vqg.yaml")
    config.tokenizer_max_length = 10
    # Don't load a checkpoint, this just makes it faster.
    config.pretrained = None
    torch.hub.set_dir(config.torch_home)

    model = decoder_from_config(config)

    # Make a string much longer than the tokenizer maximum length
    # and check to make sure it gets chopped down by the tokenizer.
    text = "x " * (config.tokenizer_max_length**2)

    tokenized = model.tokenize(text)

    assert len(tokenized.input_ids.squeeze()) <= config.tokenizer_max_length
