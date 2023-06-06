from omegaconf import OmegaConf, DictConfig
import hydra
from argparse import ArgumentParser, Namespace
from pathlib import Path
import torch
import os

# import ruamel.yaml as yaml
from typing import Tuple


def make_parser(default_config_path: str) -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--config", default=default_config_path)
    parser.add_argument(
        "--output_dir", default="/net/acadia4a/data/zkhan/mithril/sandbox"
    )
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument("--distributed", default=True, type=bool)
    parser.add_argument("--overrides", nargs="+", default=[])
    return parser


def load_config(args: Namespace) -> DictConfig:
    config_path = Path(args.config)
    with hydra.initialize(config_path=str(config_path.parent), version_base=None):
        config = hydra.compose(config_name=config_path.stem, overrides=args.overrides)
    return config


def parse_args(default_config_path: str) -> Tuple[Namespace, DictConfig]:
    """
    Parse command line arguments and config.yaml file.

    Args:
        default_config_path (str): This config will be the default if the user
            doesn't provide a different config.
    Returns:
        args (Namespace): Parsed arguments from command line.
        config (DictConfig): The parsed config.
    """
    parser = make_parser(default_config_path)
    args = parser.parse_args()
    config = load_config(args)
    return args, config


def setup(args: Namespace, config: DictConfig) -> None:
    """Do housekeeping needed in general before training.

    Args:
        args (Namespace): Parsed arguments from command line.
        config (DictConfig): The config produced by hydra.
    """
    if config.torch_home:
        torch.hub.set_dir(config.torch_home)
    args.result_dir = os.path.join(args.output_dir, "result")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    OmegaConf.save(config, os.path.join(args.output_dir, "config.yaml"))
    # yaml.dump(config, open(os.path.join(args.output_dir, "config.yaml"), "w"))
