"""
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
"""
import argparse
from inspect import Attribute
import os

# import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf

from models.blip_vqa import blip_vqa
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from data.vqa_dataset import vqa_collate_fn
from data.utils import save_result
import cli


def train(model, data_loader, optimizer, epoch, device, wandb_logger=None):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "loss", utils.SmoothedValue(window_size=1, fmt="{value:.4f}")
    )

    header = "Train Epoch: [{}]".format(epoch)
    print_freq = 50

    for i, (image, question, answer, weights, n) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        image, weights = image.to(device, non_blocking=True), weights.to(
            device, non_blocking=True
        )

        loss = model(image, question, answer, train=True, n=n, weights=weights)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if i % print_freq == 0:
            if utils.is_main_process() and wandb_logger:
                wandb_logger.log(
                    data={
                        "loss": loss.item(),
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {
        k: "{:.3f}".format(meter.global_avg)
        for k, meter in metric_logger.meters.items()
    }


@torch.no_grad()
def evaluation(model, data_loader, device, config):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Generate VQA test result:"
    print_freq = 50

    result = []

    if config["inference"] == "rank":
        answer_list = data_loader.dataset.answer_list
        answer_candidates = model.tokenizer(
            answer_list, padding="longest", return_tensors="pt"
        ).to(device)
        answer_candidates.input_ids[:, 0] = model.tokenizer.bos_token_id

    for n, (image, question, question_id) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        image = image.to(device, non_blocking=True)

        # We'll only collect these when doing rank inference for now.

        if config["inference"] == "generate":
            answers = model(image, question, train=False, inference="generate")

            for answer, ques_id in zip(answers, question_id):
                # ques_id can be either a one-element Tensor[int]  or a
                # string. We convert it to an int (not sure why), but
                # this means we have to handle each case separately.
                try:
                    ques_id = int(ques_id.item())
                except AttributeError:
                    ques_id = int(ques_id)
                result.append({"question_id": ques_id, "answer": answer})

        elif config["inference"] == "rank":
            answer_ids, answer_scores = model(
                image,
                question,
                answer_candidates,
                train=False,
                inference="rank",
                k_test=config["k_test"],
                return_scores=True,
            )
            for ques_id, answer_id, answer_score in zip(
                question_id, answer_ids, answer_scores
            ):
                # The question id is a string in some datasets (VQAv2)
                # and an integer in other datasets (A-OKVQA). When it's a
                # an integer, it gets type casted to a tensor, which we
                # can't serialize to JSON without converting it back to an int.
                try:
                    ques_id = int(ques_id.item())
                except AttributeError:
                    pass

                result.append(
                    {
                        "question_id": ques_id,
                        "answer": answer_list[answer_id],
                        "score": answer_score.item(),
                    }
                )

    return result


def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    if utils.is_main_process() and config.wandb:
        print("Is main process, creating W&B logger.")
        wandb_logger = wandb.init(
            project="mithril-alice-valley",
            entity="zakh",
            config=OmegaConf.to_container(config),
        )
    else:
        wandb_logger = None

    #### Dataset ####
    print("Creating vqa datasets")
    datasets = create_dataset(config.dataset_name, config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)
    else:
        samplers = [None, None]

    train_loader, test_loader = create_loader(
        datasets,
        samplers,
        batch_size=[config["batch_size_train"], config["batch_size_test"]],
        num_workers=[4, 4],
        is_trains=[True, False],
        collate_fns=[vqa_collate_fn, None],
    )
    #### Model ####
    print("Creating model")
    model = blip_vqa(
        pretrained=config["pretrained"],
        image_size=config["image_size"],
        vit=config["vit"],
        vit_grad_ckpt=config["vit_grad_ckpt"],
        vit_ckpt_layer=config["vit_ckpt_layer"],
    )

    model = model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config["init_lr"],
        weight_decay=config["weight_decay"],
    )

    best = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()
    epochs = list(range(0, config.max_epoch))
    for epoch in epochs:
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            cosine_lr_schedule(
                optimizer,
                epoch,
                config["max_epoch"],
                config["init_lr"],
                config["min_lr"],
            )

            train_stats = train(
                model, train_loader, optimizer, epoch, device, wandb_logger=wandb_logger
            )

        else:
            break

        if utils.is_main_process():
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
            }
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

            if config.save_last_only:
                should_save = epoch == epochs[-1]
            else:
                should_save = True

            if should_save:
                save_obj = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                    "epoch": epoch,
                }

                torch.save(
                    save_obj,
                    os.path.join(args.output_dir, "checkpoint_%02d.pth" % epoch),
                )

        dist.barrier()

    vqa_result = evaluation(model_without_ddp, test_loader, device, config)
    result_file = save_result(vqa_result, args.result_dir, "vqa_result")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    args, config = cli.parse_args(default_config_path="./configs/vqa.yaml")
    cli.setup(args, config)
    main(args, config)
