#!/bin/bash

OUTPUT_DIR=cache/teacher_weights
python -m torch.distributed.run --master_port=37770 --nproc_per_node=4 train_vqg.py \
    --config=configs/aokvqg.yaml \
    --output_dir=$OUTPUT_DIR \
    --overrides batch_size=64  wandb=True