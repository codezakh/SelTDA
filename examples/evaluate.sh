#!/bin/bash

# NOTE: This script shows how you can evaluate a list of checkpoints on a dataset.
# In this case, the dataset is ArtVQA. You can change this to work for _any_ dataset
# by changing the config file and the evaluation script (artvqa_eval.py).

parentdir_name() {
    echo "$(basename "$(dirname "$1")")"
}

# Control the number of GPUs and GPU usage by setting
# CUDA_VISIBLE_DEVICES.

# FYI, this will clobber results when two models have the same name.

zero_shot_artvqa_generate_evaluate () {
    local CHECKPOINT=$1
    local OUTPUT_DIR=cache/artvqa_evals_$(parentdir_name $CHECKPOINT)
    echo -e "\e[1;34m\e[4mGenerating 0-shot ArtVQA results for $CHECKPOINT\e[0m"
    python -m torch.distributed.run --nproc_per_node=4 train_vqa.py \
    --output_dir=$OUTPUT_DIR --evaluate \
    --config configs/artvqa.yaml \
    --overrides wandb=false \
    pretrained=$CHECKPOINT \
    batch_size_test=16
    echo -e "\e[1;32mResults are in $OUTPUT_DIR\e[0m"
    # We don't control the name of the result file, it is currently
    # harcoded to be result/vqa_result.json.
    python artvqa_eval.py $OUTPUT_DIR/result/vqa_result.json
}


MODELS_TO_EVAL=(
    # Compare synthetic vs baseline models with VQAV2 post-training and A-OKVQA finetuning.
    /net/acadia10a/data/zkhan/mithril/aokvqa_finetuned/blip_vqa_baseline/checkpoint_09.pth
    /net/acadia10a/data/zkhan/mithril/aokvqa_finetuned/33-finetune-on-aokvqa-synth_17k/checkpoint_09.pth
    /net/acadia4a/data/zkhan/mithril/aokvqa_finetuned/33-finetune-on-aokvqa-synth_34k/checkpoint_09.pth
    /net/acadia4a/data/zkhan/mithril/aokvqa_finetuned/33-finetune-on-aokvqa-synth_51k/checkpoint_09.pth
    /net/acadia4a/data/zkhan/mithril/aokvqa_finetuned/34-finetune-on-aokvqa-synth_4k/checkpoint_09.pth
    /net/acadia4a/data/zkhan/mithril/aokvqa_finetuned/34-finetune-on-aokvqa-synth_8k/checkpoint_09.pth

    # Compare synthetic vs baseline models from A-OKVQA only.
    /net/acadia10a/data/zkhan/mithril/aokvqa_finetuned/blip/checkpoint_09.pth
    /net/acadia10a/data/zkhan/mithril/aokvqa_finetuned/25-aokvqa_synth_rationale_17k/checkpoint_09.pth
    /net/acadia10a/data/zkhan/mithril/aokvqa_finetuned/25-aokvqa_synth_rationale_34k/checkpoint_09.pth
    /net/acadia10a/data/zkhan/mithril/aokvqa_finetuned/25-aokvqa_synth_rationale_51k/checkpoint_09.pth
)


for PATH_TO_CHECKPOINT in ${MODELS_TO_EVAL[@]}; do
    zero_shot_artvqa_generate_evaluate $PATH_TO_CHECKPOINT
done