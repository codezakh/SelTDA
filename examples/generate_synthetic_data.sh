#!/bin/bash
# Store the synthetic data within the same folder you store the real data.
# We have to do it this way because of the dataset reader code.
OUTPUT_DIR= cache/aokvqa
TEACHER_WEIGHTS=cache/teacher_weights/checkpoint_04.pth
OUTPUT_ANNOTATIONS_NAME=synthetic_data.json
python generate_questions.py --config=configs/generate_questions_coco.yaml --overrides \
    max_length=40 \
    output_folder=$OUTPUT_DIR \
    pretrained=$TEACHER_WEIGHTS \
    questions_per_image=2 \
    output_annotations_name=$OUTPUT_ANNOTATIONS_NAME