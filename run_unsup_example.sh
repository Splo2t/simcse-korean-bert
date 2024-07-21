#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.

CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_name_or_path kykim/bert-kor-base \
    --train_file news.txt \
    --output_dir result/my-unsup-simcse-bert \
    --num_train_epochs 3 \
    --per_device_train_batch_size 32 \
    --learning_rate 2.5e-5 \
    --max_seq_length 300 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 25 \
    --preprocessing_num_workers 48 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --logging_steps 5 \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"
