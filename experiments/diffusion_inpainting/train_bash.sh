#!/bin/bash

python train_pipeline.py \
  --train_jsonl outputs/inpaint_ALB_surname/train_2.jsonl \
  --out_dir outputs/lora_2 \
  --epochs 100 --batch 1 \
  --lr 5e-5 --rank 4 --alpha 4 --dropout 0.1 \
  --mixed_precision

python train_pipeline.py \
  --train_jsonl outputs/inpaint_ALB_surname/train_20.jsonl \
  --out_dir outputs/lora_20 \
  --epochs 40 --batch 2 \
  --lr 1e-4 --rank 8 --alpha 8 --dropout 0.05 \
  --mixed_precision
python train_pipeline.py \
  --train_jsonl outputs/inpaint_ALB_surname/train_40.jsonl \
  --out_dir outputs/lora_40 \
  --epochs 25 --batch 4 \
  --lr 1e-4 --rank 16 --alpha 16 --dropout 0.0 \
  --mixed_precision

python infer_pipeline.py 2
python infer_pipeline.py 20
python infer_pipeline.py 40


