#!/bin/bash
python train.py \
  --data_dir /sessions/jolly-cool-einstein/data/pipette_well_dataset \
  --labels /sessions/jolly-cool-einstein/data/pipette_well_dataset/labels.json \
  --epochs 5 \
  --batch_size 2 \
  --num_frames 4 \
  --val_split 0.2 \
  --output checkpoints/ \
  --device cpu \
  --backbone resnet18 \
  2>&1 | tee training.log
