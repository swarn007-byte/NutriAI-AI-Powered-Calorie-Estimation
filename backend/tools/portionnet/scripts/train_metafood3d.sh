#!/bin/bash

# Quick start script for training PortionNet on MetaFood3D

DATA_DIR="/path/to/MetaFood3D"
EXCEL_PATH="/path/to/MetaFood3D/_MetaFood3D_new_complete_dataset_nutrition_v2.xlsx"
OUTPUT_DIR="./outputs"
SEED=7

mkdir -p $OUTPUT_DIR

echo "Training PortionNet"
echo "Data directory: $DATA_DIR"
echo "Excel path: $EXCEL_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Seed: $SEED"

python src/train.py \
  --data_dir $DATA_DIR \
  --excel_path $EXCEL_PATH \
  --output_dir $OUTPUT_DIR \
  --epochs 25 \
  --batch_size 16 \
  --lr_backbone 1e-4 \
  --lr_head 5e-4 \
  --lambda_cls 1.0 \
  --lambda_reg 0.1 \
  --lambda_distill 0.5 \
  --rgb_only_ratio 0.3 \
  --num_workers 4 \
  --seed $SEED

echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR/best_model_seed${SEED}.pt"
