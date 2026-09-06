#!/bin/bash

# Multi-seed training script for reproducible results

DATA_DIR="/path/to/MetaFood3D"
EXCEL_PATH="/path/to/MetaFood3D/_MetaFood3D_new_complete_dataset_nutrition_v2.xlsx"
OUTPUT_BASE="./outputs"

SEEDS=(7 13 2023)

echo "Multi-Seed Training for PortionNet"
echo "Training with seeds: ${SEEDS[@]}"

for SEED in "${SEEDS[@]}"
do
  echo ""
  echo "Training with seed: $SEED"
  
  OUTPUT_DIR="${OUTPUT_BASE}/seed${SEED}"
  mkdir -p $OUTPUT_DIR
  
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
  
  echo "Completed training with seed: $SEED"
done

echo ""
echo "All training completed!"
echo "Results saved in: $OUTPUT_BASE"
