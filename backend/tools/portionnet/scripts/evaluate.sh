#!/bin/bash

# Evaluation script for PortionNet

DATA_DIR="/path/to/MetaFood3D"
EXCEL_PATH="/path/to/MetaFood3D/_MetaFood3D_new_complete_dataset_nutrition_v2.xlsx"
CHECKPOINT="./outputs/best_model_seed7.pt"
OUTPUT_FILE="./results_evaluation.json"

echo "Evaluating PortionNet"
echo "Data directory: $DATA_DIR"
echo "Checkpoint: $CHECKPOINT"

echo ""
echo "Evaluating in RGB-only mode..."
python src/evaluate.py \
  --data_dir $DATA_DIR \
  --excel_path $EXCEL_PATH \
  --checkpoint $CHECKPOINT \
  --mode rgb_only \
  --batch_size 16 \
  --num_workers 4 \
  --output_file "${OUTPUT_FILE%.json}_rgb.json"

echo ""
echo "Evaluating in multimodal mode..."
python src/evaluate.py \
  --data_dir $DATA_DIR \
  --excel_path $EXCEL_PATH \
  --checkpoint $CHECKPOINT \
  --mode multimodal \
  --batch_size 16 \
  --num_workers 4 \
  --output_file "${OUTPUT_FILE%.json}_multimodal.json"

echo ""
echo "Evaluation completed!"
echo "Results saved to:"
echo "  - ${OUTPUT_FILE%.json}_rgb.json"
echo "  - ${OUTPUT_FILE%.json}_multimodal.json"
