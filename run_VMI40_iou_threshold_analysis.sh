#!/bin/bash

set -euo pipefail

ANALYSIS_ROOT="/home/nohel/DATA/MultipleMyeloma_analyses"
ANALYSIS_NAME="longi_summary_all"

PRED_ROOT="/home/nohel/DATA/nnUNet_results/predict_new/full_models"
TRAINED_MODEL_ROOT="/home/nohel/DATA/nnUNet_results"
GT_ROOT="/home/nohel/DATA/nnUNet_raw/MM_GT_DATA"

MODEL_NAME="Dataset710_MM_Lesion_seg_just_VMI_40"
TRAIN_MODEL_PATH="$TRAINED_MODEL_ROOT/$MODEL_NAME"
PRED_MODEL_DIR="$PRED_ROOT/$MODEL_NAME"

OUTPUT_ROOT="$ANALYSIS_ROOT/iou_threshold_analysis_vmi40/$ANALYSIS_NAME"
SUMMARY_FILE="$OUTPUT_ROOT/vmi40_iou_threshold_summary.csv"

mkdir -p "$OUTPUT_ROOT"

TRAINER_DIR=$(find "$TRAIN_MODEL_PATH" -maxdepth 1 -type d -name "nnUNetTrainer*" | head -n 1)

if [[ ! -d "$TRAINER_DIR" ]]; then
    echo "Missing trainer directory: $TRAIN_MODEL_PATH" >&2
    exit 1
fi

PLANS_FILE="$TRAINER_DIR/plans.json"
DATASET_FILE="$TRAINER_DIR/dataset.json"

[[ -f "$PLANS_FILE" ]] || { echo "Missing: $PLANS_FILE" >&2; exit 1; }
[[ -f "$DATASET_FILE" ]] || { echo "Missing: $DATASET_FILE" >&2; exit 1; }
[[ -d "$PRED_MODEL_DIR" ]] || { echo "Missing: $PRED_MODEL_DIR" >&2; exit 1; }

printf 'iou_threshold,Dice_mean,Dice_std,F1_mean,F1_std,NSD_mean,NSD_std,n_folds\n' > "$SUMMARY_FILE"

for IOU_THRESHOLD in 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90; do
    THRESHOLD_LABEL=${IOU_THRESHOLD//./_}
    THRESHOLD_OUTPUT_DIR="$OUTPUT_ROOT/iou_threshold_$THRESHOLD_LABEL"
    MODEL_OUT_DIR="$THRESHOLD_OUTPUT_DIR/$MODEL_NAME"
    mkdir -p "$MODEL_OUT_DIR"

    echo "======================================"
    echo "Model: $MODEL_NAME"
    echo "IoU threshold: $IOU_THRESHOLD"
    echo "======================================"

    for FOLD in 0 1 2 3 4; do
        PRED_DIR="$PRED_MODEL_DIR/fold_$FOLD"
        GT_DIR="$GT_ROOT/labelsTr_fold_$FOLD"
        OUT_FILE="$MODEL_OUT_DIR/${ANALYSIS_NAME}_fold_$FOLD.json"

        [[ -d "$PRED_DIR" ]] || { echo "Skipping missing predictions: $PRED_DIR"; continue; }
        [[ -d "$GT_DIR" ]] || { echo "Skipping missing ground truth: $GT_DIR"; continue; }

        LongiSeg_evaluate_folder \
            "$GT_DIR" \
            "$PRED_DIR" \
            -pfile "$PLANS_FILE" \
            -djfile "$DATASET_FILE" \
            -iou_threshold "$IOU_THRESHOLD" \
            -o "$OUT_FILE"
    done

    IOU_THRESHOLD="$IOU_THRESHOLD" MODEL_OUT_DIR="$MODEL_OUT_DIR" SUMMARY_FILE="$SUMMARY_FILE" python3 <<'PY'
import json
import os

import numpy as np

model_dir = os.environ["MODEL_OUT_DIR"]
summary_file = os.environ["SUMMARY_FILE"]
iou_threshold = float(os.environ["IOU_THRESHOLD"])

metric_values = {"Dice": [], "F1": [], "NSD": []}

for fold in range(5):
    result_file = os.path.join(model_dir, f"longi_summary_all_fold_{fold}.json")
    if not os.path.exists(result_file):
        continue

    with open(result_file) as file:
        result = json.load(file)

    foreground_mean = result["foreground_mean"]
    for metric in metric_values:
        metric_values[metric].append(foreground_mean[metric])

if not metric_values["Dice"]:
    raise SystemExit(f"No fold results found for threshold {iou_threshold}")

summary_values = []
for metric in ("Dice", "F1", "NSD"):
    summary_values.extend([
        f"{np.mean(metric_values[metric]):.6f}",
        f"{np.std(metric_values[metric]):.6f}",
    ])

with open(summary_file, "a") as file:
    file.write(",".join([
        f"{iou_threshold:.2f}",
        *summary_values,
        str(len(metric_values["Dice"])),
    ]) + "\n")
PY
done

echo "Saved: $SUMMARY_FILE"
