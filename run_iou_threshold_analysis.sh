#!/bin/bash

set -euo pipefail

ANALYSIS_ROOT="/home/nohel/DATA/MultipleMyeloma_analyses"
ANALYSIS_NAME="longi_summary_all"

PRED_ROOT="/home/nohel/DATA/nnUNet_results/predict_new/full_models"
TRAINED_MODEL_ROOT="/home/nohel/DATA/nnUNet_results"
GT_ROOT="/home/nohel/DATA/nnUNet_raw/MM_GT_DATA"

OUTPUT_ROOT="$ANALYSIS_ROOT/iou_threshold_analysis_full_models/$ANALYSIS_NAME"
SUMMARY_FILE="$OUTPUT_ROOT/iou_threshold_f1_summary.csv"
PLOT_FILE="$OUTPUT_ROOT/iou_threshold_f1_plot.png"

MODELS=(
    "Dataset708_MM_Lesion_seg_all_together"
    "Dataset709_MM_Lesion_seg_just_ConvCT"
    "Dataset710_MM_Lesion_seg_just_VMI_40"
    "Dataset713_MM_Lesion_seg_just_CaSupp_25"
    "Dataset717_MM_Lesion_seg_all_VMI"
    "Dataset718_MM_Lesion_seg_all_CaSupp"
)

mkdir -p "$OUTPUT_ROOT"
printf 'dataset_id,model,iou_threshold,F1_mean,F1_std,n_folds\n' > "$SUMMARY_FILE"

for MODEL_NAME in "${MODELS[@]}"; do
    TRAIN_MODEL_PATH="$TRAINED_MODEL_ROOT/$MODEL_NAME"
    PRED_MODEL_DIR="$PRED_ROOT/$MODEL_NAME"
    TRAINER_DIR=$(find "$TRAIN_MODEL_PATH" -maxdepth 1 -type d -name "nnUNetTrainer*" | head -n 1)

    [[ -d "$TRAINER_DIR" ]] || { echo "Missing trainer directory: $TRAIN_MODEL_PATH" >&2; exit 1; }
    PLANS_FILE="$TRAINER_DIR/plans.json"
    DATASET_FILE="$TRAINER_DIR/dataset.json"
    [[ -f "$PLANS_FILE" && -f "$DATASET_FILE" ]] || { echo "Missing plans or dataset JSON for $MODEL_NAME" >&2; exit 1; }
    [[ -d "$PRED_MODEL_DIR" ]] || { echo "Missing predictions: $PRED_MODEL_DIR" >&2; exit 1; }

    DATASET_ID=$(echo "$MODEL_NAME" | sed -E 's/Dataset([0-9]+).*/\1/')

    for IOU_THRESHOLD in 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90; do
        THRESHOLD_LABEL=${IOU_THRESHOLD//./_}
        MODEL_OUT_DIR="$OUTPUT_ROOT/iou_threshold_$THRESHOLD_LABEL/$MODEL_NAME"
        mkdir -p "$MODEL_OUT_DIR"

        echo "Model: $MODEL_NAME | IoU threshold: $IOU_THRESHOLD"

        for FOLD in 0 1 2 3 4; do
            PRED_DIR="$PRED_MODEL_DIR/fold_$FOLD"
            GT_DIR="$GT_ROOT/labelsTr_fold_$FOLD"
            OUT_FILE="$MODEL_OUT_DIR/${ANALYSIS_NAME}_fold_$FOLD.json"

            [[ -d "$PRED_DIR" && -d "$GT_DIR" ]] || { echo "Skipping missing fold $FOLD for $MODEL_NAME"; continue; }

            LongiSeg_evaluate_folder \
                "$GT_DIR" \
                "$PRED_DIR" \
                -pfile "$PLANS_FILE" \
                -djfile "$DATASET_FILE" \
                -iou_threshold "$IOU_THRESHOLD" \
                -o "$OUT_FILE"
        done

        DATASET_ID="$DATASET_ID" MODEL_NAME="$MODEL_NAME" IOU_THRESHOLD="$IOU_THRESHOLD" MODEL_OUT_DIR="$MODEL_OUT_DIR" SUMMARY_FILE="$SUMMARY_FILE" python3 <<'PY'
import json
import os

import numpy as np

dataset_id = os.environ["DATASET_ID"]
model = os.environ["MODEL_NAME"]
iou_threshold = float(os.environ["IOU_THRESHOLD"])
model_dir = os.environ["MODEL_OUT_DIR"]
summary_file = os.environ["SUMMARY_FILE"]

f1_values = []
for fold in range(5):
    result_file = os.path.join(model_dir, f"longi_summary_all_fold_{fold}.json")
    if not os.path.exists(result_file):
        continue
    with open(result_file) as file:
        result = json.load(file)
    f1_values.append(result["foreground_mean"]["F1"])

if not f1_values:
    raise SystemExit(f"No fold results found for {model}, threshold {iou_threshold}")

with open(summary_file, "a") as file:
    file.write(",".join([
        dataset_id,
        model,
        f"{iou_threshold:.2f}",
        f"{np.mean(f1_values):.6f}",
        f"{np.std(f1_values):.6f}",
        str(len(f1_values)),
    ]) + "\n")
PY
    done

done

SUMMARY_FILE="$SUMMARY_FILE" PLOT_FILE="$PLOT_FILE" python3 <<'PY'
import os

import matplotlib.pyplot as plt
import pandas as pd

summary_file = os.environ["SUMMARY_FILE"]
plot_file = os.environ["PLOT_FILE"]
data = pd.read_csv(summary_file)

fig, ax = plt.subplots(figsize=(10, 6))
for model, model_data in data.groupby("model", sort=False):
    model_data = model_data.sort_values("iou_threshold")
    ax.errorbar(
        model_data["iou_threshold"],
        model_data["F1_mean"],
        yerr=model_data["F1_std"],
        marker="o",
        markersize=4,
        linewidth=1.5,
        capsize=3,
        label=model,
    )

ax.set_xlabel("IoU threshold")
ax.set_ylabel("F1 score")
ax.set_title("F1 score versus IoU threshold")
ax.set_xticks(sorted(data["iou_threshold"].unique()))
ax.set_ylim(0, 1)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.legend(loc="best", frameon=False)
fig.tight_layout()
fig.savefig(plot_file, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {plot_file}")
PY

echo "Saved: $SUMMARY_FILE"
