#!/bin/bash

set -euo pipefail

# =========================
# COMMON SETTINGS
# =========================

ANALYSIS_ROOT="/home/nohel/DATA/MultipleMyeloma_analyses2"
ANALYSIS_NAME="longi_summary_all"
IOU_THRESHOLD=0.1

# =========================
# PREDICTION ROOTS
# =========================

FULL_PRED_ROOT="/home/nohel/DATA/nnUNet_results/predict_new/full_models"
ZERO_PRED_ROOT="/home/nohel/DATA/nnUNet_results/predict_new/zero_input_models"

# =========================
# TRAINED MODELS (metadata)
# =========================

TRAINED_MODEL_ROOT="/home/nohel/DATA/nnUNet_results"

# =========================
# GT
# =========================

GT_ROOT="/home/nohel/DATA/nnUNet_raw/MM_GT_DATA"

# =========================
# DATASET WHITELIST
# =========================

FULL_DATASETS=(
Dataset700_MM_Lesion_seg_Leave_One_Out_without_ConvCT
Dataset701_MM_Lesion_seg_Leave_One_Out_without_VMI_40
Dataset702_MM_Lesion_seg_Leave_One_Out_without_VMI_80
Dataset703_MM_Lesion_seg_Leave_One_Out_without_VMI_120
Dataset704_MM_Lesion_seg_Leave_One_Out_without_CaSupp_25
Dataset705_MM_Lesion_seg_Leave_One_Out_without_CaSupp_50
Dataset706_MM_Lesion_seg_Leave_One_Out_without_CaSupp_75
Dataset707_MM_Lesion_seg_Leave_One_Out_without_CaSupp_100

Dataset708_MM_Lesion_seg_all_together
Dataset709_MM_Lesion_seg_just_ConvCT
Dataset710_MM_Lesion_seg_just_VMI_40
Dataset711_MM_Lesion_seg_just_VMI_80
Dataset712_MM_Lesion_seg_just_VMI_120
Dataset713_MM_Lesion_seg_just_CaSupp_25
Dataset714_MM_Lesion_seg_just_CaSupp_50
Dataset715_MM_Lesion_seg_just_CaSupp_75
Dataset716_MM_Lesion_seg_just_CaSupp_100

Dataset717_MM_Lesion_seg_all_VMI
Dataset718_MM_Lesion_seg_all_CaSupp
)

ZERO_DATASETS=(
Dataset708_MM_Lesion_seg_all_together_zero_input_channel_convCT
Dataset708_MM_Lesion_seg_all_together_zero_input_channel_VMI40
Dataset708_MM_Lesion_seg_all_together_zero_input_channel_VMI80
Dataset708_MM_Lesion_seg_all_together_zero_input_channel_VMI120
Dataset708_MM_Lesion_seg_all_together_zero_input_channel_CaSupp_25
Dataset708_MM_Lesion_seg_all_together_zero_input_channel_CaSupp_50
Dataset708_MM_Lesion_seg_all_together_zero_input_channel_CaSupp_75
Dataset708_MM_Lesion_seg_all_together_zero_input_channel_CaSupp_100
Dataset717_MM_Lesion_seg_all_VMI_zero_input_channel_VMI40
Dataset717_MM_Lesion_seg_all_VMI_zero_input_channel_VMI80
Dataset717_MM_Lesion_seg_all_VMI_zero_input_channel_VMI120
Dataset718_MM_Lesion_seg_all_CaSupp_zero_input_channel_CaSupp_25
Dataset718_MM_Lesion_seg_all_CaSupp_zero_input_channel_CaSupp_50
Dataset718_MM_Lesion_seg_all_CaSupp_zero_input_channel_CaSupp_75
Dataset718_MM_Lesion_seg_all_CaSupp_zero_input_channel_CaSupp_100
)

# =========================
# FUNCTION
# =========================

evaluate_models () {

MODEL_PRED_ROOT=$1
OUTPUT_SUBFOLDER=$2
shift 2
DATASET_LIST=("$@")

ANALYSIS_DIR="$ANALYSIS_ROOT/$OUTPUT_SUBFOLDER/$ANALYSIS_NAME"
mkdir -p "$ANALYSIS_DIR"

echo "======================================"
echo " Evaluating $OUTPUT_SUBFOLDER"
echo "======================================"

for MODEL_NAME in "${DATASET_LIST[@]}"; do

    PRED_MODEL_DIR="$MODEL_PRED_ROOT/$MODEL_NAME"
    [[ -d "$PRED_MODEL_DIR" ]] || continue

    # ====================================
    # find correct trained model
    # ====================================

    DATASET_ID=$(echo "$MODEL_NAME" | sed -E 's/Dataset([0-9]+).*/\1/')

    if [[ "$DATASET_ID" == "708" ]]; then
        TRAIN_MODEL="Dataset708_MM_Lesion_seg_all_together"
    elif [[ "$DATASET_ID" == "717" ]]; then
        TRAIN_MODEL="Dataset717_MM_Lesion_seg_all_VMI"
    elif [[ "$DATASET_ID" == "718" ]]; then
        TRAIN_MODEL="Dataset718_MM_Lesion_seg_all_CaSupp"
    else
        TRAIN_MODEL="$MODEL_NAME"
    fi

    TRAIN_MODEL_PATH="$TRAINED_MODEL_ROOT/$TRAIN_MODEL"

    TRAINER_DIR=$(find "$TRAIN_MODEL_PATH" -maxdepth 1 -type d -name "nnUNetTrainer*" | head -n 1)

    [[ -d "$TRAINER_DIR" ]] || continue

    PLANS_FILE="$TRAINER_DIR/plans.json"
    DATASET_FILE="$TRAINER_DIR/dataset.json"

    [[ -f "$PLANS_FILE" && -f "$DATASET_FILE" ]] || continue

    MODEL_OUT_DIR="$ANALYSIS_DIR/$MODEL_NAME"
    mkdir -p "$MODEL_OUT_DIR"

    echo "---- MODEL: $MODEL_NAME ----"

    # ====================================
    # FOLD LOOP
    # ====================================

    for FOLD in 0 1 2 3 4; do

        PRED_DIR="$PRED_MODEL_DIR/fold_$FOLD"
        GT_DIR="$GT_ROOT/labelsTr_fold_$FOLD"

        [[ -d "$PRED_DIR" && -d "$GT_DIR" ]] || continue

        OUT_FILE="$MODEL_OUT_DIR/${ANALYSIS_NAME}_fold_$FOLD.json"

        LongiSeg_evaluate_folder \
            "$GT_DIR" \
            "$PRED_DIR" \
            -pfile "$PLANS_FILE" \
            -djfile "$DATASET_FILE" \
            -iou_threshold "$IOU_THRESHOLD" \
            -o "$OUT_FILE"

    done

    # ====================================
    # AGGREGATION
    # ====================================

python3 <<EOF
import os, json, numpy as np

model_dir = "$MODEL_OUT_DIR"
analysis_name = "$ANALYSIS_NAME"

dice_vals=[]
f1_vals=[]
nsd_vals=[]

for fold in range(5):

    f=os.path.join(model_dir,f"{analysis_name}_fold_{fold}.json")

    if not os.path.exists(f):
        continue

    d=json.load(open(f))

    fm=d["foreground_mean"]

    dice_vals.append(fm["Dice"])
    f1_vals.append(fm["F1"])
    nsd_vals.append(fm["NSD"])

if dice_vals:

    summary={
        "Dice_mean":float(np.mean(dice_vals)),
        "Dice_std":float(np.std(dice_vals)),
        "F1_mean":float(np.mean(f1_vals)),
        "F1_std":float(np.std(f1_vals)),
        "NSD_mean":float(np.mean(nsd_vals)),
        "NSD_std":float(np.std(nsd_vals)),
        "n_folds":len(dice_vals)
    }

    out=os.path.join(model_dir,f"{analysis_name}_ALL_FOLDS.json")

    json.dump(summary,open(out,"w"),indent=4)

    print("Aggregated:",model_dir)
EOF

done

}

# =========================
# RUN
# =========================

evaluate_models "$FULL_PRED_ROOT" "full_models" "${FULL_DATASETS[@]}"

evaluate_models "$ZERO_PRED_ROOT" "zero_input_models" "${ZERO_DATASETS[@]}"

echo "======================================"
echo " DONE"
echo "======================================"
