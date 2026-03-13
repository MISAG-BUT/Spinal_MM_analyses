#!/bin/bash

set -e
shopt -s nullglob

############################################
# SETTINGS
############################################

ANALYSIS_ROOT="/home/nohel/DATA/MultipleMyeloma_analyses"
SPLITS_FILE="$ANALYSIS_ROOT/splits_final.json"

RAW_ROOT="/home/nohel/DATA/nnUNet_raw"
RESULTS_ROOT="/home/nohel/DATA/nnUNet_results"

ZERO_INPUT_BASE="$RAW_ROOT/MM_Data_for_zero_input_analysis"
FULL_INPUT_BASE="$RAW_ROOT/MM_Data_for_full_model_analysis"

PREDICT_ROOT="$RESULTS_ROOT/predict_new"
GPU_LOG_DIR="$ANALYSIS_ROOT/gpu_logs"

mkdir -p "$PREDICT_ROOT"
mkdir -p "$GPU_LOG_DIR"

TIME_LOG="$ANALYSIS_ROOT/inference_time_log.csv"

echo "MODEL,FOLD,START,END,DURATION_SEC,PARAMETERS" > "$TIME_LOG"

############################################
# FULL MODELS
############################################

FULL_MODELS=(

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

############################################
# ZERO INPUT DATASETS
############################################

datasets_708=(

Dataset708_MM_Lesion_seg_all_together_zero_input_channel_convCT
Dataset708_MM_Lesion_seg_all_together_zero_input_channel_VMI40
Dataset708_MM_Lesion_seg_all_together_zero_input_channel_VMI80
Dataset708_MM_Lesion_seg_all_together_zero_input_channel_VMI120
Dataset708_MM_Lesion_seg_all_together_zero_input_channel_CaSupp_25
Dataset708_MM_Lesion_seg_all_together_zero_input_channel_CaSupp_50
Dataset708_MM_Lesion_seg_all_together_zero_input_channel_CaSupp_75
Dataset708_MM_Lesion_seg_all_together_zero_input_channel_CaSupp_100

)

datasets_717=(

Dataset717_MM_Lesion_seg_all_VMI_zero_input_channel_VMI40
Dataset717_MM_Lesion_seg_all_VMI_zero_input_channel_VMI80
Dataset717_MM_Lesion_seg_all_VMI_zero_input_channel_VMI120

)

datasets_718=(

Dataset718_MM_Lesion_seg_all_CaSupp_zero_input_channel_CaSupp_25
Dataset718_MM_Lesion_seg_all_CaSupp_zero_input_channel_CaSupp_50
Dataset718_MM_Lesion_seg_all_CaSupp_zero_input_channel_CaSupp_75
Dataset718_MM_Lesion_seg_all_CaSupp_zero_input_channel_CaSupp_100

)

############################################
# GPU MONITOR
############################################

start_gpu_logging() {

LOGFILE=$1

nvidia-smi \
--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw \
--format=csv \
-l 1 > "$LOGFILE" &

GPU_LOG_PID=$!

}

stop_gpu_logging() {

kill $GPU_LOG_PID 2>/dev/null || true

}

############################################
# COUNT PARAMETERS
############################################

count_parameters() {

MODEL_PATH=$1
CHECKPOINT="$MODEL_PATH/fold_0/checkpoint_final.pth"

if [[ -f "$CHECKPOINT" ]]; then

python3 <<EOF
import torch
ckpt=torch.load("$CHECKPOINT",map_location="cpu",weights_only=False)
print(sum(v.numel() for v in ckpt["network_weights"].values()))
EOF

else
echo "N/A"
fi

}

############################################
# PREPARE FULL MODEL FOLDS
############################################

prepare_full_model_folds() {

MODEL=$1

IMAGE_DIR="$RAW_ROOT/$MODEL/imagesTr"
FOLD_BASE="$FULL_INPUT_BASE/$MODEL"

echo "Preparing folds for $MODEL..."

python3 <<EOF
import json, os, shutil

image_dir="$IMAGE_DIR"
fold_base="$FOLD_BASE"

with open("$SPLITS_FILE") as f:
    splits=json.load(f)

for fold_id, split in enumerate(splits):

    fold_dir=os.path.join(fold_base,f"fold_{fold_id}")
    os.makedirs(fold_dir,exist_ok=True)

    val_cases=split["val"]

    for case in val_cases:

        ch=0

        while True:

            src=os.path.join(image_dir,f"{case}_{ch:04d}.nii.gz")

            if not os.path.exists(src):
                break

            dst=os.path.join(fold_dir,os.path.basename(src))

            shutil.copy(src,dst)

            ch+=1
EOF

echo "Folds prepared in $FOLD_BASE"

}

############################################
# RUN FULL MODELS
############################################

run_full_models() {

for MODEL in "${FULL_MODELS[@]}"
do

echo "======================================"
echo "FULL MODEL: $MODEL"
echo "======================================"

prepare_full_model_folds "$MODEL"

MODEL_PATH="$RESULTS_ROOT/$MODEL/nnUNetTrainer__nnUNetPlans__3d_fullres"

PARAMS=$(count_parameters "$MODEL_PATH")

for FOLD in {0..4}
do

INPUT_DIR="$FULL_INPUT_BASE/$MODEL/fold_$FOLD"

OUTPUT_DIR="$PREDICT_ROOT/full_models/$MODEL/fold_$FOLD"

mkdir -p "$OUTPUT_DIR"

GPU_LOG_FILE="$GPU_LOG_DIR/${MODEL}_fold_${FOLD}.csv"

echo "Running fold $FOLD"

START=$(date +%s)
START_H=$(date "+%Y-%m-%d %H:%M:%S")

start_gpu_logging "$GPU_LOG_FILE"

nnUNetv2_predict_from_modelfolder \
-i "$INPUT_DIR" \
-o "$OUTPUT_DIR" \
-m "$MODEL_PATH" \
-f "$FOLD"

stop_gpu_logging

END=$(date +%s)
END_H=$(date "+%Y-%m-%d %H:%M:%S")

DURATION=$((END-START))

echo "$MODEL,$FOLD,$START_H,$END_H,$DURATION,$PARAMS" >> "$TIME_LOG"

done
done

}

############################################
# RUN ZERO INPUT DATASETS
############################################

run_zero_group() {

DATASET_ID=$1
shift

DATASETS=("$@")

for dataset in "${DATASETS[@]}"
do

echo "======================================"
echo "ZERO INPUT DATASET: $dataset"
echo "======================================"

PARAMS="NA"

for FOLD in {0..4}
do

INPUT="$ZERO_INPUT_BASE/$dataset/fold_${FOLD}/imagesTs"

OUTPUT="$PREDICT_ROOT/zero_input_models/$dataset/fold_${FOLD}"

mkdir -p "$OUTPUT"

GPU_LOG_FILE="$GPU_LOG_DIR/${dataset}_fold_${FOLD}.csv"

START=$(date +%s)
START_H=$(date "+%Y-%m-%d %H:%M:%S")

start_gpu_logging "$GPU_LOG_FILE"

nnUNetv2_predict \
-i "$INPUT" \
-o "$OUTPUT" \
-d "$DATASET_ID" \
-c 3d_fullres \
-f "$FOLD"

stop_gpu_logging

END=$(date +%s)
END_H=$(date "+%Y-%m-%d %H:%M:%S")

DURATION=$((END-START))

echo "$dataset,$FOLD,$START_H,$END_H,$DURATION,$PARAMS" >> "$TIME_LOG"

done
done

}

############################################
# RUN EVERYTHING
############################################

run_full_models

run_zero_group 708 "${datasets_708[@]}"
run_zero_group 717 "${datasets_717[@]}"
run_zero_group 718 "${datasets_718[@]}"

echo "======================================"
echo "ALL INFERENCE FINISHED"
echo "TIME LOG: $TIME_LOG"
echo "GPU LOGS: $GPU_LOG_DIR"
echo "======================================"
