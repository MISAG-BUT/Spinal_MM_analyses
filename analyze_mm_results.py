import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

ROOT = "/home/nohel/DATA/MultipleMyeloma_analyses"

FULL_DIR = os.path.join(ROOT, "full_models", "longi_summary_all")
ZERO_DIR = os.path.join(ROOT, "zero_input_models", "longi_summary_all")

OUT_ALL = os.path.join(ROOT, "results_all_folds.csv")
OUT_MEAN = os.path.join(ROOT, "results_model_means.csv")
OUT_IMPORTANCE = os.path.join(ROOT, "feature_importance.csv")

PLOT_FILE = os.path.join(ROOT, "feature_importance.png")


# =====================================================
# STEP 1 — LOAD ALL JSON RESULTS
# =====================================================

rows = []

def load_results(base_dir, model_type):

    if not os.path.exists(base_dir):
        return

    for model in os.listdir(base_dir):

        model_dir = os.path.join(base_dir, model)
        if not os.path.isdir(model_dir):
            continue

        for fold in range(5):

            json_file = os.path.join(
                model_dir,
                f"longi_summary_all_fold_{fold}.json"
            )

            if not os.path.exists(json_file):
                continue

            with open(json_file) as f:
                data = json.load(f)

            fm = data["foreground_mean"]

            rows.append({
                "model_type": model_type,
                "model_name": model,
                "fold": fold,
                "Dice": fm["Dice"],
                "F1": fm["F1"],
                "NSD": fm["NSD"]
            })


load_results(FULL_DIR, "full_model")
load_results(ZERO_DIR, "zero_input")

df = pd.DataFrame(rows)

df.to_csv(OUT_ALL, index=False)

print("Saved:", OUT_ALL)
print("Total rows:", len(df))


# =====================================================
# STEP 2 — MEAN OVER FOLDS
# =====================================================

df_mean = df.groupby(
    ["model_type", "model_name"]
).agg(
    Dice_mean=("Dice", "mean"),
    Dice_std=("Dice", "std"),
    F1_mean=("F1", "mean"),
    F1_std=("F1", "std"),
    NSD_mean=("NSD", "mean"),
    NSD_std=("NSD", "std")
).reset_index()

df_mean.to_csv(OUT_MEAN, index=False)

print("Saved:", OUT_MEAN)


# =====================================================
# STEP 3 — FEATURE IMPORTANCE
# =====================================================

# mapping base models
base_models = {
    "Dataset708": "Dataset708_MM_Lesion_seg_all_together",
    "Dataset717": "Dataset717_MM_Lesion_seg_all_VMI",
    "Dataset718": "Dataset718_MM_Lesion_seg_all_CaSupp"
}

importance_rows = []

for _, row in df_mean.iterrows():

    model = row["model_name"]

    if "zero_input_channel" not in model:
        continue

    dataset_match = re.search(r"(Dataset\d+)", model)
    if not dataset_match:
        continue

    dataset_id = dataset_match.group(1)

    if dataset_id not in base_models:
        continue

    full_model = base_models[dataset_id]

    full_row = df_mean[
        (df_mean.model_name == full_model) &
        (df_mean.model_type == "full_model")
    ]

    if len(full_row) == 0:
        continue

    full_dice = full_row.iloc[0]["Dice_mean"]

    zero_dice = row["Dice_mean"]

    drop = full_dice - zero_dice

    channel_match = re.search(
        r"zero_input_channel_(.*)",
        model
    )

    channel = channel_match.group(1)

    importance_rows.append({
        "channel": channel,
        "dataset": dataset_id,
        "Dice_drop": drop,
        "full_model_dice": full_dice,
        "zero_input_dice": zero_dice
    })


importance_df = pd.DataFrame(importance_rows)

importance_df = importance_df.sort_values(
    "Dice_drop",
    ascending=False
)

importance_df.to_csv(OUT_IMPORTANCE, index=False)

print("Saved:", OUT_IMPORTANCE)


# =====================================================
# STEP 4 — PLOT FEATURE IMPORTANCE
# =====================================================

if len(importance_df) > 0:

    plt.figure(figsize=(10,6))

    plt.bar(
        importance_df["channel"],
        importance_df["Dice_drop"]
    )

    plt.ylabel("Dice drop (importance)")
    plt.xlabel("Removed channel")
    plt.title("Channel importance (zero-input analysis)")
    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(PLOT_FILE, dpi=300)

    print("Saved:", PLOT_FILE)

print("Analysis finished.")
