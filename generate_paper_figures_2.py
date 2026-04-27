import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# ROOT SETTINGS
# =====================================================

ROOT = "/home/nohel/DATA/MultipleMyeloma_analyses/results"

THRESHOLDS = [
    "longi_summary_all",
    "longi_summary_larger_than_0_3_cubic_cm",
    "longi_summary_larger_than_0_5_cubic_cm"
]

RESULTS_DIR = os.path.join(ROOT, "results_paper_summary")
os.makedirs(RESULTS_DIR, exist_ok=True)

# =====================================================
# MODEL ORDERING
# =====================================================

def extract_dataset_id(name):
    m = re.search(r"Dataset(\d+)", name)
    return int(m.group(1)) if m else 9999

# =====================================================
# LOAD FUNCTION
# =====================================================

def load_results(threshold):

    full_dir = os.path.join(ROOT, "full_models", threshold)
    zero_dir = os.path.join(ROOT, "zero_input_models", threshold)

    rows = []

    def load_dir(base_dir, model_type):
        if not os.path.exists(base_dir):
            return

        for model in os.listdir(base_dir):
            model_dir = os.path.join(base_dir, model)

            if not os.path.isdir(model_dir):
                continue

            for fold in range(5):

                f = os.path.join(
                    model_dir,
                    f"{threshold}_fold_{fold}.json"
                )

                if not os.path.exists(f):
                    continue

                d = json.load(open(f))
                fm = d["foreground_mean"]

                rows.append({
                    "threshold": threshold,
                    "model_type": model_type,
                    "model_name": model,
                    "dataset_id": extract_dataset_id(model),
                    "fold": fold,
                    "Dice": fm["Dice"],
                    "F1": fm["F1"],
                    "NSD": fm["NSD"]
                })

    load_dir(full_dir, "full")
    load_dir(zero_dir, "zero")

    return pd.DataFrame(rows)

# =====================================================
# LOAD ALL THRESHOLDS
# =====================================================

df = pd.concat([load_results(t) for t in THRESHOLDS], ignore_index=True)

df = df.sort_values(["threshold", "dataset_id", "model_type", "fold"])

# save CSV
csv_path = os.path.join(RESULTS_DIR, "all_results.csv")
df.to_csv(csv_path, index=False)

# save XLSX
xlsx_path = os.path.join(RESULTS_DIR, "all_results.xlsx")
df.to_excel(xlsx_path, index=False)

print("Saved:", csv_path)
print("Saved:", xlsx_path)

# =====================================================
# SUMMARY TABLE
# =====================================================

summary = df.groupby(
    ["threshold", "model_name", "model_type"]
).agg(
    Dice_mean=("Dice", "mean"),
    Dice_std=("Dice", "std"),
    F1_mean=("F1", "mean"),
    F1_std=("F1", "std"),
    NSD_mean=("NSD", "mean"),
    NSD_std=("NSD", "std"),
).reset_index()

summary["dataset_id"] = summary["model_name"].apply(extract_dataset_id)

summary = summary.sort_values(["threshold", "dataset_id"])

summary.to_csv(os.path.join(RESULTS_DIR, "summary_table.csv"), index=False)
summary.to_excel(os.path.join(RESULTS_DIR, "summary_table.xlsx"), index=False)

# =====================================================
# PLOTTING FUNCTION
# =====================================================

def plot_metric(metric):

    for threshold in THRESHOLDS:

        for model_type in ["full", "zero"]:

            subset = df[
                (df.threshold == threshold) &
                (df.model_type == model_type)
            ]

            if len(subset) == 0:
                continue

            plt.figure(figsize=(14,6))

            sns.violinplot(
                data=subset,
                x="model_name",
                y=metric,
                inner="box",
                scale="width"
            )

            order = sorted(
                subset["model_name"].unique(),
                key=extract_dataset_id
            )

            plt.xticks(rotation=90)
            plt.title(f"{metric} | {threshold} | {model_type}")
            plt.xlabel("Model")
            plt.ylabel(metric)

            plt.tight_layout()

            out = os.path.join(
                RESULTS_DIR,
                f"{metric}_{threshold}_{model_type}.png"
            )

            plt.savefig(out, dpi=300)
            plt.close()

            print("Saved:", out)

# =====================================================
# MODEL TREND (mean only)
# =====================================================

def plot_trends(metric):

    for threshold in THRESHOLDS:

        plt.figure(figsize=(12,6))

        subset = summary[summary.threshold == threshold]

        for model_type in ["full", "zero"]:

            tmp = subset[subset.model_type == model_type]

            tmp = tmp.sort_values("dataset_id")

            plt.plot(
                tmp["dataset_id"],
                tmp[f"{metric}_mean"],
                marker="o",
                label=model_type
            )

        plt.title(f"Model trend ({metric}) - {threshold}")
        plt.xlabel("Dataset ID")
        plt.ylabel(metric)
        plt.legend()

        out = os.path.join(
            RESULTS_DIR,
            f"trend_{metric}_{threshold}.png"
        )

        plt.tight_layout()
        plt.savefig(out, dpi=300)
        plt.close()

        print("Saved:", out)

# =====================================================
# HEATMAP
# =====================================================

def plot_heatmap(metric):

    for threshold in THRESHOLDS:

        pivot = summary[
            summary.threshold == threshold
        ].pivot_table(
            index="model_name",
            columns="model_type",
            values=f"{metric}_mean"
        )

        pivot = pivot.sort_index(key=lambda x: x.map(extract_dataset_id))

        plt.figure(figsize=(6,10))

        sns.heatmap(pivot, annot=True, cmap="viridis", fmt=".3f")

        plt.title(f"{metric} heatmap - {threshold}")

        out = os.path.join(
            RESULTS_DIR,
            f"heatmap_{metric}_{threshold}.png"
        )

        plt.tight_layout()
        plt.savefig(out, dpi=300)
        plt.close()

        print("Saved:", out)

# =====================================================
# RUN ALL PLOTS
# =====================================================

for metric in ["Dice", "F1", "NSD"]:
    plot_metric(metric)
    plot_trends(metric)
    plot_heatmap(metric)

print("DONE ✔")