import os
import json
import glob
import re
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = "/home/nohel/DATA/MultipleMyeloma_analyses"
ANALYSIS = "longi_summary_all"
JSON_ROOT = os.path.join(ROOT, "full_models", ANALYSIS)
OUTPUT_DIR = os.path.join(ROOT, "results", "figures_final")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ANALYSES = {
    "all": "longi_summary_all",
    "0.3cm": "longi_summary_larger_than_0_3_cubic_cm",
    "0.5cm": "longi_summary_larger_than_0_5_cubic_cm",
}

TARGET_MODELS = {
    "ConvCT (709)": "Dataset709_MM_Lesion_seg_just_ConvCT",
    "VMI40 (710)": "Dataset710_MM_Lesion_seg_just_VMI_40",
    "CaSupp25 (713)": "Dataset713_MM_Lesion_seg_just_CaSupp_25",
    "All together (708)": "Dataset708_MM_Lesion_seg_all_together",
    "All VMI (717)": "Dataset717_MM_Lesion_seg_all_VMI",
    "All CaSupp (718)": "Dataset718_MM_Lesion_seg_all_CaSupp",
}

BASE_MODELS = {
    "Dataset708": "Dataset708_MM_Lesion_seg_all_together",
    "Dataset717": "Dataset717_MM_Lesion_seg_all_VMI",
    "Dataset718": "Dataset718_MM_Lesion_seg_all_CaSupp",
}

METRICS = ["Dice", "F1", "NSD"]
THRESHOLD_ORDER = ["all", "0.3cm", "0.5cm"]


def extract_dataset_id(model_name):
    match = re.search(r"Dataset(\d+)", str(model_name))
    return int(match.group(1)) if match else None


def make_dataset_label(model_name):
    match = re.search(r"Dataset(\d+)", str(model_name))
    return f"Dataset_{match.group(1)}" if match else str(model_name)


def extract_case_id(item):
    prediction_file = item.get("prediction_file", "")
    if prediction_file:
        base = os.path.basename(prediction_file)
        stem = os.path.splitext(base)[0]
        return stem
    ref = item.get("reference_file", "")
    if ref:
        return os.path.splitext(os.path.basename(ref))[0]
    return None


def load_threshold_patient_rows():
    rows = []
    for threshold_label, analysis_name in ANALYSES.items():
        for display_name, model_name in TARGET_MODELS.items():
            model_dir = os.path.join(ROOT, "full_models", analysis_name, model_name)
            if not os.path.isdir(model_dir):
                continue
            for json_file in sorted(glob.glob(os.path.join(model_dir, f"*{analysis_name}*.json"))):
                if "ALL_FOLDS" in os.path.basename(json_file):
                    continue
                with open(json_file) as fh:
                    data = json.load(fh)
                for item in data.get("metric_per_case", []):
                    if not item.get("metrics"):
                        continue
                    case_metrics = item["metrics"]
                    if not isinstance(case_metrics, dict):
                        continue
                    metric_entry = next(iter(case_metrics.values()), None)
                    if not isinstance(metric_entry, dict):
                        continue
                    rows.append({
                        "threshold": threshold_label,
                        "model_label": display_name,
                        "model_name": model_name,
                        "case_id": extract_case_id(item) or f"{os.path.basename(json_file)}_{len(rows)}",
                        "Dice": metric_entry.get("Dice"),
                        "F1": metric_entry.get("F1"),
                        "NSD": metric_entry.get("NSD"),
                    })
    if not rows:
        raise FileNotFoundError("No per-patient data loaded for threshold comparison")
    return pd.DataFrame(rows)


def load_longi_patient_rows():
    rows = []
    if not os.path.isdir(JSON_ROOT):
        raise FileNotFoundError(f"Missing JSON root: {JSON_ROOT}")
    for model_dir in sorted(glob.glob(os.path.join(JSON_ROOT, "Dataset*"))):
        if not os.path.isdir(model_dir):
            continue
        model_name = os.path.basename(model_dir)
        dataset_id = extract_dataset_id(model_name)
        if dataset_id is None:
            continue
        for json_file in sorted(glob.glob(os.path.join(model_dir, f"*{ANALYSIS}*.json"))):
            if "ALL_FOLDS" in os.path.basename(json_file):
                continue
            with open(json_file) as fh:
                data = json.load(fh)
            for item in data.get("metric_per_case", []):
                if not item.get("metrics"):
                    continue
                case_metrics = item["metrics"]
                if not isinstance(case_metrics, dict):
                    continue
                metric_entry = next(iter(case_metrics.values()), None)
                if not isinstance(metric_entry, dict):
                    continue
                rows.append({
                    "model_name": model_name,
                    "dataset_id": dataset_id,
                    "dataset_label": make_dataset_label(model_name),
                    "case_id": extract_case_id(item) or f"{os.path.basename(json_file)}_{len(rows)}",
                    "Dice": metric_entry.get("Dice"),
                    "F1": metric_entry.get("F1"),
                    "NSD": metric_entry.get("NSD"),
                })
    if not rows:
        raise FileNotFoundError("No per-patient data loaded for longi_summary_all")
    return pd.DataFrame(rows)


def save_threshold_plots(df):
    os.makedirs(os.path.join(OUTPUT_DIR, "threshold_comparison"), exist_ok=True)
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(10, 5))

        sns.boxplot(
            data=df,
            x="threshold",
            y=metric,
            hue="model_label",
            order=THRESHOLD_ORDER,
            palette=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
            showfliers=False,
            ax=ax,
        )

        for container in ax.containers:
            for box in container.boxes:
                box.set_alpha(0.95)
                box.set_edgecolor("black")
                box.set_linewidth(1.5)

        model_order = [m for m in df["model_label"].dropna().unique()]
        for model_idx, model_label in enumerate(model_order):
            container = ax.containers[model_idx] if model_idx < len(ax.containers) else None
            if container is None:
                continue
            for threshold_idx, threshold in enumerate(THRESHOLD_ORDER):
                subset = df[(df["threshold"] == threshold) & (df["model_label"] == model_label)]
                if subset.empty:
                    continue
                values = subset[metric].dropna().tolist()
                if not values:
                    continue
                if threshold_idx >= len(container.boxes):
                    continue
                box = container.boxes[threshold_idx]
                x_center = float(np.mean(box.get_path().vertices[:, 0]))
                mean_value = subset[metric].mean()
                ax.scatter(
                    [x_center + 0.02],
                    [mean_value],
                    marker="x",
                    color="red",
                    s=120,
                    zorder=3,
                    linewidths=1.8,
                )

        ax.set_title(f"{metric} comparison")
        ax.set_xlabel("Threshold")
        ax.set_ylabel(metric)
        ax.set_ylim(0.2, 1.02)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.tick_params(axis="x", rotation=0)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, title="Model", loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)

        plt.tight_layout()
        #out_path = os.path.join(OUTPUT_DIR, "threshold_comparison", f"{metric}_boxplot.png")
        out_path = os.path.join(OUTPUT_DIR, "threshold_comparison", f"lesion_size_{metric}_boxplot.png")
        plt.savefig(out_path, dpi=600, bbox_inches="tight")
        plt.close(fig)

        # fig, ax = plt.subplots(figsize=(10, 5))
        # sns.violinplot(
        #     data=df,
        #     x="threshold",
        #     y=metric,
        #     hue="model_label",
        #     order=THRESHOLD_ORDER,
        #     palette="Set2",
        #     inner="box",
        #     cut=0,
        #     density_norm="width",
        #     ax=ax,
        # )
        # ax.set_title(f"Per-patient {metric} violin comparison")
        # ax.set_xlabel("Threshold")
        # ax.set_ylabel(metric)
        # ax.grid(axis="y", linestyle="--", alpha=0.3)
        # ax.tick_params(axis="x", rotation=0)
        # handles, labels = ax.get_legend_handles_labels()
        # if handles:
        #     ax.legend(handles, labels, title="Model", loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
        # plt.tight_layout()
        # out_path = os.path.join(OUTPUT_DIR, "threshold_comparison", f"{metric}_violin.png")
        # plt.savefig(out_path, dpi=600, bbox_inches="tight")
        # plt.close(fig)


def load_model_results(base_dir, analysis_name, model_type):
    rows = []
    if not os.path.isdir(base_dir):
        return rows
    for model in sorted(os.listdir(base_dir)):
        model_dir = os.path.join(base_dir, model)
        if not os.path.isdir(model_dir):
            continue
        for fold in range(5):
            json_file = os.path.join(model_dir, f"{analysis_name}_fold_{fold}.json")
            if not os.path.exists(json_file):
                continue
            with open(json_file) as fh:
                data = json.load(fh)
            fm = data.get("foreground_mean", {})
            rows.append({
                "model_type": model_type,
                "model_name": model,
                "fold": fold,
                "Dice": fm.get("Dice"),
                "F1": fm.get("F1"),
                "NSD": fm.get("NSD"),
            })
    return rows


def compute_feature_importance(analysis_name):
    full_dir = os.path.join(ROOT, "full_models", analysis_name)
    zero_dir = os.path.join(ROOT, "zero_input_models", analysis_name)

    rows = []
    rows.extend(load_model_results(full_dir, analysis_name, "full_model"))
    rows.extend(load_model_results(zero_dir, analysis_name, "zero_input"))

    if not rows:
        return None, None, None

    df = pd.DataFrame(rows)
    df = df.sort_values(["model_type", "model_name", "fold"])

    df_mean = df.groupby(
        ["model_type", "model_name"]
    ).agg(
        Dice_mean=("Dice", "mean"),
        Dice_std=("Dice", "std"),
        F1_mean=("F1", "mean"),
        F1_std=("F1", "std"),
        NSD_mean=("NSD", "mean"),
        NSD_std=("NSD", "std"),
    ).reset_index()
    df_mean = df_mean.sort_values(["model_type", "model_name"])

    importance_rows = []
    for _, row in df_mean.iterrows():
        model_name = row["model_name"]
        if "zero_input_channel" not in model_name:
            continue
        dataset_match = re.search(r"(Dataset\d+)", model_name)
        if not dataset_match:
            continue
        dataset_id = dataset_match.group(1)
        if dataset_id not in BASE_MODELS:
            continue
        full_model = BASE_MODELS[dataset_id]
        full_row = df_mean[
            (df_mean.model_name == full_model) &
            (df_mean.model_type == "full_model")
        ]
        if full_row.empty:
            continue
        full_dice = full_row.iloc[0]["Dice_mean"]
        zero_dice = row["Dice_mean"]
        drop = full_dice - zero_dice
        channel_match = re.search(r"zero_input_channel_(.*)", model_name)
        channel = channel_match.group(1) if channel_match else model_name
        importance_rows.append({
            "channel": channel,
            "dataset": dataset_id,
            "Dice_drop": drop,
            "full_model_dice": full_dice,
            "zero_input_dice": zero_dice,
        })

    importance_df = pd.DataFrame(importance_rows)
    if not importance_df.empty:
        importance_df = importance_df.sort_values("Dice_drop", ascending=False)

    return df, df_mean, importance_df


def save_feature_importance_outputs(threshold_label, analysis_name, df, df_mean, importance_df):
    threshold_dir = os.path.join(OUTPUT_DIR, "feature_importance", threshold_label)
    images_dir = os.path.join(threshold_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    results_csv = os.path.join(threshold_dir, "results_all_folds.csv")
    mean_csv = os.path.join(threshold_dir, "results_model_means.csv")
    importance_csv = os.path.join(threshold_dir, "feature_importance.csv")
    summary_xlsx = os.path.join(threshold_dir, "feature_importance_summary.xlsx")

    df.to_csv(results_csv, index=False)
    df_mean.to_csv(mean_csv, index=False)
    importance_df.to_csv(importance_csv, index=False)

    with pd.ExcelWriter(summary_xlsx) as writer:
        df.to_excel(writer, sheet_name="all_folds", index=False)
        df_mean.to_excel(writer, sheet_name="model_means", index=False)
        importance_df.to_excel(writer, sheet_name="feature_importance", index=False)

    print(f"Saved feature importance tables for {threshold_label}: {threshold_dir}")

    if importance_df is None or importance_df.empty:
        return

    datasets = importance_df["dataset"].unique()
    for dataset in datasets:
        subset = importance_df[importance_df["dataset"] == dataset].sort_values("Dice_drop", ascending=False)
        plt.figure(figsize=(8, 5))
        plt.bar(subset["channel"], subset["Dice_drop"])
        plt.ylabel("Dice drop (importance)")
        plt.xlabel("Removed channel")
        plt.title(f"Channel importance ({threshold_label} — {dataset})")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_path = os.path.join(images_dir, f"feature_importance_{threshold_label}_{dataset}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved: {plot_path}")

    label_map = {
        "Dataset708": "All together (708)",
        "Dataset717": "All VMI (717)",
        "Dataset718": "All CaSupp (718)",
    }
    heatmap_df = importance_df.pivot_table(
        index="channel",
        columns="dataset",
        values="Dice_drop"
    )
    heatmap_df = heatmap_df.rename(columns=label_map)
    plt.figure(figsize=(6, 6))
    sns.heatmap(
        heatmap_df,
        annot=True,
        cmap="viridis",
        fmt=".3f",
        linewidths=0.5,
    )
    plt.title(f"Channel importance heatmap ({threshold_label})")
    plt.ylabel("Removed channel")
    plt.xlabel("Dataset")
    plt.tight_layout()
    heatmap_path = os.path.join(images_dir, f"feature_importance_heatmap_{threshold_label}.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {heatmap_path}")


def save_longi_plots(df):
    os.makedirs(os.path.join(OUTPUT_DIR, "longi_summary_all"), exist_ok=True)
    #groups = [
    #    (list(range(700, 719)), "models_700_718", "longi_summary_all — models 700–718"),
    #    (list(range(709, 717)), "single_input_709_716", "longi_summary_all — single-input models 709–716"),
    #    (list(range(700, 708)), "leave_one_out_700_707", "longi_summary_all — leave-one-out models 700–707"),
    #    ([708, 717, 718], "grouped_708_717_718", "longi_summary_all — grouped datasets 708, 717, 718"),
    #]
    groups = [
        (list(range(700, 719)), "models_700_718", "models 700–718"),
        (list(range(709, 717)), "single_input_709_716", "single-input models 709–716"),
        (list(range(700, 708)), "leave_one_out_700_707", "leave-one-out models 700–707"),
        ([708, 717, 718], "grouped_708_717_718", "grouped datasets 708, 717, 718"),
    ]

    
    
    for ids, prefix, title in groups:
        subset = df[df["dataset_id"].isin(ids)].copy()
        if subset.empty:
            continue
        order = [f"Dataset_{d}" for d in ids]
        subset["dataset_label"] = pd.Categorical(subset["dataset_label"], categories=order, ordered=True)
        for metric in METRICS:
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(
                data=subset,
                x="dataset_label",
                y=metric,
                order=order,
                ax=ax,
                showfliers=False,
                color="#4C78A8"
            )
            means = subset.groupby("dataset_label")[metric].mean()
            for i, value in enumerate(means.reindex(order).tolist()):
                ax.scatter([i], [value], marker="x", color="red", s=120, zorder=3, linewidths=1.8)
            sns.stripplot(
                data=subset,
                x="dataset_label",
                y=metric,
                order=order,
                ax=ax,
                color="black",
                alpha=0.35,
                size=3,
                jitter=0.1
            )
            ax.set_title(f"{metric} — {title}")
            ax.set_xlabel("Model")
            ax.set_ylabel(metric)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            plt.tight_layout()
            out_path = os.path.join(OUTPUT_DIR, "longi_summary_all", f"{prefix}_{metric}_boxplot.png")
            plt.savefig(out_path, dpi=600, bbox_inches="tight")
            plt.close(fig)

            # fig, ax = plt.subplots(figsize=(12, 6))
            # sns.violinplot(
            #     data=subset,
            #     x="dataset_label",
            #     y=metric,
            #     order=order,
            #     ax=ax,
            #     inner="box",
            #     cut=0,
            #     density_norm="width",
            #     color="#6BAED6"
            # )
            # ax.set_title(f"{metric} violin — {title}")
            # ax.set_xlabel("Model")
            # ax.set_ylabel(metric)
            # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            # ax.grid(axis="y", linestyle="--", alpha=0.3)
            # plt.tight_layout()
            # out_path = os.path.join(OUTPUT_DIR, "longi_summary_all", f"{prefix}_{metric}_violin.png")
            # plt.savefig(out_path, dpi=600, bbox_inches="tight")
            # plt.close(fig)


def main():
    threshold_df = load_threshold_patient_rows()
    threshold_df.to_csv(os.path.join(OUTPUT_DIR, "threshold_comparison_per_patient_values.csv"), index=False)
    save_threshold_plots(threshold_df)

    for threshold_label, analysis_name in ANALYSES.items():
        df, df_mean, importance_df = compute_feature_importance(analysis_name)
        if df is None:
            print(f"No feature importance results for threshold: {threshold_label}")
            continue
        save_feature_importance_outputs(threshold_label, analysis_name, df, df_mean, importance_df)

    longi_df = load_longi_patient_rows()
    longi_df.to_csv(os.path.join(OUTPUT_DIR, "longi_summary_all_per_patient_values.csv"), index=False)
    save_longi_plots(longi_df)

    print(f"Saved final figures to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
