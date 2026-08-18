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
OUTPUT_DIR = os.path.join(ROOT, "results", "figures_final_new_2")
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

DATASET_LABELS = {
    700: "LOO w/o ConvCT (700)",
    701: "LOO w/o VMI40 (701)",
    702: "LOO w/o VMI80 (702)",
    703: "LOO w/o VMI120 (703)",
    704: "LOO w/o CaSupp25 (704)",
    705: "LOO w/o CaSupp50 (705)",
    706: "LOO w/o CaSupp75 (706)",
    707: "LOO w/o CaSupp100 (707)",
    708: "All together (708)",
    709: "ConvCT (709)",
    710: "VMI40 (710)",
    711: "VMI80 (711)",
    712: "VMI120 (712)",
    713: "CaSupp25 (713)",
    714: "CaSupp50 (714)",
    715: "CaSupp75 (715)",
    716: "CaSupp100 (716)",
    717: "All VMI (717)",
    718: "All CaSupp (718)",
}

DATASET_GROUP = {
    700: "LOO",
    701: "LOO",
    702: "LOO",
    703: "LOO",
    704: "LOO",
    705: "LOO",
    706: "LOO",
    707: "LOO",
    708: "All together",
    709: "ConvCT",
    710: "VMI",
    711: "VMI",
    712: "VMI",
    713: "CaSupp",
    714: "CaSupp",
    715: "CaSupp",
    716: "CaSupp",
    717: "VMI",
    718: "CaSupp",
}

GROUP_COLORS = {
    "LOO": "#72B0F1",
    "ConvCT": "#ff7f0e",
    "VMI": "#2ca02c",
    "CaSupp": "#d62728",
    "All together": "#9467bd",
}

METRICS = ["Dice", "F1", "NSD"]
THRESHOLD_ORDER = ["all", "0.3cm", "0.5cm"]
THRESHOLD_DISPLAY = {
    "all": "all",
    # use mathtext so the 3 is superscript (no caret shown)
    "0.3cm": r"0.3 cm$^3$",
    "0.5cm": r"0.5 cm$^3$",
}


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
                    color="black",
                    s=120,
                    zorder=3,
                    linewidths=1.8,
                )

        ax.set_title(f"{metric} comparison")
        ax.set_xlabel("Threshold")
        # show threshold labels in cubic centimeters
        ax.set_xticklabels([THRESHOLD_DISPLAY.get(t, t) for t in THRESHOLD_ORDER])
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
        # Dice importance (drop)
        drop = full_dice - zero_dice

        # Also compute detection performance drop (F1)
        full_f1 = full_row.iloc[0]["F1_mean"]
        zero_f1 = row["F1_mean"]
        f1_drop = full_f1 - zero_f1
        channel_match = re.search(r"zero_input_channel_(.*)", model_name)
        channel = channel_match.group(1) if channel_match else model_name
        importance_rows.append({
            "channel": channel,
            "dataset": dataset_id,
            "Dice_drop": drop,
            "F1_drop": f1_drop,
            "full_model_dice": full_dice,
            "zero_input_dice": zero_dice,
            "full_model_f1": full_f1,
            "zero_input_f1": zero_f1,
        })

    importance_df = pd.DataFrame(importance_rows)
    if not importance_df.empty:
        # rename channel column to clarify it's the zero-input channel
        if "channel" in importance_df.columns:
            importance_df = importance_df.rename(columns={"channel": "zero_input_channel"})
        # sort by dataset first, then by Dice_drop descending
        sort_cols = ["dataset", "Dice_drop"]
        importance_df = importance_df.sort_values(sort_cols, ascending=[True, False])
        # ensure dataset is first column and zero_input_channel second
        cols = [c for c in importance_df.columns if c not in ("dataset", "zero_input_channel")]
        importance_df = importance_df[["dataset", "zero_input_channel"] + cols]

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
        plt.bar(subset["zero_input_channel"], subset["Dice_drop"])
        plt.ylabel("Dice drop (importance)")
        plt.xlabel("Removed channel")
        plt.title(f"Channel importance ({threshold_label} — {dataset})")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_path = os.path.join(images_dir, f"feature_importance_{threshold_label}_{dataset}.png")
        #plt.savefig(plot_path, dpi=600)
        plt.close()
        print(f"Saved: {plot_path}")

    label_map = {
        "Dataset708": "All together (708)",
        "Dataset717": "All VMI (717)",
        "Dataset718": "All CaSupp (718)",
    }

    # pivot tables for Dice and F1 drops
    dice_heatmap_df = importance_df.pivot_table(
        index="zero_input_channel",
        columns="dataset",
        values="Dice_drop"
    )
    f1_heatmap_df = importance_df.pivot_table(
        index="zero_input_channel",
        columns="dataset",
        values="F1_drop"
    )

    # rename dataset columns to friendly labels
    dice_heatmap_df = dice_heatmap_df.rename(columns=label_map)
    f1_heatmap_df = f1_heatmap_df.rename(columns=label_map)

    # save pivot matrices
    dice_csv = os.path.join(threshold_dir, "feature_importance_heatmap_Dice.csv")
    f1_csv = os.path.join(threshold_dir, "feature_importance_heatmap_F1.csv")
    dice_heatmap_df.to_csv(dice_csv)
    f1_heatmap_df.to_csv(f1_csv)

    # append heatmap sheets to the summary workbook
    try:
        with pd.ExcelWriter(summary_xlsx, mode="a", if_sheet_exists="replace") as writer:
            dice_heatmap_df.to_excel(writer, sheet_name="heatmap_Dice")
            f1_heatmap_df.to_excel(writer, sheet_name="heatmap_F1")
    except Exception:
        # fallback: write a new workbook with the extra sheets (overwrites)
        with pd.ExcelWriter(summary_xlsx) as writer:
            df.to_excel(writer, sheet_name="all_folds", index=False)
            df_mean.to_excel(writer, sheet_name="model_means", index=False)
            importance_df.to_excel(writer, sheet_name="feature_importance", index=False)
            dice_heatmap_df.to_excel(writer, sheet_name="heatmap_Dice")
            f1_heatmap_df.to_excel(writer, sheet_name="heatmap_F1")

    # plot Dice heatmap with colorbar
    plt.figure(figsize=(6, 6))
    dice_cbar_min = dice_heatmap_df.min().min()
    dice_cbar_max = dice_heatmap_df.max().max()
    sns.heatmap(
        dice_heatmap_df,
        annot=True,
        cmap="viridis",
        fmt=".3f",
        linewidths=0.5,
        vmin=dice_cbar_min,
        vmax=dice_cbar_max,
        cbar_kws={"label": "Dice drop"},
    )
    plt.title(f"Channel importance heatmap — Dice drop")
    plt.ylabel("Removed channel")
    plt.xlabel("Dataset")
    plt.tight_layout()
    heatmap_path = os.path.join(images_dir, f"feature_importance_heatmap_{threshold_label}_Dice.png")
    plt.savefig(heatmap_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved: {heatmap_path}")

    # plot F1 heatmap with same color range as Dice heatmap
    plt.figure(figsize=(6, 6))
    sns.heatmap(
        f1_heatmap_df,
        annot=True,
        cmap="viridis",
        fmt=".3f",
        linewidths=0.5,
        vmin=dice_cbar_min,
        vmax=dice_cbar_max,
        cbar_kws={"label": "F1 drop"},
    )
    plt.title(f"Channel importance heatmap — F1 drop")
    plt.ylabel("Removed channel")
    plt.xlabel("Dataset")
    plt.tight_layout()
    heatmap_path_f1 = os.path.join(images_dir, f"feature_importance_heatmap_{threshold_label}_F1.png")
    plt.savefig(heatmap_path_f1, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved: {heatmap_path_f1}")


def save_longi_plots(df):
    os.makedirs(os.path.join(OUTPUT_DIR, "longi_summary_all"), exist_ok=True)
    groups = [
        (list(range(700, 719)), "models_700_718", "models 700–718"),
        (list(range(709, 717)), "single_input_709_716", "single-input models 709–716"),
        (list(range(700, 709)), "leave_one_out_700_708", "leave-one-out models 700–708"),
        ([708, 717, 718], "grouped_708_717_718", "grouped datasets 708, 717, 718"),
    ]

    
    
    for ids, prefix, title in groups:
        subset = df[df["dataset_id"].isin(ids)].copy()
        if subset.empty:
            continue
        order = [f"Dataset_{d}" for d in ids]
        subset["dataset_label"] = pd.Categorical(subset["dataset_label"], categories=order, ordered=True)
        for metric in METRICS:
            fig, ax = plt.subplots(figsize=(12, 5))
            palette = [
                GROUP_COLORS.get(DATASET_GROUP.get(dataset_id, "LOO"), "#4C78A8")
                for dataset_id in ids
            ]
            sns.boxplot(
                data=subset,
                x="dataset_label",
                y=metric,
                order=order,
                palette=palette,
                ax=ax,
                showfliers=False,
            )
            for i, artist in enumerate(ax.artists):
                artist.set_edgecolor("black")
                artist.set_alpha(0.9)
            means = subset.groupby("dataset_label")[metric].mean()
            for i, value in enumerate(means.reindex(order).tolist()):
                ax.scatter([i], [value], marker="x", color="black", s=120, zorder=3, linewidths=1.8)
            sns.stripplot(
                data=subset,
                x="dataset_label",
                y=metric,
                order=order,
                ax=ax,
                color="black",
                alpha=0.35,
                size=3,
                jitter=0.1,
            )
            ax.set_title(f"{metric} — {title}")
            ax.set_xlabel("Model")
            ax.set_ylabel(metric)

            xtick_labels = []
            for label in order:
                match = re.search(r"Dataset_(\d+)", label)
                if match:
                    dataset_id = int(match.group(1))
                    xtick_labels.append(DATASET_LABELS.get(dataset_id, label))
                else:
                    xtick_labels.append(label)
            ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=8)
            ax.grid(axis="y", linestyle="--", alpha=0.3)

            legend_groups = {}
            for dataset_id in ids:
                group = DATASET_GROUP.get(dataset_id, "LOO")
                if dataset_id == 717:
                    group = "VMI"
                elif dataset_id == 718:
                    group = "CaSupp"
                legend_groups[group] = GROUP_COLORS.get(group, "#4C78A8")

            dataset_handles = [
                Patch(
                    facecolor=color,
                    edgecolor="black",
                    label=group,
                )
                for group, color in legend_groups.items()
            ]

            if dataset_handles:
                ax.legend(
                    handles=dataset_handles,
                    title="Dataset",
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    frameon=False,
                )

            plt.tight_layout(rect=[0, 0, 0.78, 1])
            out_path = os.path.join(OUTPUT_DIR, "longi_summary_all", f"{prefix}_{metric}_boxplot.png")
            plt.savefig(out_path, dpi=600, bbox_inches="tight")
            plt.close(fig)


def save_loo_difference_plots(df):
    """
    Create difference plots: LOO models (700-707) minus GT (708).
    For each case_id, compute: LOO_value - GT_value
    """
    os.makedirs(os.path.join(OUTPUT_DIR, "loo_difference_from_gt"), exist_ok=True)
    
    # Filter for LOO models (700-707) and GT (708)
    loo_ids = list(range(700, 708))  # 700-707
    gt_id = 708
    
    df_filtered = df[df["dataset_id"].isin(loo_ids + [gt_id])].copy()
    
    if df_filtered.empty:
        print("Warning: No LOO or GT data found")
        return
    
    # Get GT data
    gt_data = df_filtered[df_filtered["dataset_id"] == gt_id][["case_id", "Dice", "F1", "NSD"]].copy()
    gt_data = gt_data.rename(columns={"Dice": "GT_Dice", "F1": "GT_F1", "NSD": "GT_NSD"})
    
    # Process LOO models
    difference_rows = []
    for loo_id in loo_ids:
        loo_data = df_filtered[df_filtered["dataset_id"] == loo_id][["case_id", "dataset_id", "dataset_label", "Dice", "F1", "NSD"]].copy()
        
        # Merge with GT data
        merged = loo_data.merge(gt_data, on="case_id", how="inner")
        
        # Compute differences (LOO - GT)
        merged["Dice_diff"] = merged["Dice"] - merged["GT_Dice"]
        merged["F1_diff"] = merged["F1"] - merged["GT_F1"]
        merged["NSD_diff"] = merged["NSD"] - merged["GT_NSD"]
        
        difference_rows.append(merged)
    
    diff_df = pd.concat(difference_rows, ignore_index=True)
    
    if diff_df.empty:
        print("Warning: No matching case_ids between LOO and GT")
        return
    
    # Save difference data
    diff_csv = os.path.join(OUTPUT_DIR, "loo_difference_from_gt", "loo_vs_gt_differences.csv")
    diff_df.to_csv(diff_csv, index=False)
    print(f"Saved LOO vs GT differences: {diff_csv}")
    
    # Create plots
    order = [f"Dataset_{d}" for d in loo_ids]
    diff_df["dataset_label"] = pd.Categorical(diff_df["dataset_label"], categories=order, ordered=True)
    
    for metric in ["Dice_diff", "F1_diff", "NSD_diff"]:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        palette = [
            GROUP_COLORS.get("LOO", "#72B0F1")
            for _ in loo_ids
        ]
        
        sns.boxplot(
            data=diff_df,
            x="dataset_label",
            y=metric,
            order=order,
            palette=palette,
            ax=ax,
            showfliers=False,
        )
        
        # Format artist
        for i, artist in enumerate(ax.artists):
            artist.set_edgecolor("black")
            artist.set_alpha(0.9)
        
        # Add mean markers
        means = diff_df.groupby("dataset_label")[metric].mean()
        for i, value in enumerate(means.reindex(order).tolist()):
            ax.scatter([i], [value], marker="x", color="red", s=120, zorder=3, linewidths=1.8, label="Mean" if i == 0 else "")
        
        # Add individual points
        sns.stripplot(
            data=diff_df,
            x="dataset_label",
            y=metric,
            order=order,
            ax=ax,
            color="black",
            alpha=0.35,
            size=3,
            jitter=0.1,
        )
        
        # Labels and formatting
        metric_name = metric.replace("_diff", "")
        ax.set_title(f"{metric_name} difference (LOO - GT)")
        ax.set_xlabel("LOO Model")
        ax.set_ylabel(f"{metric_name} difference")
        
        # Set x-axis labels to friendly names
        xtick_labels = [DATASET_LABELS.get(loo_ids[i], f"Dataset_{loo_ids[i]}") for i in range(len(loo_ids))]
        ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=8)
        
        # Add zero line
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, "loo_difference_from_gt", f"loo_vs_gt_{metric_name}_diff.png")
        plt.savefig(out_path, dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

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
    
    # Generate LOO difference plots
    save_loo_difference_plots(longi_df)

    print(f"Saved final figures to: {OUTPUT_DIR}")



if __name__ == "__main__":
    main()
