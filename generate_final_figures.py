import os
import json
import glob
import re
import shutil
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
            width=0.6,
            fliersize=0,
            boxprops={"edgecolor": "black", "linewidth": 1.2},
            whiskerprops={"color": "black", "linewidth": 1.0},
            capprops={"color": "black", "linewidth": 1.0},
            medianprops={"color": "black", "linewidth": 1.3},
            ax=ax,
        )
        for patch in ax.artists:
            patch.set_alpha(0.95)
        means = df.groupby("threshold")[metric].mean()
        for i, value in enumerate(means.reindex(THRESHOLD_ORDER).tolist()):
            ax.scatter([i], [value], marker="x", color="red", s=120, zorder=3, linewidths=1.8)
        ax.set_title(f"Per-patient {metric} comparison")
        ax.set_xlabel("Threshold")
        ax.set_ylabel(metric)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.tick_params(axis="x", rotation=0)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, title="Model", loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, "threshold_comparison", f"{metric}_boxplot.png")
        plt.savefig(out_path, dpi=600, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.violinplot(
            data=df,
            x="threshold",
            y=metric,
            hue="model_label",
            order=THRESHOLD_ORDER,
            palette="Set2",
            inner="box",
            cut=0,
            density_norm="width",
            ax=ax,
        )
        ax.set_title(f"Per-patient {metric} violin comparison")
        ax.set_xlabel("Threshold")
        ax.set_ylabel(metric)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.tick_params(axis="x", rotation=0)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, title="Model", loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, "threshold_comparison", f"{metric}_violin.png")
        plt.savefig(out_path, dpi=600, bbox_inches="tight")
        plt.close(fig)


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

            fig, ax = plt.subplots(figsize=(12, 6))
            sns.violinplot(
                data=subset,
                x="dataset_label",
                y=metric,
                order=order,
                ax=ax,
                inner="box",
                cut=0,
                density_norm="width",
                color="#6BAED6"
            )
            ax.set_title(f"{metric} violin — {title}")
            ax.set_xlabel("Model")
            ax.set_ylabel(metric)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            plt.tight_layout()
            out_path = os.path.join(OUTPUT_DIR, "longi_summary_all", f"{prefix}_{metric}_violin.png")
            plt.savefig(out_path, dpi=600, bbox_inches="tight")
            plt.close(fig)


def main():
    threshold_df = load_threshold_patient_rows()
    threshold_df.to_csv(os.path.join(OUTPUT_DIR, "threshold_comparison_per_patient_values.csv"), index=False)
    save_threshold_plots(threshold_df)

    longi_df = load_longi_patient_rows()
    longi_df.to_csv(os.path.join(OUTPUT_DIR, "longi_summary_all_per_patient_values.csv"), index=False)
    save_longi_plots(longi_df)

    print(f"Saved final figures to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
