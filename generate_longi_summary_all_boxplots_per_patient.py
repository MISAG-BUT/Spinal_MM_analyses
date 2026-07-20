import os
import json
import glob
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ROOT = "/home/nohel/DATA/MultipleMyeloma_analyses"
ANALYSIS = "longi_summary_all"
JSON_ROOT = os.path.join(ROOT, "full_models", ANALYSIS)
OUTPUT_DIR = os.path.join(ROOT, "results", "threshold_all_figures_per_patient")
os.makedirs(OUTPUT_DIR, exist_ok=True)


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


def load_patient_level_rows():
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

                case_id = extract_case_id(item) or f"{os.path.basename(json_file)}_{len(rows)}"

                rows.append({
                    "model_name": model_name,
                    "dataset_id": dataset_id,
                    "dataset_label": make_dataset_label(model_name),
                    "case_id": case_id,
                    "Dice": metric_entry.get("Dice"),
                    "F1": metric_entry.get("F1"),
                    "NSD": metric_entry.get("NSD"),
                })

    if not rows:
        raise FileNotFoundError("No patient-level JSON metrics were found")

    return pd.DataFrame(rows)


def make_plots(filtered, order, title_suffix, output_prefix):
    metrics = ["Dice", "F1", "NSD"]

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(12, 6))

        sns.boxplot(
            data=filtered,
            x="dataset_label",
            y=metric,
            order=order,
            ax=ax,
            showfliers=False,
            color="#4C78A8"
        )
        sns.stripplot(
            data=filtered,
            x="dataset_label",
            y=metric,
            order=order,
            ax=ax,
            color="black",
            alpha=0.35,
            size=3,
            jitter=0.1
        )

        ax.set_title(f"{metric} — {title_suffix}")
        ax.set_xlabel("Model")
        ax.set_ylabel(metric)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f"{output_prefix}_{metric}_boxplot.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {output_path}")

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(12, 6))

        sns.violinplot(
            data=filtered,
            x="dataset_label",
            y=metric,
            order=order,
            ax=ax,
            inner="box",
            cut=0,
            density_norm="width",
            color="#6BAED6"
        )

        ax.set_title(f"{metric} violin — {title_suffix}")
        ax.set_xlabel("Model")
        ax.set_ylabel(metric)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f"{output_prefix}_{metric}_violin.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {output_path}")


def main():
    patient_df = load_patient_level_rows()
    patient_df.to_csv(os.path.join(OUTPUT_DIR, "per_patient_longi_summary_all_values.csv"), index=False)

    full_range_ids = list(range(700, 719))
    filtered = patient_df[patient_df["dataset_id"].isin(full_range_ids)].copy()
    order = [f"Dataset_{d}" for d in full_range_ids]
    filtered["dataset_label"] = pd.Categorical(filtered["dataset_label"], categories=order, ordered=True)
    make_plots(filtered, order, "longi_summary_all — models 700–718", "models_700_718")

    single_input_ids = list(range(709, 717))
    single_input = patient_df[patient_df["dataset_id"].isin(single_input_ids)].copy()
    single_input_order = [f"Dataset_{d}" for d in single_input_ids]
    single_input["dataset_label"] = pd.Categorical(single_input["dataset_label"], categories=single_input_order, ordered=True)
    make_plots(single_input, single_input_order, "longi_summary_all — single-input models 709–716", "single_input_709_716")

    leave_one_out_ids = list(range(700, 708))
    leave_one_out = patient_df[patient_df["dataset_id"].isin(leave_one_out_ids)].copy()
    leave_one_out_order = [f"Dataset_{d}" for d in leave_one_out_ids]
    leave_one_out["dataset_label"] = pd.Categorical(leave_one_out["dataset_label"], categories=leave_one_out_order, ordered=True)
    make_plots(leave_one_out, leave_one_out_order, "longi_summary_all — leave-one-out models 700–707", "leave_one_out_700_707")

    grouped_ids = [708, 717, 718]
    grouped = patient_df[patient_df["dataset_id"].isin(grouped_ids)].copy()
    grouped_order = [f"Dataset_{d}" for d in grouped_ids]
    grouped["dataset_label"] = pd.Categorical(grouped["dataset_label"], categories=grouped_order, ordered=True)
    make_plots(grouped, grouped_order, "longi_summary_all — grouped datasets 708, 717, 718", "grouped_708_717_718")


if __name__ == "__main__":
    main()
