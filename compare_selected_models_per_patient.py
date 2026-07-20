import os
import json
import glob
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ROOT = "/home/nohel/DATA/MultipleMyeloma_analyses"
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

OUTPUT_DIR = os.path.join(ROOT, "results", "threshold_comparison_per_patient")
os.makedirs(OUTPUT_DIR, exist_ok=True)

METRICS = ["Dice", "F1", "NSD"]


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


def load_per_patient_rows():
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

                per_case = data.get("metric_per_case", [])
                if not per_case:
                    continue

                for item in per_case:
                    if not item.get("metrics"):
                        continue

                    case_metrics = item["metrics"]
                    if not isinstance(case_metrics, dict):
                        continue

                    # take the first metric entry (usually one per case)
                    metric_entry = next(iter(case_metrics.values()), None)
                    if not isinstance(metric_entry, dict):
                        continue

                    case_id = extract_case_id(item)
                    if not case_id:
                        case_id = f"case_{len(rows)}"

                    rows.append({
                        "threshold": threshold_label,
                        "model_label": display_name,
                        "model_name": model_name,
                        "case_id": case_id,
                        "Dice": metric_entry.get("Dice"),
                        "F1": metric_entry.get("F1"),
                        "NSD": metric_entry.get("NSD"),
                    })

    if not rows:
        raise FileNotFoundError("No per-patient metric data could be loaded")

    return pd.DataFrame(rows)


def save_summary_table(df):
    summary = (
        df.groupby(["threshold", "model_label"])
        .agg(
            Dice_mean=("Dice", "mean"),
            Dice_std=("Dice", "std"),
            F1_mean=("F1", "mean"),
            F1_std=("F1", "std"),
            NSD_mean=("NSD", "mean"),
            NSD_std=("NSD", "std"),
        )
        .reset_index()
    )

    summary_path = os.path.join(OUTPUT_DIR, "selected_models_per_patient_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")
    return summary


def make_plots(df):
    threshold_order = ["all", "0.3cm", "0.5cm"]

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(10, 5))

        sns.boxplot(
            data=df,
            x="threshold",
            y=metric,
            hue="model_label",
            order=threshold_order,
            palette="Set2",
            showfliers=False,
            ax=ax,
        )

        ax.set_title(f"Per-patient {metric} comparison")
        ax.set_xlabel("Threshold")
        ax.set_ylabel(metric)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.tick_params(axis="x", rotation=0)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, title="Model", loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)

        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, f"selected_models_per_patient_{metric}_boxplot.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {out_path}")

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.violinplot(
            data=df,
            x="threshold",
            y=metric,
            hue="model_label",
            order=threshold_order,
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
        out_path = os.path.join(OUTPUT_DIR, f"selected_models_per_patient_{metric}_violin.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    df = load_per_patient_rows()
    df.to_csv(os.path.join(OUTPUT_DIR, "selected_models_per_patient_values.csv"), index=False)
    save_summary_table(df)
    make_plots(df)
    print("Done")
