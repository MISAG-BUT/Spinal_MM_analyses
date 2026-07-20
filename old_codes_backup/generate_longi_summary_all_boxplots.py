import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ROOT = "/home/nohel/DATA/MultipleMyeloma_analyses"
ANALYSIS = "longi_summary_all"
RESULTS_DIR = os.path.join(ROOT, "results", ANALYSIS)
CSV_PATH = os.path.join(RESULTS_DIR, "results_all_folds.csv")
OUTPUT_DIR = os.path.join(ROOT, "results", "threshold_all_figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_dataset_id(model_name):
    match = re.search(r"Dataset(\d+)", str(model_name))
    return int(match.group(1)) if match else None


def make_dataset_label(model_name):
    match = re.search(r"Dataset(\d+)", str(model_name))
    return f"Dataset_{match.group(1)}" if match else str(model_name)


def extract_zero_input_channel(model_name):
    match = re.search(r"zero_input_channel_(.+)", str(model_name))
    return match.group(1) if match else None


def make_zero_input_heatmap(all_df, metrics=("Dice", "F1", "NSD"), datasets=(708, 717, 718)):
    rows = []

    for dataset_id in datasets:
        full_baseline = all_df[
            (all_df["dataset_id"] == dataset_id) &
            (all_df["model_type"] == "full_model")
        ]

        if full_baseline.empty:
            continue

        full_mean = full_baseline.groupby("model_name").agg({metric: "mean" for metric in metrics})
        if len(full_mean) == 0:
            continue

        full_row = full_mean.iloc[0]

        zero_inputs = all_df[
            (all_df["dataset_id"] == dataset_id) &
            (all_df["model_type"] == "zero_input")
        ]

        if zero_inputs.empty:
            continue

        zero_mean = zero_inputs.groupby("model_name").agg({metric: "mean" for metric in metrics})

        for model_name, zero_row in zero_mean.iterrows():
            channel = extract_zero_input_channel(model_name)
            if not channel:
                continue

            for metric in metrics:
                drop = full_row[metric] - zero_row[metric]
                rows.append({
                    "dataset_id": dataset_id,
                    "channel": channel,
                    "metric": metric,
                    "drop": drop,
                })

    if not rows:
        return

    heatmap_df = pd.DataFrame(rows)

    for metric in metrics:
        pivot = heatmap_df[heatmap_df["metric"] == metric].pivot_table(
            index="channel",
            columns="dataset_id",
            values="drop",
            aggfunc="mean"
        )

        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            ax=ax,
            linewidths=0.5,
            cbar_kws={"label": f"{metric} drop"}
        )
        ax.set_title(f"Zero-input channel heatmap — {metric}")
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Zero-input channel")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, f"zero_input_channel_heatmap_{metric}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {output_path}")


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
            scale="width",
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
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Missing input file: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df["dataset_id"] = df["model_name"].apply(extract_dataset_id)
    df["dataset_label"] = df["model_name"].apply(make_dataset_label)

    full_models = df[df["model_type"] == "full_model"].copy()

    full_range_ids = list(range(700, 719))
    filtered = full_models[full_models["dataset_id"].isin(full_range_ids)].copy()

    if filtered.empty:
        raise ValueError("No rows found for datasets 700-718")

    order = [f"Dataset_{d}" for d in full_range_ids]
    filtered["dataset_label"] = pd.Categorical(filtered["dataset_label"], categories=order, ordered=True)
    make_plots(filtered, order, "longi_summary_all — models 700–718", "models_700_718")

    single_input_ids = list(range(709, 717))
    single_input = full_models[full_models["dataset_id"].isin(single_input_ids)].copy()

    if single_input.empty:
        raise ValueError("No rows found for datasets 709-716")

    single_input_order = [f"Dataset_{d}" for d in single_input_ids]
    single_input["dataset_label"] = pd.Categorical(single_input["dataset_label"], categories=single_input_order, ordered=True)
    make_plots(single_input, single_input_order, "longi_summary_all — single-input models 709–716", "single_input_709_716")

    leave_one_out_ids = list(range(700, 708))
    leave_one_out = full_models[full_models["dataset_id"].isin(leave_one_out_ids)].copy()

    if leave_one_out.empty:
        raise ValueError("No rows found for datasets 700-707")

    leave_one_out_order = [f"Dataset_{d}" for d in leave_one_out_ids]
    leave_one_out["dataset_label"] = pd.Categorical(leave_one_out["dataset_label"], categories=leave_one_out_order, ordered=True)
    make_plots(leave_one_out, leave_one_out_order, "longi_summary_all — leave-one-out models 700–707", "leave_one_out_700_707")

    grouped_ids = [708, 717, 718]
    grouped = full_models[full_models["dataset_id"].isin(grouped_ids)].copy()

    if grouped.empty:
        raise ValueError("No rows found for datasets 708, 717, 718")

    grouped_order = [f"Dataset_{d}" for d in grouped_ids]
    grouped["dataset_label"] = pd.Categorical(grouped["dataset_label"], categories=grouped_order, ordered=True)
    make_plots(grouped, grouped_order, "longi_summary_all — grouped datasets 708, 717, 718", "grouped_708_717_718")

    make_zero_input_heatmap(df)


if __name__ == "__main__":
    main()
