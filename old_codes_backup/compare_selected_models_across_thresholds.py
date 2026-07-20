import os
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

OUTPUT_DIR = os.path.join(ROOT, "results", "threshold_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_threshold_results():
    rows = []

    for threshold_label, analysis_name in ANALYSES.items():
        csv_path = os.path.join(ROOT, "results", analysis_name, "results_all_folds.csv")

        if not os.path.exists(csv_path):
            print(f"Skipping missing file: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        df = df[df["model_type"] == "full_model"].copy()

        relevant = df[df["model_name"].isin(TARGET_MODELS.values())].copy()
        if relevant.empty:
            continue

        relevant["threshold"] = threshold_label
        relevant["model_label"] = relevant["model_name"].map(
            lambda name: next(label for label, model_name in TARGET_MODELS.items() if model_name == name)
        )

        rows.append(relevant)

    if not rows:
        raise FileNotFoundError("No threshold result files were found")

    combined = pd.concat(rows, ignore_index=True)
    return combined


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

    summary_path = os.path.join(OUTPUT_DIR, "selected_models_threshold_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary table: {summary_path}")
    return summary


def make_plots(df):
    metrics = ["Dice", "F1", "NSD"]

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 5))

        sns.boxplot(
            data=df,
            x="threshold",
            y=metric,
            hue="model_label",
            order=["all", "0.3cm", "0.5cm"],
            palette="Set2",
            showfliers=False,
            ax=ax,
        )

        ax.set_title(f"{metric} comparison across thresholds")
        ax.set_xlabel("Threshold")
        ax.set_ylabel(metric)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.tick_params(axis="x", rotation=0)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, title="Model", loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)

        plt.tight_layout()

        plot_path = os.path.join(OUTPUT_DIR, f"selected_models_{metric}_boxplot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {plot_path}")

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.violinplot(
            data=df,
            x="threshold",
            y=metric,
            hue="model_label",
            order=["all", "0.3cm", "0.5cm"],
            palette="Set2",
            inner="box",
            cut=0,
            scale="width",
            ax=ax,
        )

        ax.set_title(f"{metric} violin comparison across thresholds")
        ax.set_xlabel("Threshold")
        ax.set_ylabel(metric)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.tick_params(axis="x", rotation=0)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, title="Model", loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)

        plt.tight_layout()

        plot_path = os.path.join(OUTPUT_DIR, f"selected_models_{metric}_violin.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {plot_path}")

    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5), sharey=False)

    for ax, metric in zip(axes, metrics):
        sns.boxplot(
            data=df,
            x="threshold",
            y=metric,
            order=["all", "0.3cm", "0.5cm"],
            palette="Set2",
            showfliers=False,
            ax=ax,
        )

        ax.set_title(metric)
        ax.set_xlabel("Threshold")
        ax.set_ylabel(metric)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.tick_params(axis="x", rotation=0)
        ax.set_xlabel("Threshold")
        ax.set_ylabel(metric)
        if ax.legend_ is not None:
            ax.legend_.remove()

    fig.suptitle("Comparison of selected models across thresholds", fontsize=14, y=1.08)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plot_path = os.path.join(OUTPUT_DIR, "selected_models_threshold_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {plot_path}")

    fig_violin, axes_violin = plt.subplots(1, len(metrics), figsize=(18, 5), sharey=False)

    for ax, metric in zip(axes_violin, metrics):
        sns.violinplot(
            data=df,
            x="threshold",
            y=metric,
            order=["all", "0.3cm", "0.5cm"],
            palette="Set2",
            inner="box",
            cut=0,
            scale="width",
            ax=ax,
        )

        ax.set_title(f"{metric} violin")
        ax.set_xlabel("Threshold")
        ax.set_ylabel(metric)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.tick_params(axis="x", rotation=0)
        if ax.legend_ is not None:
            ax.legend_.remove()

    fig_violin.suptitle("Violin comparison of selected models across thresholds", fontsize=14, y=1.08)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plot_path_violin = os.path.join(OUTPUT_DIR, "selected_models_threshold_violin_comparison.png")
    plt.savefig(plot_path_violin, dpi=300, bbox_inches="tight")
    plt.close(fig_violin)
    print(f"Saved plot: {plot_path_violin}")


if __name__ == "__main__":
    df = load_threshold_results()
    save_summary_table(df)
    make_plots(df)
    print("Done")
