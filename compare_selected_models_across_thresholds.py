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

    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5), sharey=False)

    for ax, metric in zip(axes, metrics):
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

        ax.set_title(metric)
        ax.set_xlabel("Threshold")
        ax.set_ylabel(metric)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.tick_params(axis="x", rotation=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Model", loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3)

    fig.suptitle("Comparison of selected models across thresholds", fontsize=14, y=1.08)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plot_path = os.path.join(OUTPUT_DIR, "selected_models_threshold_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {plot_path}")

    fig2, axes2 = plt.subplots(3, 1, figsize=(10, 12), sharex=False)

    for ax2, metric in zip(axes2, metrics):
        sns.barplot(
            data=(
                df.groupby(["threshold", "model_label", "fold"])[metric]
                .mean()
                .reset_index()
                .groupby(["threshold", "model_label"])[metric]
                .mean()
                .reset_index()
            ),
            x="threshold",
            y=metric,
            hue="model_label",
            palette="Set2",
            ax=ax2,
        )
        ax2.set_title(f"Mean {metric} by threshold")
        ax2.set_xlabel("Threshold")
        ax2.set_ylabel(f"Mean {metric}")
        ax2.grid(axis="y", linestyle="--", alpha=0.3)

    handles2, labels2 = axes2[0].get_legend_handles_labels()
    fig2.legend(handles2, labels2, title="Model", loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3)

    fig2.suptitle("Mean metric comparison across thresholds", fontsize=14, y=1.08)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plot_path2 = os.path.join(OUTPUT_DIR, "selected_models_threshold_mean_comparison.png")
    plt.savefig(plot_path2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved plot: {plot_path2}")


if __name__ == "__main__":
    df = load_threshold_results()
    save_summary_table(df)
    make_plots(df)
    print("Done")
