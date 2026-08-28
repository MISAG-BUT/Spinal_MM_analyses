import itertools
import os

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, shapiro, wilcoxon


RESULTS_DIR = "/home/nohel/DATA/MultipleMyeloma_analyses/results/figures_final_new_2"
OUTPUT_DIR = os.path.join(RESULTS_DIR, "statistical_analysis")

LONGI_FILE = os.path.join(RESULTS_DIR, "longi_summary_all_per_patient_values.csv")
THRESHOLD_FILE = os.path.join(RESULTS_DIR, "threshold_comparison_per_patient_values.csv")

METRICS = ["Dice", "F1"]
SINGLE_INPUT_IDS = list(range(709, 717))
THRESHOLD_ORDER = ["all", "0.3cm", "0.5cm"]


def holm_adjust(p_values):
    """Return Holm-adjusted p-values in the original order."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p_values), np.nan, dtype=float)
    valid = np.isfinite(p_values)
    valid_indices = np.flatnonzero(valid)
    order = valid_indices[np.argsort(p_values[valid])]

    running_max = 0.0
    for rank, index in enumerate(order):
        adjusted_value = (len(order) - rank) * p_values[index]
        running_max = max(running_max, adjusted_value)
        adjusted[index] = min(running_max, 1.0)
    return adjusted


def significance_label(p_value):
    if pd.isna(p_value):
        return "NA"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def paired_rank_biserial(x, y):
    """Effect size for paired data: positive values favor x over y."""
    differences = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    differences = differences[np.isfinite(differences) & (differences != 0)]
    if len(differences) == 0:
        return 0.0

    ranks = rankdata(np.abs(differences), method="average")
    rank_sum = ranks.sum()
    return float((ranks[differences > 0].sum() - ranks[differences < 0].sum()) / rank_sum)


def paired_wilcoxon(x, y):
    differences = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    differences = differences[np.isfinite(differences)]
    if len(differences) == 0 or np.allclose(differences, 0):
        return np.nan
    return float(wilcoxon(differences, alternative="two-sided", zero_method="wilcox").pvalue)


def shapiro_p_value(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3 or len(values) > 5000:
        return np.nan
    return float(shapiro(values).pvalue)


def run_paired_analysis(data, conditions, condition_column, analysis_name):
    omnibus_rows = []
    pairwise_rows = []
    normality_rows = []
    raw_pairwise_rows = []

    for metric in METRICS:
        selected = data[data[condition_column].isin(conditions)].copy()
        selected[metric] = pd.to_numeric(selected[metric], errors="coerce")
        selected = selected.dropna(subset=["case_id", condition_column, metric])

        # Keep only complete patient blocks for a valid paired comparison.
        complete_cases = (
            selected.groupby("case_id")[condition_column]
            .nunique()
            .loc[lambda values: values == len(conditions)]
            .index
        )
        complete = selected[selected["case_id"].isin(complete_cases)]
        wide = complete.pivot_table(
            index="case_id",
            columns=condition_column,
            values=metric,
            aggfunc="first",
        ).reindex(columns=conditions).dropna()

        n = len(wide)
        for condition in conditions:
            normality_p = shapiro_p_value(wide[condition].to_numpy())
            normality_rows.append({
                "analysis": analysis_name,
                "metric": metric,
                "condition": condition,
                "n": n,
                "shapiro_p_value": normality_p,
                "normal_distribution": (
                    "yes" if pd.notna(normality_p) and normality_p >= 0.05
                    else "no" if pd.notna(normality_p) else "not tested"
                ),
            })
        if n >= 2 and len(conditions) >= 3:
            statistic, omnibus_p = friedmanchisquare(
                *(wide[condition].to_numpy() for condition in conditions)
            )
        else:
            statistic, omnibus_p = np.nan, np.nan

        omnibus_rows.append({
            "analysis": analysis_name,
            "metric": metric,
            "n_complete_patients": n,
            "n_conditions": len(conditions),
            "test_used": "friedman" if n >= 2 and len(conditions) >= 3 else "not tested",
            "test_statistic": statistic,
            "p_value": omnibus_p,
            "significance": significance_label(omnibus_p),
        })

        for first_condition, second_condition in itertools.combinations(conditions, 2):
            if n >= 2:
                first_values = wide[first_condition].to_numpy()
                second_values = wide[second_condition].to_numpy()
                raw_p = paired_wilcoxon(first_values, second_values)
                effect = paired_rank_biserial(first_values, second_values)
            else:
                raw_p, effect = np.nan, np.nan

            raw_pairwise_rows.append({
                "analysis": analysis_name,
                "metric": metric,
                "condition_1": first_condition,
                "condition_2": second_condition,
                "n_complete_patients": n,
                "test_used": "wilcoxon" if n >= 2 else "not tested",
                "p_value": raw_p,
                "rank_biserial_effect": effect,
            })

    raw_pairwise = pd.DataFrame(raw_pairwise_rows)
    if not raw_pairwise.empty:
        raw_pairwise["holm_p_value"] = np.nan
        for metric, indices in raw_pairwise.groupby("metric").groups.items():
            raw_pairwise.loc[indices, "holm_p_value"] = holm_adjust(
                raw_pairwise.loc[indices, "p_value"].to_numpy()
            )
        raw_pairwise["significance"] = raw_pairwise["holm_p_value"].map(significance_label)
        pairwise_rows = raw_pairwise.to_dict("records")

    return (
        pd.DataFrame(omnibus_rows),
        pd.DataFrame(pairwise_rows),
        pd.DataFrame(normality_rows),
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    longi = pd.read_csv(LONGI_FILE)
    longi["dataset_id"] = pd.to_numeric(longi["dataset_id"], errors="coerce")
    single_input = longi[longi["dataset_id"].isin(SINGLE_INPUT_IDS)].copy()
    single_conditions = [f"Dataset_{dataset_id}" for dataset_id in SINGLE_INPUT_IDS]
    single_input["condition"] = single_input["dataset_id"].map(
        lambda dataset_id: f"Dataset_{int(dataset_id)}"
    )
    single_omnibus, single_pairwise, single_normality = run_paired_analysis(
        single_input,
        single_conditions,
        "condition",
        "single_input_models_709_716",
    )

    # Threshold analysis is currently disabled.
    # threshold = pd.read_csv(THRESHOLD_FILE)
    # threshold_omnibus_rows = []
    # threshold_pairwise_rows = []
    # threshold_normality_rows = []
    # for model_label, model_data in threshold.groupby("model_label"):
    #     model_omnibus, model_pairwise, model_normality = run_paired_analysis(
    #         model_data,
    #         THRESHOLD_ORDER,
    #         "threshold",
    #         f"thresholds_{model_label}",
    #     )
    #     model_omnibus.insert(1, "model_label", model_label)
    #     model_pairwise.insert(1, "model_label", model_label)
    #     model_normality.insert(1, "model_label", model_label)
    #     threshold_omnibus_rows.append(model_omnibus)
    #     threshold_pairwise_rows.append(model_pairwise)
    #     threshold_normality_rows.append(model_normality)
    #
    # threshold_omnibus = pd.concat(threshold_omnibus_rows, ignore_index=True)
    # threshold_pairwise = pd.concat(threshold_pairwise_rows, ignore_index=True)
    # threshold_normality = pd.concat(threshold_normality_rows, ignore_index=True)

    single_omnibus.to_csv(os.path.join(OUTPUT_DIR, "single_input_omnibus_tests.csv"), index=False)
    single_pairwise.to_csv(os.path.join(OUTPUT_DIR, "single_input_pairwise_tests.csv"), index=False)
    single_normality.to_csv(os.path.join(OUTPUT_DIR, "single_input_normality.csv"), index=False)
    # threshold_omnibus.to_csv(os.path.join(OUTPUT_DIR, "threshold_omnibus_tests.csv"), index=False)
    # threshold_pairwise.to_csv(os.path.join(OUTPUT_DIR, "threshold_pairwise_tests.csv"), index=False)
    # threshold_normality.to_csv(os.path.join(OUTPUT_DIR, "threshold_normality.csv"), index=False)

    print(f"Saved statistical analysis results to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
