import os
import re
import pandas as pd

# =====================================================
# ROOT
# =====================================================

ROOT = "/home/nohel/DATA/MultipleMyeloma_analyses/results"

ANALYSES = [
    "longi_summary_all",
    "longi_summary_larger_than_0_3_cubic_cm",
    "longi_summary_larger_than_0_5_cubic_cm"
]

OUTPUT_DIR = os.path.join(ROOT, "supplementary_tables_and_figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# MODEL SORTING
# =====================================================

def extract_dataset_id(name):
    m = re.search(r"Dataset(\d+)", str(name))
    return int(m.group(1)) if m else 9999

# =====================================================
# LOAD ALL DATA
# =====================================================

all_rows = []

for analysis in ANALYSES:
    file_path = os.path.join(ROOT, analysis, "results_all_folds.csv")

    if not os.path.exists(file_path):
        continue

    df = pd.read_csv(file_path)
    df["analysis"] = analysis
    all_rows.append(df)

if not all_rows:
    raise FileNotFoundError("No results files found for supplementary tables")

df = pd.concat(all_rows, ignore_index=True)

# sort models nicely
df["dataset_id"] = df["model_name"].apply(extract_dataset_id)
df = df.sort_values(["dataset_id", "model_type", "fold"])

# =====================================================
# SAVE FULL TABLES
# =====================================================

xlsx_path = os.path.join(OUTPUT_DIR, "MM_full_results_all_thresholds.xlsx")
csv_path = os.path.join(OUTPUT_DIR, "MM_full_results_all_thresholds.csv")

df.to_csv(csv_path, index=False)
df.to_excel(xlsx_path, index=False)

print("Saved:", csv_path)
print("Saved:", xlsx_path)

# =====================================================
# SUMMARY TABLE (MEAN ± STD)
# =====================================================

summary = df.groupby(
    ["analysis", "model_name", "model_type"]
).agg(
    Dice_mean=("Dice", "mean"),
    Dice_std=("Dice", "std"),
    F1_mean=("F1", "mean"),
    F1_std=("F1", "std"),
    NSD_mean=("NSD", "mean"),
    NSD_std=("NSD", "std"),
).reset_index()

summary["dataset_id"] = summary["model_name"].apply(extract_dataset_id)
summary = summary.sort_values(["analysis", "dataset_id"])

summary.to_csv(os.path.join(OUTPUT_DIR, "MM_summary_table.csv"), index=False)
summary.to_excel(os.path.join(OUTPUT_DIR, "MM_summary_table.xlsx"), index=False)

print("Saved:", os.path.join(OUTPUT_DIR, "MM_summary_table.csv"))
print("Saved:", os.path.join(OUTPUT_DIR, "MM_summary_table.xlsx"))
print("Generated supplementary tables only.")