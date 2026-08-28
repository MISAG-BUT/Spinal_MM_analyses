import os
import json
import glob
import re
import pandas as pd

############################################
# PATHS
############################################

ROOT = "/home/nohel/DATA/MultipleMyeloma_analyses"
TRAINING_RESULTS_ROOT = "/home/nohel/DATA/nnUNet_results"

GPU_LOG_DIR = os.path.join(ROOT, "gpu_logs")
TIME_LOG = os.path.join(ROOT, "inference_time_log.csv")

# output directory
OUTPUT_DIR = os.path.join(ROOT, "gpu_analyses_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "gpu_usage_summary.csv")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "gpu_usage_model_means.csv")
ARTICLE_SUMMARY_FILE = os.path.join(OUTPUT_DIR, "gpu_usage_article_summary.csv")
TRAINING_GPU_FILE = os.path.join(OUTPUT_DIR, "training_gpu_usage.csv")
TRAINING_GPU_SUMMARY_FILE = os.path.join(OUTPUT_DIR, "training_gpu_usage_summary.csv")

############################################
# LOAD RUNTIME LOG
############################################

time_df = pd.read_csv(TIME_LOG)

time_df = time_df.rename(columns={
    "MODEL": "model",
    "FOLD": "fold",
    "DURATION_SEC": "runtime_sec",
    "PARAMETERS": "parameters"
})

############################################
# PROCESS GPU LOG FILES
############################################

rows = []

for file in os.listdir(GPU_LOG_DIR):

    if not file.endswith(".csv"):
        continue

    path = os.path.join(GPU_LOG_DIR, file)

    name = file.replace(".csv", "")

    model, fold = name.split("_fold_")
    fold = int(fold)

    # load GPU log
    df = pd.read_csv(path, skipinitialspace=True)

    # remove units and convert to numeric
    df["utilization.gpu [%]"] = df["utilization.gpu [%]"].str.replace(" %", "").astype(float)
    df["utilization.memory [%]"] = df["utilization.memory [%]"].str.replace(" %", "").astype(float)
    df["memory.used [MiB]"] = df["memory.used [MiB]"].str.replace(" MiB", "").astype(float)
    df["power.draw [W]"] = df["power.draw [W]"].str.replace(" W", "").astype(float)

    avg_gpu = df["utilization.gpu [%]"].mean()
    max_gpu = df["utilization.gpu [%]"].max()

    avg_mem = df["memory.used [MiB]"].mean()
    max_mem = df["memory.used [MiB]"].max()

    avg_power = df["power.draw [W]"].mean()

    energy_wh = df["power.draw [W]"].sum() / 3600

    rows.append({
        "model": model,
        "fold": fold,
        "avg_gpu_util_percent": avg_gpu,
        "max_gpu_util_percent": max_gpu,
        "avg_memory_MiB": avg_mem,
        "max_memory_MiB": max_mem,
        "avg_power_W": avg_power,
        "energy_Wh": energy_wh
    })

gpu_df = pd.DataFrame(rows)

############################################
# MERGE WITH RUNTIME
############################################

merged = gpu_df.merge(
    time_df[["model", "fold", "runtime_sec", "parameters"]],
    on=["model", "fold"],
    how="left"
)

# sort results
merged = merged.sort_values(by=["model", "fold"]).reset_index(drop=True)

merged.to_csv(OUTPUT_FILE, index=False)

print("Saved:", OUTPUT_FILE)

############################################
# MODEL LEVEL SUMMARY
############################################

model_summary = merged.groupby("model").agg({

    "runtime_sec": "mean",
    "parameters": "mean",
    "avg_gpu_util_percent": "mean",
    "max_gpu_util_percent": "max",
    "avg_memory_MiB": "mean",
    "max_memory_MiB": "max",
    "avg_power_W": "mean",
    "energy_Wh": "sum"

}).reset_index()

# sort models
model_summary = model_summary.sort_values(by="model").reset_index(drop=True)

model_summary.to_csv(SUMMARY_FILE, index=False)

print("Saved:", SUMMARY_FILE)

############################################
# ARTICLE SUMMARY BY MODEL GROUP
############################################

article_metrics = [
    "runtime_sec",
    "avg_gpu_util_percent",
    "max_gpu_util_percent",
    "avg_memory_MiB",
    "max_memory_MiB",
    "avg_power_W",
    "energy_Wh",
]

article_summary_rows = []
article_groups = {
    "LOO (700-707)": set(range(700, 708)),
    "All together (708)": {708},
    "Single input (709-716)": set(range(709, 717)),
    "All VMI (717)": {717},
    "All CaSupp (718)": {718},
}

article_source = merged[
    ~merged["model"].astype(str).str.contains("zero.?input", case=False, regex=True, na=False)
].copy()
article_source["dataset_id"] = article_source["model"].str.extract(r"Dataset(\d+)")[0].astype(float)

for group_name, dataset_ids in article_groups.items():
    group_data = article_source[article_source["dataset_id"].isin(dataset_ids)]
    if group_data.empty:
        continue

    summary_row = {"model_group": group_name}
    parameter_values = pd.to_numeric(group_data["parameters"], errors="coerce").dropna()
    summary_row["parameters"] = parameter_values.median() if not parameter_values.empty else None
    for metric in article_metrics:
        values = pd.to_numeric(group_data[metric], errors="coerce").dropna()
        if values.empty:
            summary_row[f"{metric}_median"] = None
            summary_row[f"{metric}_range_min"] = None
            summary_row[f"{metric}_range_max"] = None
        else:
            summary_row[f"{metric}_median"] = values.median()
            summary_row[f"{metric}_range_min"] = values.min()
            summary_row[f"{metric}_range_max"] = values.max()
    article_summary_rows.append(summary_row)

article_summary = pd.DataFrame(article_summary_rows).round(3)
article_summary.to_csv(ARTICLE_SUMMARY_FILE, index=False)

print("Saved:", ARTICLE_SUMMARY_FILE)

############################################
# LOAD GPU TYPES USED DURING TRAINING
############################################

training_gpu_rows = []

for dataset_id in range(700, 719):
    model_pattern = os.path.join(
        TRAINING_RESULTS_ROOT,
        f"Dataset{dataset_id}_MM_Lesion_seg_*",
        "nnUNetTrainer__nnUNetPlans__3d_fullres",
        "fold_*",
        "debug.json",
    )

    for debug_file in sorted(glob.glob(model_pattern)):
        fold_match = re.search(r"/fold_(\d+)/debug\.json$", debug_file)
        if not fold_match:
            continue

        fold = int(fold_match.group(1))
        if fold not in range(5):
            continue

        fold_dir = os.path.dirname(debug_file)
        model_dir = os.path.dirname(os.path.dirname(fold_dir))
        model = os.path.basename(model_dir)

        with open(debug_file) as fh:
            debug_data = json.load(fh)

        epoch_times = []
        for training_log in sorted(glob.glob(os.path.join(fold_dir, "training_log_*.txt"))):
            with open(training_log) as fh:
                log_text = fh.read()
            epoch_times.extend(
                float(value)
                for value in re.findall(r"Epoch time:\s*([0-9]+(?:\.[0-9]+)?)\s*s", log_text)
            )

        training_gpu_rows.append({
            "dataset_id": dataset_id,
            "model": model,
            "fold": fold,
            "gpu_name": debug_data.get("gpu_name"),
            "mean_epoch_time_sec": sum(epoch_times) / len(epoch_times) if epoch_times else None,
            "epoch_count": len(epoch_times),
        })

training_gpu_df = pd.DataFrame(training_gpu_rows)
training_gpu_df = training_gpu_df.sort_values(
    by=["dataset_id", "model", "fold"]
).reset_index(drop=True)
training_gpu_df.to_csv(TRAINING_GPU_FILE, index=False)

training_gpu_summary = (
    training_gpu_df.groupby(["dataset_id", "model", "gpu_name"], dropna=False)
    .agg(
        folds=("fold", "nunique"),
        mean_epoch_time_sec=("mean_epoch_time_sec", "mean"),
        epoch_count=("epoch_count", "sum"),
    )
    .reset_index()
    .sort_values(by=["dataset_id", "model", "gpu_name"])
    .reset_index(drop=True)
)
training_gpu_summary.to_csv(TRAINING_GPU_SUMMARY_FILE, index=False)

print("Saved:", TRAINING_GPU_FILE)
print("Saved:", TRAINING_GPU_SUMMARY_FILE)