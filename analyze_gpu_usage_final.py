import os
import pandas as pd

############################################
# PATHS
############################################

ROOT = "/home/nohel/DATA/MultipleMyeloma_analyses"

GPU_LOG_DIR = os.path.join(ROOT, "gpu_logs")
TIME_LOG = os.path.join(ROOT, "inference_time_log.csv")

# output directory
OUTPUT_DIR = os.path.join(ROOT, "gpu_analyses_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "gpu_usage_summary.csv")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "gpu_usage_model_means.csv")

############################################
# LOAD RUNTIME LOG
############################################

time_df = pd.read_csv(TIME_LOG)

time_df = time_df.rename(columns={
    "MODEL": "model",
    "FOLD": "fold",
    "DURATION_SEC": "runtime_sec"
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
    time_df[["model", "fold", "runtime_sec"]],
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