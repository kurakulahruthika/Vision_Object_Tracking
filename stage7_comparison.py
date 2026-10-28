
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# STAGE 7: COMPARATIVE PERFORMANCE ANALYSIS
# --------------------------------------------------

OUT_DIR = os.path.join("results", "stage7")
os.makedirs(OUT_DIR, exist_ok=True)

FILES = {
    "Stage 5": "results/stage5_tracking_data.csv",
    "Stage 6A": "results/stage6A_tracking_data.csv",
    "Final motion": "results/final_motion_data.csv",
}

data = {}

for name, path in FILES.items():
    if not os.path.exists(path):
        print(f"WARNING: Missing file: {path}")
        continue

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    data[name] = df

    print(f"\n{name}")
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())

if not data:
    raise FileNotFoundError("No tracking CSV files were found.")

# --------------------------------------------------
# 1. COMMON METRIC SUMMARY
# --------------------------------------------------

rows = []

for name, df in data.items():
    row = {
        "dataset": name,
        "frames": len(df),
    }

    # Match equivalent concepts while preserving
    # the original source column names.
    feature_col = next(
        (c for c in ["target_features", "target_points"]
         if c in df.columns),
        None
    )

    inlier_col = next(
        (c for c in ["target_inliers", "ransac_inliers"]
         if c in df.columns),
        None
    )

    confidence_col = (
        "confidence" if "confidence" in df.columns else None
    )

    if feature_col:
        values = pd.to_numeric(df[feature_col], errors="coerce")
        row["feature_column"] = feature_col
        row["mean_features"] = values.mean()
        row["min_features"] = values.min()
        row["max_features"] = values.max()

    if inlier_col:
        values = pd.to_numeric(df[inlier_col], errors="coerce")
        row["inlier_column"] = inlier_col
        row["mean_inliers"] = values.mean()

    if confidence_col:
        values = pd.to_numeric(
            df[confidence_col], errors="coerce"
        )
        row["mean_confidence"] = values.mean()
        row["min_confidence"] = values.min()

    # Keep each dataset's status labels separate.
    status_col = next(
        (c for c in ["tracking_status", "status"]
         if c in df.columns),
        None
    )

    if status_col:
        row["status_column"] = status_col
        row["status_labels"] = (
            df[status_col].fillna("MISSING")
            .astype(str).nunique()
        )

    # Motion metrics, when available.
    for col in ["velocity", "speed", "acceleration"]:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            row[f"mean_{col}"] = values.mean()

    if "kalman_used" in df.columns:
        kalman = pd.to_numeric(
            df["kalman_used"], errors="coerce"
        )
        row["kalman_used_frames"] = int((kalman == 1).sum())

    rows.append(row)

summary = pd.DataFrame(rows)
summary_path = os.path.join(OUT_DIR, "stage7_metric_summary.csv")
summary.to_csv(summary_path, index=False)

print("\n========== STAGE 7 METRIC SUMMARY ==========")
print(summary.round(3).to_string(index=False))
print("\nSaved:", summary_path)

# --------------------------------------------------
# 2. MEAN FEATURE COUNTS
# --------------------------------------------------

feature_rows = []

for name, df in data.items():
    col = next(
        (c for c in ["target_features", "target_points"]
         if c in df.columns),
        None
    )
    if col:
        vals = pd.to_numeric(df[col], errors="coerce")
        feature_rows.append((name, vals.mean()))

if feature_rows:
    names, means = zip(*feature_rows)
    plt.figure(figsize=(8, 5))
    plt.bar(names, means)
    plt.ylabel("Mean recorded target feature count")
    plt.title("Stage 7: Mean Target Feature Count")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "mean_features.png"), dpi=200
    )
    plt.close()

# --------------------------------------------------
# 3. MEAN CONFIDENCE
# --------------------------------------------------

confidence_rows = []

for name, df in data.items():
    if "confidence" in df.columns:
        vals = pd.to_numeric(df["confidence"], errors="coerce")
        confidence_rows.append((name, vals.mean()))

if confidence_rows:
    names, means = zip(*confidence_rows)
    plt.figure(figsize=(8, 5))
    plt.bar(names, means)
    plt.ylabel("Mean recorded confidence")
    plt.title("Stage 7: Mean Tracking Confidence")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "mean_confidence.png"), dpi=200
    )
    plt.close()

# --------------------------------------------------
# 4. CONFIDENCE OVER TIME
# --------------------------------------------------

plt.figure(figsize=(11, 5))

for name, df in data.items():
    if "confidence" not in df.columns:
        continue

    frame_col = "frame" if "frame" in df.columns else None
    if frame_col is None:
        continue

    frames = pd.to_numeric(df[frame_col], errors="coerce")
    confidence = pd.to_numeric(
        df["confidence"], errors="coerce"
    )

    plt.plot(frames, confidence, label=name, linewidth=1)

plt.xlabel("Frame")
plt.ylabel("Recorded confidence")
plt.title("Tracking Confidence Over Time")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(OUT_DIR, "confidence_over_time.png"), dpi=200
)
plt.close()

# --------------------------------------------------
# 5. STAGE 6A TRACKING STATUS COUNTS
# --------------------------------------------------

if "Stage 6A" in data:
    df = data["Stage 6A"]
    if "tracking_status" in df.columns:
        counts = df["tracking_status"].fillna(
            "MISSING"
        ).value_counts()

        counts.to_csv(
            os.path.join(OUT_DIR, "stage6A_status_counts.csv"),
            header=["frames"]
        )

        plt.figure(figsize=(8, 5))
        plt.bar(counts.index.astype(str), counts.values)
        plt.ylabel("Number of frames")
        plt.title("Stage 6A: Tracking Status Distribution")
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUT_DIR, "stage6A_status.png"), dpi=200
        )
        plt.close()

# --------------------------------------------------
# 6. FINAL MOTION ANALYSIS
# --------------------------------------------------

if "Final motion" in data:
    df = data["Final motion"]

    if "frame" in df.columns and "speed" in df.columns:
        frames = pd.to_numeric(df["frame"], errors="coerce")
        speed = pd.to_numeric(df["speed"], errors="coerce")

        plt.figure(figsize=(11, 5))
        plt.plot(frames, speed, linewidth=1)
        plt.xlabel("Frame")
        plt.ylabel("Recorded speed")
        plt.title("Final Motion Analysis: Speed Over Time")
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUT_DIR, "final_speed_over_time.png"),
            dpi=200
        )
        plt.close()

print("\nStage 7 comparison completed.")
print("All available outputs saved in:", OUT_DIR)
