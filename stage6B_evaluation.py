
import os
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# STAGE 6B: QUANTITATIVE EVALUATION
# =====================================================

CSV_FILE = "results/stage6A_tracking_data.csv"
OUTPUT_DIR = "results/stage6B"

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(
        f"CSV not found: {CSV_FILE}\n"
        "Run Stage 6A first and check its output path."
    )

df = pd.read_csv(CSV_FILE)

required_columns = [
    "frame",
    "target_points",
    "ransac_inliers",
    "confidence",
    "velocity",
    "tracking_status",
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(f"Missing CSV columns: {missing}")

if df.empty:
    raise ValueError("The tracking CSV contains no data.")

# Convert numeric columns safely
numeric_columns = [
    "frame",
    "target_points",
    "ransac_inliers",
    "confidence",
    "velocity",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(subset=["frame"]).sort_values("frame")

# =====================================================
# CALCULATE METRICS
# =====================================================

total_frames = len(df)

status_counts = df["tracking_status"].value_counts()

tracking_frames = int(
    (df["tracking_status"] == "TRACKING").sum()
)

tracking_percentage = (
    100 * tracking_frames / total_frames
    if total_frames else 0
)

summary = {
    "Total logged frames": total_frames,
    "Frames labelled TRACKING": tracking_frames,
    "TRACKING-labelled percentage (%)":
        round(tracking_percentage, 2),
    "Mean target feature points":
        round(df["target_points"].mean(), 2),
    "Minimum target feature points":
        int(df["target_points"].min()),
    "Maximum target feature points":
        int(df["target_points"].max()),
    "Mean RANSAC inliers":
        round(df["ransac_inliers"].mean(), 2),
    "Mean confidence":
        round(df["confidence"].mean(), 3),
    "Mean reported velocity (pixels/frame)":
        round(df["velocity"].mean(), 2),
}

summary_file = os.path.join(
    OUTPUT_DIR, "evaluation_summary.txt"
)

with open(summary_file, "w", encoding="utf-8") as f:
    f.write("STAGE 6B - EVALUATION SUMMARY\n")
    f.write("=" * 40 + "\n")

    for metric, value in summary.items():
        f.write(f"{metric}: {value}\n")

    f.write("\nTracking status counts:\n")
    f.write(status_counts.to_string())
    f.write("\n")

print("\nSTAGE 6B EVALUATION")
print("=" * 40)

for metric, value in summary.items():
    print(f"{metric}: {value}")

print("\nTracking status counts:")
print(status_counts)

# =====================================================
# GRAPH 1: FEATURE POINT STABILITY
# =====================================================

plt.figure(figsize=(12, 5))
plt.plot(
    df["frame"],
    df["target_points"],
    label="Target feature points"
)
plt.xlabel("Frame")
plt.ylabel("Feature points")
plt.title("Stage 6A - Target Feature Stability")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "feature_stability.png"),
    dpi=200
)
plt.close()

# =====================================================
# GRAPH 2: CONFIDENCE OVER TIME
# =====================================================

plt.figure(figsize=(12, 5))
plt.plot(df["frame"], df["confidence"])
plt.axhline(
    0.65, linestyle="--",
    label="High-confidence threshold"
)
plt.axhline(
    0.35, linestyle=":",
    label="Low-confidence threshold"
)
plt.xlabel("Frame")
plt.ylabel("Confidence")
plt.title("Stage 6A - Tracking Confidence")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "confidence_over_time.png"),
    dpi=200
)
plt.close()

# =====================================================
# GRAPH 3: RANSAC INLIERS
# =====================================================

plt.figure(figsize=(12, 5))
plt.plot(df["frame"], df["ransac_inliers"])
plt.xlabel("Frame")
plt.ylabel("RANSAC inliers")
plt.title("Stage 6A - Motion Estimation Reliability")
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "ransac_inliers.png"),
    dpi=200
)
plt.close()

# =====================================================
# GRAPH 4: TRACKING STATUS COUNTS
# =====================================================

plt.figure(figsize=(9, 5))
status_counts.plot(kind="bar")
plt.xlabel("Tracking status")
plt.ylabel("Number of frames")
plt.title("Stage 6A - Tracking Status Distribution")
plt.xticks(rotation=30, ha="right")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "tracking_status.png"),
    dpi=200
)
plt.close()

# Save a copy of the data used for evaluation
df.to_csv(
    os.path.join(OUTPUT_DIR, "evaluated_tracking_data.csv"),
    index=False
)

print("\nEvaluation complete!")
print("Results saved in:", OUTPUT_DIR)
