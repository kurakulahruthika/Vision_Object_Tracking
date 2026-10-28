
import os
import pandas as pd
import matplotlib.pyplot as plt

base = "results/stage8"

gt = pd.read_csv(os.path.join(base, "ground_truth.csv"))
trackers = {
    "Stage 5": "results/stage5_tracking_data.csv",
    "Original Stage 6A": "results/stage6A_tracking_data.csv",
    "Corrected Stage 6A": "results/stage6A_corrected_tracking_data.csv",
}

# Verify frame overlap across all trackers and ground truth.
frame_sets = {
    name: set(pd.read_csv(path)["frame"].dropna().astype(int))
    for name, path in trackers.items()
}
gt_frames = set(gt["frame"].dropna().astype(int))

common = gt_frames.copy()
for frames in frame_sets.values():
    common &= frames

print("Ground-truth annotated frames:", len(gt_frames))
for name, frames in frame_sets.items():
    print(f"{name} annotated-frame overlap: {len(gt_frames & frames)}")

print("Frames common to ALL trackers and ground truth:", len(common))
if common:
    print("Common frame numbers:", sorted(common))

# Load existing evaluation results.
df = pd.read_csv(os.path.join(base, "stage8e_per_frame_results.csv"))

styles = {
    "Stage 5": ("iou_stage_5", "center_error_stage_5"),
    "Original Stage 6A": (
        "iou_original_stage_6a",
        "center_error_original_stage_6a",
    ),
    "Corrected Stage 6A": (
        "iou_corrected_stage_6a",
        "center_error_corrected_stage_6a",
    ),
}

# Use the exact column names actually saved by Stage 8E.
for name, group in df.groupby("tracker"):
    print(f"\n{name}: {len(group)} evaluation rows")
    print("Available metric columns:",
          [c for c in group.columns if c.startswith("iou_")
           or c.startswith("center_error_")])

# Plot each tracker's available IoU values.
plt.figure()
for name, group in df.groupby("tracker"):
    iou_cols = [c for c in group.columns if c.startswith("iou_")]
    if iou_cols:
        values = group.dropna(subset=iou_cols)
        plt.plot(values["frame"], values[iou_cols[0]],
                 marker=".", markersize=3, label=name)

plt.axvline(260, linestyle="--", label="Late-video region")
plt.xlabel("Frame")
plt.ylabel("Intersection over Union (IoU)")
plt.title("Tracker Accuracy Across Annotated Frames")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(base, "stage8f_iou_comparison.png"), dpi=160)
plt.close()

# Plot center error.
plt.figure()
for name, group in df.groupby("tracker"):
    error_cols = [
        c for c in group.columns if c.startswith("center_error_")
    ]
    if error_cols:
        values = group.dropna(subset=error_cols)
        plt.plot(values["frame"], values[error_cols[0]],
                 marker=".", markersize=3, label=name)

plt.axvline(260, linestyle="--", label="Late-video region")
plt.xlabel("Frame")
plt.ylabel("Center error (pixels)")
plt.title("Tracking Center Error Across Annotated Frames")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(base, "stage8f_center_error_comparison.png"), dpi=160
)
plt.close()

print("\nGraphs saved in results/stage8/")
