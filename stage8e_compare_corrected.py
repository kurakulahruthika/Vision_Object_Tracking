
import os
import cv2
import numpy as np
import pandas as pd

# Input files
GROUND_TRUTH = "results/stage8/ground_truth.csv"

TRACKERS = {
    "Stage 5": "results/stage5_tracking_data.csv",
    "Original Stage 6A": "results/stage6A_tracking_data.csv",
    "Corrected Stage 6A": "results/stage6A_corrected_tracking_data.csv",
}

OUTPUT_DIR = "results/stage8"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def calculate_iou(box1, box2):
    """Boxes are represented as x, y, width, height."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    left = max(x1, x2)
    top = max(y1, y2)
    right = min(x1 + w1, x2 + w2)
    bottom = min(y1 + h1, y2 + h2)

    intersection = max(0, right - left) * max(0, bottom - top)
    area1 = max(0, w1) * max(0, h1)
    area2 = max(0, w2) * max(0, h2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def load_boxes(path, x_col, y_col, w_col, h_col):
    df = pd.read_csv(path)

    required = ["frame", x_col, y_col, w_col, h_col]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"{path} is missing columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    df = df[required].copy()
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce")

    for col in [x_col, y_col, w_col, h_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required)
    df = df.drop_duplicates(subset="frame", keep="last")

    return df.set_index("frame")


def evaluate_tracker(gt, pred, prefix):
    common = gt.index.intersection(pred.index)

    if len(common) == 0:
        print(f"WARNING: No common frames for {prefix}")
        return None

    records = []

    for frame in sorted(common):
        g = gt.loc[frame]
        p = pred.loc[frame]

        gt_box = (
            float(g["gt_x"]), float(g["gt_y"]),
            float(g["gt_width"]), float(g["gt_height"])
        )
        pred_box = (
            float(p["x"]), float(p["y"]),
            float(p["width"]), float(p["height"])
        )

        gt_cx = gt_box[0] + gt_box[2] / 2
        gt_cy = gt_box[1] + gt_box[3] / 2
        pred_cx = pred_box[0] + pred_box[2] / 2
        pred_cy = pred_box[1] + pred_box[3] / 2

        center_error = float(np.hypot(
            gt_cx - pred_cx, gt_cy - pred_cy
        ))

        records.append({
            "frame": int(frame),
            f"iou_{prefix}": calculate_iou(gt_box, pred_box),
            f"center_error_{prefix}": center_error,
        })

    return pd.DataFrame(records).set_index("frame")


# Load ground truth
gt = load_boxes(
    GROUND_TRUTH, "gt_x", "gt_y", "gt_width", "gt_height"
)

# Load and evaluate each tracker
all_results = []
summary = []

for name, path in TRACKERS.items():
    if not os.path.exists(path):
        print(f"WARNING: File not found: {path}")
        continue

    # Tracker CSVs use x, y, width, height.
    pred = load_boxes(path, "x", "y", "width", "height")
    prefix = name.lower().replace(" ", "_")

    result = evaluate_tracker(gt, pred, prefix)

    if result is None:
        continue

    result = result.reset_index()
    result["tracker"] = name
    all_results.append(result)

    iou_col = f"iou_{prefix}"
    error_col = f"center_error_{prefix}"

    late = result[result["frame"] >= 260]

    summary.append({
        "Tracker": name,
        "Common annotated frames": len(result),
        "Mean IoU": result[iou_col].mean(),
        "Median IoU": result[iou_col].median(),
        "IoU >= 0.50 (%)":
            (result[iou_col] >= 0.50).mean() * 100,
        "Mean center error (px)": result[error_col].mean(),
        "Late mean IoU": late[iou_col].mean(),
        "Late center error (px)": late[error_col].mean(),
    })

if not summary:
    raise RuntimeError(
        "No tracker was evaluated. Check the CSV paths and columns."
    )

summary_df = pd.DataFrame(summary)
summary_path = os.path.join(OUTPUT_DIR, "stage8e_summary.csv")
summary_df.to_csv(summary_path, index=False)

# Save per-frame results separately for each tracker.
per_frame = pd.concat(all_results, ignore_index=True)
per_frame_path = os.path.join(
    OUTPUT_DIR, "stage8e_per_frame_results.csv"
)
per_frame.to_csv(per_frame_path, index=False)

print("\n========== CORRECTED TRACKER EVALUATION ==========")
print(summary_df.round(4).to_string(index=False))
print(f"\nSummary saved to: {summary_path}")
print(f"Per-frame results saved to: {per_frame_path}")
