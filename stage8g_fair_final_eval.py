
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = "results/stage8"
os.makedirs(OUT, exist_ok=True)

GT_FILE = f"{OUT}/ground_truth.csv"

TRACKERS = {
    "Stage 5 (Baseline)": "results/stage5_tracking_data.csv",
    "Stage 6A (Original)": "results/stage6A_tracking_data.csv",
    "Stage 6A (Corrected)": "results/stage6A_corrected_tracking_data.csv",
    "Stage 9A (Transition)": "results/stage9A_transition_aware_data.csv",
    "Stage 10 (Master)": "results/stage10_master_tracking_data.csv",
}


def load_csv(path, columns):
    df = pd.read_csv(path)
    required = ["frame"] + columns
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    df = df[required].copy()
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().drop_duplicates("frame", keep="last")
    df["frame"] = df["frame"].astype(int)
    return df.set_index("frame")


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    left, top = max(ax, bx), max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)

    inter = max(0, right - left) * max(0, bottom - top)
    area_a = max(0, aw) * max(0, ah)
    area_b = max(0, bw) * max(0, bh)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


gt = load_csv(
    GT_FILE, ["gt_x", "gt_y", "gt_width", "gt_height"]
)

predictions = {}
for name, path in TRACKERS.items():
    predictions[name] = load_csv(
        path, ["x", "y", "width", "height"]
    )

# Require a frame to exist in GT and every tracker.
common = set(gt.index)
for pred in predictions.values():
    common &= set(pred.index)

common = sorted(common)
if not common:
    raise RuntimeError("No common frames found.")

print(f"Fair comparison frames: {len(common)}")
print(f"Frames: {common}")

records = []
for frame in common:
    g = gt.loc[frame]
    gt_box = (
        g["gt_x"], g["gt_y"], g["gt_width"], g["gt_height"]
    )
    gt_cx = gt_box[0] + gt_box[2] / 2
    gt_cy = gt_box[1] + gt_box[3] / 2

    row = {"frame": frame}

    for name, pred in predictions.items():
        p = pred.loc[frame]
        box = (p["x"], p["y"], p["width"], p["height"])

        cx = box[0] + box[2] / 2
        cy = box[1] + box[3] / 2
        key = name.lower().replace(" ", "_")

        row[f"iou_{key}"] = iou(gt_box, box)
        row[f"center_error_{key}"] = float(
            np.hypot(gt_cx - cx, gt_cy - cy)
        )

    records.append(row)

detail = pd.DataFrame(records)
detail.to_csv(f"{OUT}/stage8g_fair_per_frame.csv", index=False)

summary = []
for name in TRACKERS:
    key = name.lower().replace(" ", "_")
    iou_col = f"iou_{key}"
    err_col = f"center_error_{key}"

    late = detail[detail["frame"] >= 260]

    summary.append({
        "Tracker": name,
        "Frames": len(detail),
        "Mean IoU": detail[iou_col].mean(),
        "Median IoU": detail[iou_col].median(),
        "IoU >= 0.50 (%)":
            (detail[iou_col] >= 0.50).mean() * 100,
        "Mean center error (px)": detail[err_col].mean(),
        "Late mean IoU": late[iou_col].mean(),
        "Late center error (px)": late[err_col].mean(),
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv(f"{OUT}/stage8g_fair_summary.csv", index=False)

print("\n========== FAIR FINAL COMPARISON ==========")
print(summary_df.round(4).to_string(index=False))

# IoU graph
plt.figure()
for name in TRACKERS:
    key = name.lower().replace(" ", "_")
    plt.plot(
        detail["frame"], detail[f"iou_{key}"],
        marker=".", markersize=3, label=name
    )
plt.axvline(260, linestyle="--", label="Late-video region")
plt.xlabel("Frame")
plt.ylabel("IoU")
plt.title("Fair Tracker Comparison — IoU")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/stage8g_iou.png", dpi=160)
plt.close()

# Center error graph
plt.figure()
for name in TRACKERS:
    key = name.lower().replace(" ", "_")
    plt.plot(
        detail["frame"], detail[f"center_error_{key}"],
        marker=".", markersize=3, label=name
    )
plt.axvline(260, linestyle="--", label="Late-video region")
plt.xlabel("Frame")
plt.ylabel("Center error (pixels)")
plt.title("Fair Tracker Comparison — Center Error")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/stage8g_center_error.png", dpi=160)
plt.close()

print("\nSaved:")
print(f"{OUT}/stage8g_fair_summary.csv")
print(f"{OUT}/stage8g_fair_per_frame.csv")
print(f"{OUT}/stage8g_iou.png")
print(f"{OUT}/stage8g_center_error.png")
