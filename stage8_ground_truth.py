
import os
import cv2
import numpy as np
import pandas as pd

VIDEO = "data/input.mp4"
STAGE5 = "results/stage5_tracking_data.csv"
STAGE6A = "results/stage6A_tracking_data.csv"
OUT_DIR = "results/stage8"

os.makedirs(OUT_DIR, exist_ok=True)

GT_FILE = os.path.join(OUT_DIR, "ground_truth.csv")
EVAL_FILE = os.path.join(OUT_DIR, "iou_evaluation.csv")

# Annotate the whole video at regular intervals.
# Add denser annotations around the known drift period.
frames_to_annotate = sorted(
    set(range(1, 331, 10))
    | set(range(260, 331, 5))
    | {330}
)

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Video frames:", total_frames)
print("Frames selected for annotation:", len(frames_to_annotate))
print("\nDraw a tight rectangle around the vehicle.")
print("Press ENTER or SPACE to confirm the box.")
print("Press C to skip a frame.")

annotations = []

for frame_no in frames_to_annotate:
    if frame_no > total_frames:
        continue

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)
    ok, frame = cap.read()

    if not ok:
        print("Could not read frame:", frame_no)
        continue

    display = frame.copy()
    cv2.putText(
        display,
        f"GROUND TRUTH - Frame {frame_no}/{total_frames}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    print(f"\nAnnotating frame {frame_no}")
    roi = cv2.selectROI(
        "Ground Truth Annotation",
        display,
        showCrosshair=True,
        fromCenter=False
    )

    x, y, w, h = map(int, roi)

    if w <= 0 or h <= 0:
        print("Skipped frame:", frame_no)
        continue

    annotations.append({
        "frame": frame_no,
        "gt_x": x,
        "gt_y": y,
        "gt_width": w,
        "gt_height": h
    })

    # Save after every annotation so progress is not lost.
    pd.DataFrame(annotations).to_csv(GT_FILE, index=False)
    print("Saved annotation:", frame_no, x, y, w, h)

cap.release()
cv2.destroyAllWindows()

if not annotations:
    raise RuntimeError("No ground-truth boxes were annotated.")

gt = pd.read_csv(GT_FILE)


def add_iou(pred, gt_frame):
    px = float(pred["x"])
    py = float(pred["y"])
    pw = float(pred["width"])
    ph = float(pred["height"])

    gx = float(gt_frame["gt_x"])
    gy = float(gt_frame["gt_y"])
    gw = float(gt_frame["gt_width"])
    gh = float(gt_frame["gt_height"])

    left = max(px, gx)
    top = max(py, gy)
    right = min(px + pw, gx + gw)
    bottom = min(py + ph, gy + gh)

    intersection = max(0.0, right - left) * max(
        0.0, bottom - top
    )

    pred_area = max(0.0, pw) * max(0.0, ph)
    gt_area = max(0.0, gw) * max(0.0, gh)
    union = pred_area + gt_area - intersection

    return intersection / union if union > 0 else 0.0


evaluations = []

for stage, path in [
    ("Stage 5", STAGE5),
    ("Stage 6A", STAGE6A)
]:
    if not os.path.exists(path):
        print("Missing prediction file:", path)
        continue

    pred = pd.read_csv(path)

    required = {"frame", "x", "y", "width", "height"}
    if not required.issubset(pred.columns):
        print("Missing bounding-box columns in:", path)
        continue

    merged = gt.merge(
        pred[["frame", "x", "y", "width", "height"]],
        on="frame",
        how="inner"
    )

    if merged.empty:
        print("No matching frames for", stage)
        continue

    merged["stage"] = stage
    merged["IoU"] = merged.apply(
        lambda row: add_iou(row, row), axis=1
    )
    evaluations.append(merged)

if not evaluations:
    raise RuntimeError(
        "No matching ground-truth and prediction frames."
    )

result = pd.concat(evaluations, ignore_index=True)
result.to_csv(EVAL_FILE, index=False)

print("\n========== GROUND-TRUTH EVALUATION ==========")

for stage, group in result.groupby("stage"):
    print(f"\n{stage}")
    print("Annotated frames evaluated:", len(group))
    print("Mean IoU:", round(group["IoU"].mean(), 4))
    print("Median IoU:", round(group["IoU"].median(), 4))
    print("IoU >= 0.50 (%):",
          round((group["IoU"] >= 0.50).mean() * 100, 2))
    print("IoU >= 0.75 (%):",
          round((group["IoU"] >= 0.75).mean() * 100, 2))

    difficult = group[group["frame"] >= 260]
    if not difficult.empty:
        print("Mean IoU, frames 260 onward:",
              round(difficult["IoU"].mean(), 4))

print("\nEvaluation CSV:", EVAL_FILE)
print("Ground-truth CSV:", GT_FILE)
