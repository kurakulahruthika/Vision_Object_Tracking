
import os
import cv2
import pandas as pd
import numpy as np

VIDEO = "data/input.mp4"
GT_FILE = "results/stage8/ground_truth.csv"
S5_FILE = "results/stage5_tracking_data.csv"
S6_FILE = "results/stage6A_tracking_data.csv"

OUT_DIR = "results/stage8"
IMAGE_DIR = os.path.join(OUT_DIR, "overlays")
os.makedirs(IMAGE_DIR, exist_ok=True)


def iou(box_a, box_b):
    ax, ay, aw, ah = map(float, box_a)
    bx, by, bw, bh = map(float, box_b)

    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)

    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = max(0, aw) * max(0, ah)
    area_b = max(0, bw) * max(0, bh)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


gt = pd.read_csv(GT_FILE)
s5 = pd.read_csv(S5_FILE)
s6 = pd.read_csv(S6_FILE)

# Retain only annotated frames present in BOTH trackers.
gt = gt.rename(columns={
    "gt_x": "gt_x",
    "gt_y": "gt_y",
    "gt_width": "gt_width",
    "gt_height": "gt_height"
})

s5 = s5[["frame", "x", "y", "width", "height"]].rename(
    columns={
        "x": "s5_x", "y": "s5_y",
        "width": "s5_width", "height": "s5_height"
    }
)

s6 = s6[["frame", "x", "y", "width", "height"]].rename(
    columns={
        "x": "s6_x", "y": "s6_y",
        "width": "s6_width", "height": "s6_height"
    }
)

common = gt.merge(s5, on="frame", how="inner")
common = common.merge(s6, on="frame", how="inner")
common = common.sort_values("frame").reset_index(drop=True)

if common.empty:
    raise RuntimeError("No common annotated frames found.")

common["iou_stage5"] = common.apply(
    lambda r: iou(
        [r.gt_x, r.gt_y, r.gt_width, r.gt_height],
        [r.s5_x, r.s5_y, r.s5_width, r.s5_height]
    ), axis=1
)

common["iou_stage6A"] = common.apply(
    lambda r: iou(
        [r.gt_x, r.gt_y, r.gt_width, r.gt_height],
        [r.s6_x, r.s6_y, r.s6_width, r.s6_height]
    ), axis=1
)

common["center_error_stage5"] = common.apply(
    lambda r: np.hypot(
        (r.s5_x + r.s5_width / 2) -
        (r.gt_x + r.gt_width / 2),
        (r.s5_y + r.s5_height / 2) -
        (r.gt_y + r.gt_height / 2)
    ), axis=1
)

common["center_error_stage6A"] = common.apply(
    lambda r: np.hypot(
        (r.s6_x + r.s6_width / 2) -
        (r.gt_x + r.gt_width / 2),
        (r.s6_y + r.s6_height / 2) -
        (r.gt_y + r.gt_height / 2)
    ), axis=1
)

csv_path = os.path.join(OUT_DIR, "common_frame_comparison.csv")
common.to_csv(csv_path, index=False)

print("\n========== FAIR COMMON-FRAME COMPARISON ==========")
print("Annotated frames available in both trackers:", len(common))

for stage, col, err_col in [
    ("Stage 5", "iou_stage5", "center_error_stage5"),
    ("Stage 6A", "iou_stage6A", "center_error_stage6A")
]:
    print(f"\n{stage}")
    print("Mean IoU:", round(common[col].mean(), 4))
    print("Median IoU:", round(common[col].median(), 4))
    print("IoU >= 0.50 (%):",
          round((common[col] >= 0.50).mean() * 100, 2))
    print("Mean center error (pixels):",
          round(common[err_col].mean(), 2))

    late = common[common["frame"] >= 260]
    if not late.empty:
        print("Late-video mean IoU:",
              round(late[col].mean(), 4))
        print("Late-video mean center error:",
              round(late[err_col].mean(), 2))

# Print individual results for diagnosis.
print("\n--- Per-frame IoU ---")
print(common[
    ["frame", "iou_stage5", "iou_stage6A",
     "center_error_stage5", "center_error_stage6A"]
].round(3).to_string(index=False))

# Draw ground truth and predictions on selected video frames.
cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError("Could not open original video.")

wanted = {11, 101, 201, 260, 270, 280, 295, 305, 310, 320}
selected = common[common["frame"].isin(wanted)]

for _, r in selected.iterrows():
    frame_no = int(r["frame"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)
    ok, frame = cap.read()

    if not ok:
        continue

    def draw(row, prefix, color, label):
        x = int(row[prefix + "_x"])
        y = int(row[prefix + "_y"])
        w = int(row[prefix + "_width"])
        h = int(row[prefix + "_height"])

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame, label, (max(0, x), max(20, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    draw(r, "gt", (0, 255, 0), "GROUND TRUTH")
    draw(r, "s5", (255, 0, 0), "STAGE 5")
    draw(r, "s6", (0, 0, 255), "STAGE 6A")

    path = os.path.join(IMAGE_DIR, f"frame_{frame_no:03d}.jpg")
    cv2.imwrite(path, frame)
    print("Saved overlay:", path)

cap.release()

print("\nComparison CSV:", csv_path)
print("Overlay images:", IMAGE_DIR)
