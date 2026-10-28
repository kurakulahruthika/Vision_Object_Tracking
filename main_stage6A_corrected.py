import cv2
import numpy as np
import csv
import os
import time

# ============================================================
# SETTINGS
# ============================================================
INPUT_VIDEO = "data/input.mp4"
OUTPUT_VIDEO = "results/stage6A_corrected_tracking.mp4"
CSV_FILE = "results/stage6A_corrected_tracking_data.csv"
os.makedirs("results", exist_ok=True)

# ============================================================
# OPTICAL FLOW / FEATURE PARAMETERS
# ============================================================
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

FEATURE_PARAMS = dict(
    maxCorners=200,
    qualityLevel=0.01,
    minDistance=7,
    blockSize=7,
)

MIN_TARGET_POINTS = 8
MIN_RANSAC_INLIERS = 6
FB_THRESHOLD = 1.5
MAX_CENTER_JUMP = 100.0
MIN_SCALE_PER_FRAME = 0.75
MAX_SCALE_PER_FRAME = 1.33
MIN_BOX_SIZE = 12
MAX_BOX_WIDTH_FRACTION = 0.60
MAX_BOX_HEIGHT_FRACTION = 0.80
REDETECT_MARGIN = 1.5

# Confidence threshold for a measured update.
HIGH_CONFIDENCE = 0.48

# ============================================================
# HELPERS
# ============================================================
def clip_box(cx, cy, bw, bh, frame_width, frame_height):
    """Return a valid image-clipped box (x, y, w, h)."""
    bw = int(np.clip(round(bw), MIN_BOX_SIZE, frame_width * MAX_BOX_WIDTH_FRACTION))
    bh = int(np.clip(round(bh), MIN_BOX_SIZE, frame_height * MAX_BOX_HEIGHT_FRACTION))
    x = int(round(cx - bw / 2))
    y = int(round(cy - bh / 2))
    x = max(0, min(x, frame_width - bw))
    y = max(0, min(y, frame_height - bh))
    return x, y, bw, bh


def detect_features(gray, box):
    """Detect features only inside the current target box."""
    x, y, bw, bh = box
    mask = np.zeros(gray.shape, dtype=np.uint8)
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(gray.shape[1], x + bw)
    y2 = min(gray.shape[0], y + bh)
    if x2 <= x1 or y2 <= y1:
        return None
    mask[y1:y2, x1:x2] = 255
    return cv2.goodFeaturesToTrack(gray, mask=mask, **FEATURE_PARAMS)


def transform_box(M, box):
    """Transform all four box corners by an affine model and return new box."""
    x, y, bw, bh = box
    corners = np.array(
        [[[x, y]], [[x + bw, y]], [[x + bw, y + bh]], [[x, y + bh]]],
        dtype=np.float32,
    )
    transformed = cv2.transform(corners, M).reshape(-1, 2)
    xmin, ymin = transformed.min(axis=0)
    xmax, ymax = transformed.max(axis=0)
    cx = float((xmin + xmax) / 2.0)
    cy = float((ymin + ymax) / 2.0)
    nw = float(xmax - xmin)
    nh = float(ymax - ymin)
    return cx, cy, nw, nh


def clamp_scale(new_size, old_size, minimum, maximum):
    if old_size <= 0 or not np.isfinite(new_size):
        return old_size
    ratio = new_size / old_size
    ratio = float(np.clip(ratio, minimum, maximum))
    return old_size * ratio


# ============================================================
# OPEN VIDEO AND SELECT TARGET
# ============================================================
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise RuntimeError(f"Could not open input video: {INPUT_VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

ret, first_frame = cap.read()
if not ret:
    cap.release()
    raise RuntimeError("Could not read the first frame.")

first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
print("Select the target vehicle in the first frame, then press ENTER/SPACE.")
roi = cv2.selectROI("Select Target", first_frame, False, False)
cv2.destroyWindow("Select Target")
x, y, w, h = map(int, roi)
if w <= 0 or h <= 0:
    cap.release()
    raise RuntimeError("Invalid target ROI selected.")

center_x = x + w / 2.0
center_y = y + h / 2.0
box = (x, y, w, h)
initial_w, initial_h = float(w), float(h)
target_points = detect_features(first_gray, box)
if target_points is None or len(target_points) < 4:
    cap.release()
    raise RuntimeError("Too few features in the selected ROI. Select a textured region of the target.")

# Kalman state [cx, cy, vx, vy].
kalman = cv2.KalmanFilter(4, 2)
kalman.transitionMatrix = np.array(
    [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
    dtype=np.float32,
)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2.0
kalman.errorCovPost = np.eye(4, dtype=np.float32)
kalman.statePost = np.array([[center_x], [center_y], [0], [0]], dtype=np.float32)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_width, frame_height))
if not out.isOpened():
    cap.release()
    raise RuntimeError(f"Could not create output video: {OUTPUT_VIDEO}")

csv_file = open(CSV_FILE, "w", newline="", encoding="utf-8")
writer = csv.writer(csv_file)
writer.writerow([
    "frame", "x", "y", "width", "height", "center_x", "center_y",
    "dx", "dy", "velocity", "target_points", "ransac_inliers",
    "fb_error", "confidence", "tracking_status"
])

prev_gray = first_gray.copy()
previous_center = np.array([center_x, center_y], dtype=np.float32)
frame_number = 1
accepted_updates = 0
rejected_updates = 0
redetections = 0
start_time = time.time()

print(f"Video: {frame_width}x{frame_height} @ {fps:.2f} FPS; {total_frames} frames")
print("Tracking started. Press Q in the video window to stop.")

# ============================================================
# MAIN TRACKING LOOP
# ============================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_number += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    prediction = kalman.predict()
    predicted_center = np.array([prediction[0, 0], prediction[1, 0]], dtype=np.float32)

    dx = dy = velocity = 0.0
    fb_error = 999.0
    confidence = 0.0
    inliers_count = 0
    status_text = "PREDICTION"
    measured_box = None

    # If too few points remain, search around the predicted location.
    if target_points is None or len(target_points) < MIN_TARGET_POINTS:
        search_w = max(w, int(initial_w * 0.75))
        search_h = max(h, int(initial_h * 0.75))
        search_box = clip_box(
            predicted_center[0], predicted_center[1],
            search_w * REDETECT_MARGIN, search_h * REDETECT_MARGIN,
            frame_width, frame_height,
        )
        recovered = detect_features(gray, search_box)
        if recovered is not None and len(recovered) >= 4:
            target_points = recovered
            redetections += 1

    # Forward-backward optical flow and robust affine estimation.
    if target_points is not None and len(target_points) >= 4:
        next_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, target_points, None, **LK_PARAMS
        )
        if next_points is not None:
            backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
                gray, prev_gray, next_points, None, **LK_PARAMS
            )
            if backward_points is not None:
                old_all = target_points.reshape(-1, 2)
                new_all = next_points.reshape(-1, 2)
                back_all = backward_points.reshape(-1, 2)
                fb_dist = np.linalg.norm(old_all - back_all, axis=1)
                valid = (
                    forward_status.reshape(-1).astype(bool)
                    & backward_status.reshape(-1).astype(bool)
                    & (fb_dist < FB_THRESHOLD)
                    & np.isfinite(new_all).all(axis=1)
                )
                old_pts = old_all[valid]
                new_pts = new_all[valid]
                if len(old_pts) >= 4:
                    fb_error = float(np.mean(fb_dist[valid]))
                    M, ransac_mask = cv2.estimateAffinePartial2D(
                        old_pts, new_pts,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=3.0,
                        maxIters=2000,
                        confidence=0.99,
                    )
                    if M is not None and ransac_mask is not None:
                        inlier_mask = ransac_mask.reshape(-1).astype(bool)
                        old_inliers = old_pts[inlier_mask]
                        new_inliers = new_pts[inlier_mask]
                        inliers_count = len(new_inliers)

                        if inliers_count >= MIN_RANSAC_INLIERS:
                            # Transform the whole box with the same robust model.
                            candidate_cx, candidate_cy, candidate_w, candidate_h = transform_box(M, box)

                            # Limit frame-to-frame scale changes to avoid sudden box explosions.
                            candidate_w = clamp_scale(candidate_w, w, MIN_SCALE_PER_FRAME, MAX_SCALE_PER_FRAME)
                            candidate_h = clamp_scale(candidate_h, h, MIN_SCALE_PER_FRAME, MAX_SCALE_PER_FRAME)

                            measured_center = np.array([candidate_cx, candidate_cy], dtype=np.float32)
                            movement = float(np.linalg.norm(measured_center - previous_center))
                            pred_distance = float(np.linalg.norm(measured_center - predicted_center))

                            point_score = min(len(new_inliers) / 30.0, 1.0)
                            inlier_score = min(inliers_count / 20.0, 1.0)
                            fb_score = max(0.0, min(1.0, 1.0 - fb_error / FB_THRESHOLD))
                            prediction_score = max(0.0, min(1.0, 1.0 - pred_distance / MAX_CENTER_JUMP))
                            confidence = (
                                0.35 * point_score
                                + 0.35 * inlier_score
                                + 0.20 * fb_score
                                + 0.10 * prediction_score
                            )

                            # Accept strong geometric consensus even if Kalman prediction disagrees.
                            strong_geometry = inliers_count >= 12 and fb_error < 0.8
                            accept = (
                                confidence >= HIGH_CONFIDENCE
                                and movement <= MAX_CENTER_JUMP
                            ) or strong_geometry

                            if accept:
                                center_x, center_y = float(candidate_cx), float(candidate_cy)
                                w = int(np.clip(candidate_w, MIN_BOX_SIZE, frame_width * MAX_BOX_WIDTH_FRACTION))
                                h = int(np.clip(candidate_h, MIN_BOX_SIZE, frame_height * MAX_BOX_HEIGHT_FRACTION))
                                accepted_updates += 1
                                status_text = "TRACKING"
                                dx = float(center_x - previous_center[0])
                                dy = float(center_y - previous_center[1])
                                velocity = float(np.hypot(dx, dy))
                                measurement = np.array([[center_x], [center_y]], dtype=np.float32)
                                kalman.correct(measurement)
                                previous_center = np.array([center_x, center_y], dtype=np.float32)
                                # Continue with only geometric inliers.
                                target_points = new_inliers.reshape(-1, 1, 2).astype(np.float32)
                            else:
                                rejected_updates += 1
                                status_text = "RECOVERY"
                                # Do not permanently freeze on the old prediction.
                                center_x, center_y = float(predicted_center[0]), float(predicted_center[1])
                                previous_center = predicted_center.copy()
                                target_points = None
                        else:
                            status_text = "FEW_INLIERS"
                            target_points = None
                    else:
                        status_text = "NO_RANSAC_MODEL"
                        target_points = None
                else:
                    status_text = "FEW_VALID_POINTS"
                    target_points = None
            else:
                status_text = "BACKWARD_FLOW_FAILED"
                target_points = None
        else:
            status_text = "FORWARD_FLOW_FAILED"
            target_points = None
    else:
        status_text = "FEW_POINTS"

    # Keep the box within image bounds and ensure valid dimensions.
    x, y, w, h = clip_box(center_x, center_y, w, h, frame_width, frame_height)
    center_x, center_y = x + w / 2.0, y + h / 2.0
    box = (x, y, w, h)

    # Re-detect within the updated target box when feature support is low.
    if target_points is None or len(target_points) < MIN_TARGET_POINTS:
        new_points = detect_features(gray, box)
        if new_points is not None and len(new_points) >= 4:
            target_points = new_points
            redetections += 1

    # Draw tracking box and points.
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if target_points is not None:
        for p in target_points.reshape(-1, 2):
            px, py = int(p[0]), int(p[1])
            if 0 <= px < frame_width and 0 <= py < frame_height:
                cv2.circle(frame, (px, py), 2, (0, 255, 255), -1)

    cv2.circle(frame, (int(center_x), int(center_y)), 4, (0, 0, 255), -1)
    cv2.putText(frame, f"Status: {status_text}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"Confidence: {confidence:.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"Features: {0 if target_points is None else len(target_points)}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"RANSAC inliers: {inliers_count}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"Velocity: {velocity:.2f} px/frame", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    writer.writerow([
        frame_number, x, y, w, h, center_x, center_y,
        dx, dy, velocity,
        0 if target_points is None else len(target_points),
        inliers_count, fb_error, confidence, status_text,
    ])

    out.write(frame)
    cv2.imshow("Stage 6A Corrected - Adaptive Optical Flow", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    prev_gray = gray.copy()

elapsed = time.time() - start_time
processed = frame_number - 1
cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()

print("\n======================================")
print("CORRECTED STAGE 6A RESULTS")
print("======================================")
print("Frames processed       :", processed)
print("Original video FPS     :", f"{fps:.2f}")
print("Average processing FPS :", f"{processed / elapsed:.2f}" if elapsed > 0 else "N/A")
print("Accepted updates       :", accepted_updates)
print("Rejected updates       :", rejected_updates)
print("Feature re-detections  :", redetections)
print("Output video saved to  :", OUTPUT_VIDEO)
print("Tracking data saved to:", CSV_FILE)
print("======================================")
