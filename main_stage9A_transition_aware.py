import cv2
import numpy as np
import csv
import os
import time

# ============================================================
# STAGE 9A: TRANSITION-AWARE TRACKING (EXPERIMENTAL)
# Baseline: user's corrected Stage 6A implementation.
# New logic: transition-risk detection + cautious acceptance +
# bounded local recovery + logging of transition indicators.
# ============================================================

INPUT_VIDEO = "data/input.mp4"
OUTPUT_VIDEO = "results/stage9A_transition_aware.mp4"
CSV_FILE = "results/stage9A_transition_aware_data.csv"
os.makedirs("results", exist_ok=True)

LK_PARAMS = dict(
    winSize=(21, 21), maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)
FEATURE_PARAMS = dict(maxCorners=200, qualityLevel=0.01, minDistance=7, blockSize=7)

MIN_TARGET_POINTS = 8
MIN_RANSAC_INLIERS = 6
FB_THRESHOLD = 1.5
MAX_CENTER_JUMP = 100.0
MIN_SCALE_PER_FRAME = 0.75
MAX_SCALE_PER_FRAME = 1.33
MIN_BOX_SIZE = 12
MAX_BOX_WIDTH_FRACTION = 0.60
MAX_BOX_HEIGHT_FRACTION = 0.80
HIGH_CONFIDENCE = 0.48

# Stage 9A transition detector parameters; tune only after reviewing results.
EDGE_MARGIN_FRACTION = 0.10
SMALL_BOX_FRACTION = 0.025       # target area / frame area
LOW_FEATURE_COUNT = 15
MOTION_DISAGREEMENT_PX = 35.0
RECOVERY_MARGIN_FACTOR = 1.35
MAX_RECOVERY_CENTER_SHIFT = 90.0

def clip_box(cx, cy, bw, bh, fw, fh):
    bw = int(np.clip(round(bw), MIN_BOX_SIZE, fw * MAX_BOX_WIDTH_FRACTION))
    bh = int(np.clip(round(bh), MIN_BOX_SIZE, fh * MAX_BOX_HEIGHT_FRACTION))
    x = int(np.clip(round(cx - bw / 2), 0, fw - bw))
    y = int(np.clip(round(cy - bh / 2), 0, fh - bh))
    return x, y, bw, bh

def detect_features(gray, box):
    x, y, bw, bh = box
    mask = np.zeros(gray.shape, dtype=np.uint8)
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(gray.shape[1], x + bw), min(gray.shape[0], y + bh)
    if x2 <= x1 or y2 <= y1:
        return None
    mask[y1:y2, x1:x2] = 255
    return cv2.goodFeaturesToTrack(gray, mask=mask, **FEATURE_PARAMS)

def transform_box(M, box):
    x, y, bw, bh = box
    corners = np.array([[[x, y]], [[x+bw, y]], [[x+bw, y+bh]], [[x, y+bh]]], np.float32)
    transformed = cv2.transform(corners, M).reshape(-1, 2)
    xmin, ymin = transformed.min(axis=0)
    xmax, ymax = transformed.max(axis=0)
    return float((xmin+xmax)/2), float((ymin+ymax)/2), float(xmax-xmin), float(ymax-ymin)

def clamp_scale(new_size, old_size):
    if old_size <= 0 or not np.isfinite(new_size):
        return old_size
    return old_size * float(np.clip(new_size / old_size, MIN_SCALE_PER_FRAME, MAX_SCALE_PER_FRAME))

def transition_signals(box, feature_count, confidence, pred_center, measured_center, fw, fh):
    x, y, bw, bh = box
    cx, cy = x + bw/2, y + bh/2
    near_edge = (
        cx < fw * EDGE_MARGIN_FRACTION or cx > fw * (1 - EDGE_MARGIN_FRACTION)
        or cy < fh * EDGE_MARGIN_FRACTION or cy > fh * (1 - EDGE_MARGIN_FRACTION)
    )
    small_target = (bw * bh) / float(fw * fh) < SMALL_BOX_FRACTION
    low_features = feature_count < LOW_FEATURE_COUNT
    disagreement = float(np.linalg.norm(measured_center - pred_center)) if measured_center is not None else 0.0
    motion_disagreement = measured_center is not None and disagreement > MOTION_DISAGREEMENT_PX
    # Risk is a transparent heuristic, not a learned probability.
    risk = int(near_edge) + int(small_target) + int(low_features) + int(motion_disagreement) + int(confidence < 0.35)
    return near_edge, small_target, low_features, motion_disagreement, disagreement, risk

def local_recovery(gray, predicted_center, old_box, fw, fh):
    """Detect features in a bounded search region around prediction; reject implausible shifts."""
    _, _, bw, bh = old_box
    sw = min(fw * 0.45, max(bw * RECOVERY_MARGIN_FACTOR * 2.0, bw + 50))
    sh = min(fh * 0.45, max(bh * RECOVERY_MARGIN_FACTOR * 2.0, bh + 50))
    search_box = clip_box(predicted_center[0], predicted_center[1], sw, sh, fw, fh)
    points = detect_features(gray, search_box)
    if points is None or len(points) < MIN_TARGET_POINTS:
        return None, search_box
    pts = points.reshape(-1, 2)
    # Keep only features near the predicted neighborhood; this is a spatial safeguard,
    # not proof that every feature belongs to the target.
    radius = max(45.0, 0.75 * max(bw, bh))
    keep = np.linalg.norm(pts - predicted_center.reshape(1, 2), axis=1) <= radius
    points = points[keep].reshape(-1, 1, 2).astype(np.float32)
    if len(points) < MIN_TARGET_POINTS:
        return None, search_box
    return points, search_box

cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise RuntimeError(f"Could not open input video: {INPUT_VIDEO}")
fps = cap.get(cv2.CAP_PROP_FPS)
fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
ret, first_frame = cap.read()
if not ret:
    cap.release()
    raise RuntimeError("Could not read first frame.")
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
print("Select the target vehicle in the first frame; press ENTER/SPACE.")
roi = cv2.selectROI("Select Target", first_frame, False, False)
cv2.destroyWindow("Select Target")
x, y, w, h = map(int, roi)
if w <= 0 or h <= 0:
    cap.release()
    raise RuntimeError("Invalid target ROI.")
box = (x, y, w, h)
center_x, center_y = x + w/2.0, y + h/2.0
target_points = detect_features(first_gray, box)
if target_points is None or len(target_points) < 4:
    cap.release()
    raise RuntimeError("Too few initial features; select a textured target region.")

kalman = cv2.KalmanFilter(4, 2)
kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2.0
kalman.errorCovPost = np.eye(4, dtype=np.float32)
kalman.statePost = np.array([[center_x],[center_y],[0],[0]], np.float32)

out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))
if not out.isOpened():
    cap.release()
    raise RuntimeError(f"Could not create output video: {OUTPUT_VIDEO}")
csv_file = open(CSV_FILE, "w", newline="", encoding="utf-8")
writer = csv.writer(csv_file)
writer.writerow([
    "frame","x","y","width","height","center_x","center_y","dx","dy","velocity",
    "target_points","ransac_inliers","fb_error","confidence","tracking_status",
    "near_edge","small_target","low_features","motion_disagreement","prediction_distance",
    "transition_risk","recovery_attempted"
])

prev_gray = first_gray.copy()
previous_center = np.array([center_x, center_y], np.float32)
frame_number = 1
accepted_updates = rejected_updates = redetections = transition_frames = 0
start_time = time.time()
print(f"Video: {fw}x{fh} @ {fps:.2f} FPS; {total_frames} frames")
print("Stage 9A running. Press Q to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_number += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prediction = kalman.predict()
    predicted_center = np.array([prediction[0,0], prediction[1,0]], np.float32)

    dx = dy = velocity = 0.0
    fb_error, confidence, inliers_count = 999.0, 0.0, 0
    status_text = "PREDICTION"
    measured_center = None
    recovery_attempted = 0
    feature_count_before = 0 if target_points is None else len(target_points)

    if target_points is not None and len(target_points) >= 4:
        next_points, st_f, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, target_points, None, **LK_PARAMS)
        if next_points is not None:
            back_points, st_b, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, next_points, None, **LK_PARAMS)
            if back_points is not None:
                old_all = target_points.reshape(-1,2)
                new_all = next_points.reshape(-1,2)
                back_all = back_points.reshape(-1,2)
                fb_dist = np.linalg.norm(old_all - back_all, axis=1)
                valid = st_f.reshape(-1).astype(bool) & st_b.reshape(-1).astype(bool) & (fb_dist < FB_THRESHOLD) & np.isfinite(new_all).all(axis=1)
                old_pts, new_pts = old_all[valid], new_all[valid]
                if len(old_pts) >= 4:
                    fb_error = float(np.mean(fb_dist[valid]))
                    M, rmask = cv2.estimateAffinePartial2D(old_pts, new_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=2000, confidence=0.99)
                    if M is not None and rmask is not None:
                        inlier_mask = rmask.reshape(-1).astype(bool)
                        new_inliers = new_pts[inlier_mask]
                        inliers_count = len(new_inliers)
                        if inliers_count >= MIN_RANSAC_INLIERS:
                            ccx, ccy, cw, ch = transform_box(M, box)
                            cw, ch = clamp_scale(cw, w), clamp_scale(ch, h)
                            measured_center = np.array([ccx, ccy], np.float32)
                            pred_dist = float(np.linalg.norm(measured_center - predicted_center))
                            point_score = min(len(new_inliers)/30.0, 1.0)
                            inlier_score = min(inliers_count/20.0, 1.0)
                            fb_score = float(np.clip(1.0 - fb_error/FB_THRESHOLD, 0.0, 1.0))
                            pred_score = float(np.clip(1.0 - pred_dist/MAX_CENTER_JUMP, 0.0, 1.0))
                            confidence = 0.35*point_score + 0.35*inlier_score + 0.20*fb_score + 0.10*pred_score
                            near_edge, small_target, low_features, motion_disagreement, disagreement, risk = transition_signals(
                                box, feature_count_before, confidence, predicted_center, measured_center, fw, fh
                            )
                            # Novel experimental policy: high-risk transitions require stronger
                            # consensus and better prediction agreement; no unconditional geometry override.
                            required_inliers = MIN_RANSAC_INLIERS + (2 if risk >= 2 else 0) + (3 if risk >= 3 else 0)
                            required_confidence = HIGH_CONFIDENCE + (0.08 if risk >= 2 else 0.0) + (0.08 if risk >= 3 else 0.0)
                            max_jump = MAX_CENTER_JUMP if risk < 2 else MAX_CENTER_JUMP * 0.65
                            accept = (
                                inliers_count >= required_inliers
                                and confidence >= required_confidence
                                and disagreement <= max_jump
                            )
                            if accept:
                                center_x, center_y = ccx, ccy
                                w = int(np.clip(cw, MIN_BOX_SIZE, fw*MAX_BOX_WIDTH_FRACTION))
                                h = int(np.clip(ch, MIN_BOX_SIZE, fh*MAX_BOX_HEIGHT_FRACTION))
                                accepted_updates += 1
                                status_text = "TRACKING"
                                dx, dy = center_x-previous_center[0], center_y-previous_center[1]
                                velocity = float(np.hypot(dx,dy))
                                kalman.correct(np.array([[center_x],[center_y]], np.float32))
                                previous_center = np.array([center_x,center_y], np.float32)
                                target_points = new_inliers.reshape(-1,1,2).astype(np.float32)
                            else:
                                rejected_updates += 1
                                status_text = "TRANSITION_GUARD"
                                center_x, center_y = float(predicted_center[0]), float(predicted_center[1])
                                previous_center = predicted_center.copy()
                                target_points = None
                        else:
                            status_text, target_points = "FEW_INLIERS", None
                    else:
                        status_text, target_points = "NO_RANSAC_MODEL", None
                else:
                    status_text, target_points = "FEW_VALID_POINTS", None
            else:
                status_text, target_points = "BACKWARD_FLOW_FAILED", None
        else:
            status_text, target_points = "FORWARD_FLOW_FAILED", None
    else:
        status_text = "FEW_POINTS"

    # Derive transition flags for logging/visualization using current box and current confidence.
    box = clip_box(center_x, center_y, w, h, fw, fh)
    x, y, w, h = box
    center_x, center_y = x+w/2.0, y+h/2.0
    current_count = 0 if target_points is None else len(target_points)
    near_edge, small_target, low_features, motion_disagreement, disagreement, risk = transition_signals(
        box, current_count, confidence, predicted_center, measured_center, fw, fh
    )
    if risk >= 2:
        transition_frames += 1

    if target_points is None or len(target_points) < MIN_TARGET_POINTS:
        recovery_attempted = 1
        recovered, search_box = local_recovery(gray, predicted_center, box, fw, fh)
        if recovered is not None:
            target_points = recovered
            redetections += 1
            # Keep prediction as the center during recovery; do not instantly accept
            # the newly detected features as verified target motion.
            status_text = "LOCAL_RECOVERY"
        else:
            target_points = None
            if status_text == "PREDICTION":
                status_text = "RECOVERY_FAILED"

    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
    if target_points is not None:
        for p in target_points.reshape(-1,2):
            px, py = int(p[0]), int(p[1])
            if 0 <= px < fw and 0 <= py < fh:
                cv2.circle(frame, (px,py), 2, (0,255,255), -1)
    cv2.circle(frame, (int(center_x),int(center_y)), 4, (0,0,255), -1)
    cv2.putText(frame, f"Status: {status_text}", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255,255,255), 2)
    cv2.putText(frame, f"Confidence: {confidence:.2f}  Risk: {risk}", (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255,255,255), 2)
    cv2.putText(frame, f"Features: {current_count}  Inliers: {inliers_count}", (20,90), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255,255,255), 2)
    cv2.putText(frame, f"Edge: {int(near_edge)} Small: {int(small_target)}", (20,120), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255,255,255), 2)

    writer.writerow([
        frame_number,x,y,w,h,center_x,center_y,dx,dy,velocity,current_count,inliers_count,
        fb_error,confidence,status_text,int(near_edge),int(small_target),int(low_features),
        int(motion_disagreement),disagreement,risk,recovery_attempted
    ])
    out.write(frame)
    cv2.imshow("Stage 9A - Transition-Aware Experimental Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    prev_gray = gray.copy()

elapsed = time.time()-start_time
processed = frame_number-1
cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()

print("\n======================================")
print("STAGE 9A EXPERIMENTAL RESULTS")
print("======================================")
print("Frames processed       :", processed)
print("Original video FPS     :", f"{fps:.2f}")
print("Average processing FPS :", f"{processed/elapsed:.2f}" if elapsed > 0 else "N/A")
print("Accepted updates       :", accepted_updates)
print("Rejected updates       :", rejected_updates)
print("Local re-detections    :", redetections)
print("Transition-risk frames :", transition_frames)
print("Output video saved to  :", OUTPUT_VIDEO)
print("Tracking data saved to:", CSV_FILE)
print("NOTE: These are tracking-operation metrics, not ground-truth accuracy.")
print("======================================")
