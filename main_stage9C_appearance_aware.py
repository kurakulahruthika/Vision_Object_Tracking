
import cv2
import numpy as np
import csv
import os
import time

# ============================================================
# STAGE 9C:
# APPEARANCE-AWARE TRACKING + OPTICAL FLOW
#
# Features:
# 1. Lucas-Kanade optical flow
# 2. Forward-backward validation
# 3. RANSAC motion estimation
# 4. Multi-scale template matching
# 5. Trusted bounding-box safeguards
# 6. Confidence-aware template updates
# 7. Local and global appearance recovery
#
# Input:
#   data/input.mp4
#
# Outputs:
#   results/stage9C_appearance_aware.mp4
#   results/stage9C_appearance_aware_data.csv
# ============================================================

INPUT_VIDEO = "data/input.mp4"
OUTPUT_VIDEO = "results/stage9C_appearance_aware.mp4"
CSV_FILE = "results/stage9C_appearance_aware_data.csv"

os.makedirs("results", exist_ok=True)

# ---------------- Configuration ----------------

LK_PARAMS = {
    "winSize": (21, 21),
    "maxLevel": 3,
    "criteria": (
        cv2.TERM_CRITERIA_EPS
        | cv2.TERM_CRITERIA_COUNT,
        30,
        0.01
    )
}

FEATURE_PARAMS = {
    "maxCorners": 200,
    "qualityLevel": 0.01,
    "minDistance": 7,
    "blockSize": 7
}

SCALE_FACTORS = [
    0.60, 0.70, 0.80, 0.90,
    1.00, 1.10, 1.20, 1.30, 1.45
]

FB_THRESHOLD = 1.5
MIN_FEATURES = 6
MIN_INLIERS = 5

LOCAL_MATCH_THRESHOLD = 0.48
GLOBAL_MATCH_THRESHOLD = 0.52
TEMPLATE_UPDATE_THRESHOLD = 0.72

LOCAL_SEARCH_FACTOR = 2.5
LOCAL_MIN_MARGIN = 100

MAX_CENTER_JUMP = 150.0
MIN_SIZE_RATIO = 0.72
MAX_SIZE_RATIO = 1.20

MAX_BOX_AREA_FRACTION = 0.12


# ---------------- Bounding-box utilities ----------------

def clip_box(cx, cy, width, height, frame_width, frame_height):
    width = int(np.clip(
        round(width), 12, frame_width * 0.60
    ))

    height = int(np.clip(
        round(height), 12, frame_height * 0.80
    ))

    x = int(np.clip(
        round(cx - width / 2),
        0,
        frame_width - width
    ))

    y = int(np.clip(
        round(cy - height / 2),
        0,
        frame_height - height
    ))

    return x, y, width, height


def center_of(box):
    x, y, width, height = box

    return np.array([
        x + width / 2.0,
        y + height / 2.0
    ], dtype=np.float32)


def detect_features(gray, box):
    x, y, width, height = box

    mask = np.zeros(gray.shape, dtype=np.uint8)

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(gray.shape[1], x + width)
    y2 = min(gray.shape[0], y + height)

    if x2 <= x1 or y2 <= y1:
        return None

    mask[y1:y2, x1:x2] = 255

    return cv2.goodFeaturesToTrack(
        gray,
        mask=mask,
        **FEATURE_PARAMS
    )


def safe_crop(gray, box):
    x, y, width, height = box

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(gray.shape[1], x + width)
    y2 = min(gray.shape[0], y + height)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = gray[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return crop.copy()


# ---------------- Appearance matching ----------------

def appearance_search(
    gray,
    template,
    trusted_box,
    predicted_center,
    global_search=False
):
    """
    Search for a similar appearance at multiple scales.

    Returns:
        best_box, best_score

    The score is a normalized template-matching score.
    It is NOT a probability that the candidate is the target.
    """

    frame_height, frame_width = gray.shape

    template_height, template_width = template.shape[:2]

    _, _, box_width, box_height = trusted_box

    if global_search:
        sx1, sy1 = 0, 0
        sx2, sy2 = frame_width, frame_height

    else:
        margin_x = max(
            LOCAL_MIN_MARGIN,
            int(box_width * LOCAL_SEARCH_FACTOR)
        )

        margin_y = max(
            LOCAL_MIN_MARGIN,
            int(box_height * LOCAL_SEARCH_FACTOR)
        )

        sx1 = max(
            0, int(predicted_center[0] - margin_x)
        )

        sy1 = max(
            0, int(predicted_center[1] - margin_y)
        )

        sx2 = min(
            frame_width,
            int(predicted_center[0] + margin_x)
        )

        sy2 = min(
            frame_height,
            int(predicted_center[1] + margin_y)
        )

    search = gray[sy1:sy2, sx1:sx2]

    if search.size == 0:
        return None, -1.0

    search_edges = cv2.Canny(search, 50, 150)

    best_box = None
    best_score = -1.0

    for scale in SCALE_FACTORS:

        width = max(
            12, int(template_width * scale)
        )

        height = max(
            12, int(template_height * scale)
        )

        if (
            width > search.shape[1]
            or height > search.shape[0]
        ):
            continue

        resized = cv2.resize(
            template,
            (width, height),
            interpolation=(
                cv2.INTER_AREA
                if scale < 1.0
                else cv2.INTER_LINEAR
            )
        )

        template_edges = cv2.Canny(
            resized, 50, 150
        )

        if cv2.countNonZero(template_edges) == 0:
            continue

        result = cv2.matchTemplate(
            search_edges,
            template_edges,
            cv2.TM_CCOEFF_NORMED
        )

        _, score, _, location = cv2.minMaxLoc(result)

        if np.isfinite(score) and score > best_score:

            x = sx1 + location[0]
            y = sy1 + location[1]

            best_box = (x, y, width, height)
            best_score = float(score)

    return best_box, best_score


# ---------------- Optical-flow estimation ----------------

def estimate_flow(
    previous_gray,
    current_gray,
    points,
    trusted_box
):
    """
    Estimate target-region motion using LK optical flow
    and a RANSAC partial-affine model.
    """

    if points is None or len(points) < 4:
        return None, None, 0, 999.0, 0.0

    next_points, status_forward, _ = (
        cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            points,
            None,
            **LK_PARAMS
        )
    )

    if next_points is None:
        return None, None, 0, 999.0, 0.0

    back_points, status_backward, _ = (
        cv2.calcOpticalFlowPyrLK(
            current_gray,
            previous_gray,
            next_points,
            None,
            **LK_PARAMS
        )
    )

    if back_points is None:
        return None, None, 0, 999.0, 0.0

    old = points.reshape(-1, 2)
    new = next_points.reshape(-1, 2)
    back = back_points.reshape(-1, 2)

    fb_errors = np.linalg.norm(old - back, axis=1)

    valid = (
        status_forward.reshape(-1).astype(bool)
        & status_backward.reshape(-1).astype(bool)
        & (fb_errors < FB_THRESHOLD)
        & np.isfinite(new).all(axis=1)
    )

    old_valid = old[valid]
    new_valid = new[valid]

    if len(old_valid) < 4:
        return None, None, 0, 999.0, 0.0

    mean_fb_error = float(np.mean(fb_errors[valid]))

    matrix, ransac_mask = cv2.estimateAffinePartial2D(
        old_valid,
        new_valid,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=1500,
        confidence=0.99
    )

    if matrix is None or ransac_mask is None:
        return None, None, 0, mean_fb_error, 0.0

    inlier_mask = ransac_mask.reshape(-1).astype(bool)

    inlier_points = new_valid[inlier_mask]
    inlier_count = len(inlier_points)

    if inlier_count < MIN_INLIERS:
        return None, None, inlier_count, mean_fb_error, 0.0

    x, y, width, height = trusted_box

    corners = np.array([
        [[x, y]],
        [[x + width, y]],
        [[x + width, y + height]],
        [[x, y + height]]
    ], dtype=np.float32)

    transformed = cv2.transform(corners, matrix).reshape(-1, 2)

    low = transformed.min(axis=0)
    high = transformed.max(axis=0)

    center = np.array([
        (low[0] + high[0]) / 2.0,
        (low[1] + high[1]) / 2.0
    ], dtype=np.float32)

    score = (
        min(inlier_count / 25.0, 1.0)
        * np.clip(1.0 - mean_fb_error / FB_THRESHOLD, 0, 1)
    )

    return (
        center,
        inlier_points.reshape(-1, 1, 2).astype(np.float32),
        inlier_count,
        mean_fb_error,
        float(score)
    )


# ---------------- Video setup ----------------

cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open input video: {INPUT_VIDEO}"
    )

fps = cap.get(cv2.CAP_PROP_FPS)

frame_width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

frame_height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

ret, first_frame = cap.read()

if not ret:
    cap.release()
    raise RuntimeError("Could not read first frame.")

first_gray = cv2.cvtColor(
    first_frame, cv2.COLOR_BGR2GRAY
)

print("Select the vehicle in the first frame.")
print("Use a reasonably tight bounding box.")

roi = cv2.selectROI(
    "Select Vehicle",
    first_frame,
    False,
    False
)

cv2.destroyWindow("Select Vehicle")

x, y, width, height = map(int, roi)

if width <= 0 or height <= 0:
    cap.release()
    raise RuntimeError("Invalid ROI selected.")

trusted_box = (x, y, width, height)

template = safe_crop(first_gray, trusted_box)

if template is None:
    cap.release()
    raise RuntimeError("Could not create initial template.")

# Preserve a fixed template size for matching.
template_width = max(24, min(180, width))
template_height = max(24, min(180, height))

template = cv2.resize(
    template,
    (template_width, template_height)
)

template = cv2.equalizeHist(template)

center = center_of(trusted_box)

# Kalman state: x, y, vx, vy
kalman = cv2.KalmanFilter(4, 2)

kalman.transitionMatrix = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], dtype=np.float32)

kalman.measurementMatrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
], dtype=np.float32)

kalman.processNoiseCov = (
    np.eye(4, dtype=np.float32) * 0.05
)

kalman.measurementNoiseCov = (
    np.eye(2, dtype=np.float32) * 3.0
)

kalman.errorCovPost = np.eye(
    4, dtype=np.float32
)

kalman.statePost = np.array([
    [center[0]],
    [center[1]],
    [0],
    [0]
], dtype=np.float32)

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (frame_width, frame_height)
)

if not out.isOpened():
    cap.release()
    raise RuntimeError("Could not create output video.")

csv_file = open(
    CSV_FILE, "w", newline="", encoding="utf-8"
)

writer = csv.writer(csv_file)

writer.writerow([
    "frame", "x", "y", "width", "height",
    "center_x", "center_y", "dx", "dy", "velocity",
    "appearance_score", "global_search",
    "target_points", "ransac_inliers",
    "flow_score", "tracking_status",
    "template_updated", "box_area_fraction"
])

previous_gray = first_gray.copy()

points = detect_features(
    first_gray, trusted_box
)

frame_number = 1

accepted_updates = 0
rejected_frames = 0
global_searches = 0
template_updates = 0

start_time = time.time()


# ---------------- Main tracking loop ----------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    gray = cv2.cvtColor(
        frame, cv2.COLOR_BGR2GRAY
    )

    prediction = kalman.predict()

    predicted_center = np.array([
        prediction[0, 0],
        prediction[1, 0]
    ], dtype=np.float32)

    previous_center = center_of(trusted_box)

    dx = 0.0
    dy = 0.0
    velocity = 0.0

    inlier_count = 0
    flow_score = 0.0

    template_updated = 0
    global_search = 0

    # 1. Estimate optical-flow motion.
    (
        flow_center,
        new_points,
        inlier_count,
        fb_error,
        flow_score
    ) = estimate_flow(
        previous_gray,
        gray,
        points,
        trusted_box
    )

    # 2. Search near the predicted position.
    match_box, appearance_score = appearance_search(
        gray,
        template,
        trusted_box,
        predicted_center,
        global_search=False
    )

    # 3. If local matching is weak, search the whole frame.
    if (
        match_box is None
        or appearance_score < LOCAL_MATCH_THRESHOLD
    ):

        global_box, global_score = appearance_search(
            gray,
            template,
            trusted_box,
            predicted_center,
            global_search=True
        )

        global_search = 1
        global_searches += 1

        if (
            global_box is not None
            and global_score > appearance_score
        ):
            match_box = global_box
            appearance_score = global_score

    status = "NO_APPEARANCE_MATCH"
    accepted = False

    # 4. Validate appearance candidate.
    if match_box is not None:

        mx, my, mw, mh = match_box

        candidate_center = np.array([
            mx + mw / 2.0,
            my + mh / 2.0
        ], dtype=np.float32)

        distance_from_prediction = float(
            np.linalg.norm(
                candidate_center - predicted_center
            )
        )

        old_width = trusted_box[2]
        old_height = trusted_box[3]

        width_ratio = mw / max(1, old_width)
        height_ratio = mh / max(1, old_height)

        area_fraction = (
            mw * mh
        ) / float(frame_width * frame_height)

        size_ok = (
            MIN_SIZE_RATIO <= width_ratio <= MAX_SIZE_RATIO
            and MIN_SIZE_RATIO <= height_ratio <= MAX_SIZE_RATIO
            and area_fraction <= MAX_BOX_AREA_FRACTION
        )

        location_ok = (
            distance_from_prediction <= MAX_CENTER_JUMP
        )

        threshold = (
            GLOBAL_MATCH_THRESHOLD
            if global_search
            else LOCAL_MATCH_THRESHOLD
        )

        appearance_ok = appearance_score >= threshold

        flow_agrees = (
            flow_center is not None
            and np.linalg.norm(
                flow_center - candidate_center
            ) <= max(
                80.0,
                0.75 * max(old_width, old_height)
            )
        )

        # Appearance is required. Flow agreement is supporting evidence.
        if appearance_ok and size_ok and location_ok:

            accepted = True

            trusted_box = (
                int(mx), int(my), int(mw), int(mh)
            )

            center = center_of(trusted_box)

            dx = float(center[0] - previous_center[0])
            dy = float(center[1] - previous_center[1])

            velocity = float(np.hypot(dx, dy))

            kalman.correct(
                np.array([
                    [center[0]],
                    [center[1]]
                ], dtype=np.float32)
            )

            accepted_updates += 1

            if flow_agrees:
                status = "APPEARANCE_FLOW_AGREE"
            else:
                status = "APPEARANCE_ONLY"

            # Re-detect features only inside the accepted box.
            points = detect_features(
                gray, trusted_box
            )

            # Update template only when matching is strong.
            # This reduces, but does not eliminate, drift.
            if appearance_score >= TEMPLATE_UPDATE_THRESHOLD:

                crop = safe_crop(gray, trusted_box)

                if crop is not None and crop.size > 0:

                    template = cv2.resize(
                        crop,
                        (
                            template.shape[1],
                            template.shape[0]
                        )
                    )

                    template = cv2.equalizeHist(template)

                    template_updates += 1
                    template_updated = 1

        else:

            status = "MATCH_REJECTED"
            rejected_frames += 1
            points = None

    else:

        rejected_frames += 1
        points = None

    # 5. Render the last accepted trusted box.
    x, y, width, height = trusted_box

    area_fraction = (
        width * height
    ) / float(frame_width * frame_height)

    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + height),
        (0, 255, 0),
        2
    )

    cv2.circle(
        frame,
        (int(center[0]), int(center[1])),
        4,
        (0, 0, 255),
        -1
    )

    cv2.putText(
        frame,
        f"Status: {status}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Appearance: {appearance_score:.3f}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Flow inliers: {inlier_count}  Flow score: {flow_score:.2f}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Global search: {global_search}  Area: {area_fraction:.3f}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2
    )

    writer.writerow([
        frame_number,
        x, y, width, height,
        center[0], center[1],
        dx, dy, velocity,
        appearance_score,
        global_search,
        0 if points is None else len(points),
        inlier_count,
        flow_score,
        status,
        template_updated,
        area_fraction
    ])

    out.write(frame)

    cv2.imshow(
        "Stage 9C - Appearance-Aware Tracker",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    previous_gray = gray.copy()


# ---------------- Cleanup and summary ----------------

elapsed = time.time() - start_time
processed_frames = frame_number - 1

cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()

print("\n======================================")
print("STAGE 9C RESULTS")
print("======================================")

print("Frames processed        :", processed_frames)
print("Original video FPS      :", f"{fps:.2f}")

if elapsed > 0:
    print(
        "Average processing FPS  :",
        f"{processed_frames / elapsed:.2f}"
    )

print("Accepted updates        :", accepted_updates)
print("Rejected/lost frames    :", rejected_frames)
print("Global appearance searches:", global_searches)
print("Template updates        :", template_updates)

print("Output video saved to   :", OUTPUT_VIDEO)
print("Tracking data saved to  :", CSV_FILE)

print(
    "Reminder: template matching scores are not "
    "probabilities of correct target identity."
)

print("======================================")

