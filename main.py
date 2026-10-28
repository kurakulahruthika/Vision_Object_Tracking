import cv2
import numpy as np
import csv
import os
import time


# ============================================================
# SETTINGS
# ============================================================

INPUT_VIDEO = "data/input.mp4"

OUTPUT_VIDEO = "results/stage6A_hybrid_tracking.mp4"

CSV_FILE = "results/stage6A_tracking_data.csv"

os.makedirs("results", exist_ok=True)


# ============================================================
# OPTICAL FLOW PARAMETERS
# ============================================================

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(
        cv2.TERM_CRITERIA_EPS |
        cv2.TERM_CRITERIA_COUNT,
        30,
        0.01
    )
)


# ============================================================
# TARGET FEATURE PARAMETERS
# ============================================================

FEATURE_PARAMS_TARGET = dict(
    maxCorners=200,
    qualityLevel=0.01,
    minDistance=7,
    blockSize=7
)


# ============================================================
# TRACKING PARAMETERS
# ============================================================

MIN_TARGET_POINTS = 12

MIN_RANSAC_INLIERS = 8

FB_THRESHOLD = 1.5

HIGH_CONFIDENCE = 0.65

LOW_CONFIDENCE = 0.35

MAX_CENTER_JUMP = 80


# ============================================================
# KALMAN FILTER
#
# State:
# [x, y, vx, vy]
# ============================================================

kalman = cv2.KalmanFilter(4, 2)


kalman.transitionMatrix = np.array(
    [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ],
    dtype=np.float32
)


kalman.measurementMatrix = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ],
    dtype=np.float32
)


kalman.processNoiseCov = (
    np.eye(4, dtype=np.float32) * 0.03
)


kalman.measurementNoiseCov = (
    np.eye(2, dtype=np.float32) * 0.5
)


kalman.errorCovPost = (
    np.eye(4, dtype=np.float32)
)


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(INPUT_VIDEO)


if not cap.isOpened():

    print("ERROR: Could not open video.")

    exit()


fps = cap.get(cv2.CAP_PROP_FPS)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)


print("Video opened successfully!")

print("Video FPS:", fps)

print(
    "Resolution:",
    width,
    "x",
    height
)

print(
    "Total Frames:",
    total_frames
)


# ============================================================
# READ FIRST FRAME
# ============================================================

ret, first_frame = cap.read()


if not ret:

    print("ERROR: Could not read first frame.")

    cap.release()

    exit()


first_gray = cv2.cvtColor(
    first_frame,
    cv2.COLOR_BGR2GRAY
)


# ============================================================
# SELECT OBJECT
# ============================================================

print("\nSelect the vehicle you want to track.")


roi = cv2.selectROI(
    "Select Vehicle",
    first_frame,
    False,
    False
)


cv2.destroyWindow(
    "Select Vehicle"
)


x, y, w, h = roi


if w == 0 or h == 0:

    print("ERROR: Invalid ROI.")

    cap.release()

    exit()


print("\nSelected object:")

print("x =", x)

print("y =", y)

print("width =", w)

print("height =", h)


# ============================================================
# INITIAL TARGET FEATURES
# ============================================================

target_mask = np.zeros_like(
    first_gray
)


target_mask[
    y:y + h,
    x:x + w
] = 255


target_points = cv2.goodFeaturesToTrack(
    first_gray,
    mask=target_mask,
    **FEATURE_PARAMS_TARGET
)


if target_points is None:

    print(
        "ERROR: No target features detected."
    )

    cap.release()

    exit()


print(
    "Initial target feature points:",
    len(target_points)
)


# ============================================================
# INITIAL TARGET CENTER
# ============================================================

center_x = x + w / 2

center_y = y + h / 2


# ============================================================
# INITIALIZE KALMAN FILTER
# ============================================================

kalman.statePost = np.array(
    [
        [center_x],
        [center_y],
        [0],
        [0]
    ],
    dtype=np.float32
)


# ============================================================
# PREVIOUS FRAME
# ============================================================

prev_gray = first_gray.copy()


# ============================================================
# VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)


out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# CSV FILE
# ============================================================

csv_file = open(
    CSV_FILE,
    "w",
    newline=""
)


csv_writer = csv.writer(
    csv_file
)


csv_writer.writerow(
    [
        "frame",
        "x",
        "y",
        "width",
        "height",
        "center_x",
        "center_y",
        "dx",
        "dy",
        "velocity",
        "target_points",
        "ransac_inliers",
        "fb_error",
        "confidence",
        "tracking_status"
    ]
)


# ============================================================
# TRACKING VARIABLES
# ============================================================

frame_number = 1


previous_center = np.array(
    [center_x, center_y],
    dtype=np.float32
)


trajectory = []


accepted_updates = 0

rejected_updates = 0

recovery_frames = 0

feature_redetections = 0


min_target_points_seen = 999999

max_target_points_seen = 0


min_ransac_inliers_seen = 999999

max_ransac_inliers_seen = 0


start_time = time.time()


# ============================================================
# START MESSAGE
# ============================================================

print("\n======================================")

print("STAGE 6A")

print("HYBRID ADAPTIVE TRACKING")

print("======================================")

print("Optical Flow")

print("Forward-Backward Validation")

print("RANSAC Motion Estimation")

print("Kalman Prediction")

print("Confidence-Based Update")

print("Anti-Drift Protection")

print("======================================")

print("Press Q to stop.\n")


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()


    if not ret:

        break


    frame_number += 1


    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # ========================================================
    # KALMAN PREDICTION
    # ========================================================

    prediction = kalman.predict()


    predicted_center = np.array(
        [
            prediction[0, 0],
            prediction[1, 0]
        ],
        dtype=np.float32
    )


    # ========================================================
    # DEFAULT VALUES
    # ========================================================

    confidence = 0.0

    dx = 0.0

    dy = 0.0

    velocity = 0.0

    ransac_inliers = 0

    fb_error = 999.0

    tracking_status = "PREDICTION"


    # ========================================================
    # FEATURE RE-DETECTION
    # ========================================================

    if (
        target_points is None
        or
        len(target_points) < MIN_TARGET_POINTS
    ):

        recovery_x = int(
            predicted_center[0] - w / 2
        )

        recovery_y = int(
            predicted_center[1] - h / 2
        )


        recovery_x = max(
            0,
            min(
                recovery_x,
                width - w
            )
        )


        recovery_y = max(
            0,
            min(
                recovery_y,
                height - h
            )
        )


        recovery_mask = np.zeros_like(
            gray
        )


        recovery_mask[
            recovery_y:recovery_y + h,
            recovery_x:recovery_x + w
        ] = 255


        new_points = cv2.goodFeaturesToTrack(
            gray,
            mask=recovery_mask,
            **FEATURE_PARAMS_TARGET
        )


        if new_points is not None:

            target_points = new_points

            feature_redetections += 1

            recovery_frames += 1


    # ========================================================
    # OPTICAL FLOW
    # ========================================================

    if (
        target_points is not None
        and
        len(target_points) >= 4
    ):

        # ----------------------------------------------------
        # FORWARD OPTICAL FLOW
        # ----------------------------------------------------

        next_points, status, error = (
            cv2.calcOpticalFlowPyrLK(
                prev_gray,
                gray,
                target_points,
                None,
                **LK_PARAMS
            )
        )


        if next_points is not None:

            # ------------------------------------------------
            # BACKWARD OPTICAL FLOW
            # ------------------------------------------------

            backward_points, back_status, back_error = (
                cv2.calcOpticalFlowPyrLK(
                    gray,
                    prev_gray,
                    next_points,
                    None,
                    **LK_PARAMS
                )
            )


            if backward_points is not None:

                original_points = (
                    target_points.reshape(-1, 2)
                )

                forward_points = (
                    next_points.reshape(-1, 2)
                )

                backward_points_flat = (
                    backward_points.reshape(-1, 2)
                )


                # --------------------------------------------
                # FORWARD-BACKWARD ERROR
                # --------------------------------------------

                fb_dist = np.linalg.norm(
                    original_points -
                    backward_points_flat,
                    axis=1
                )


                valid_forward = (
                    status.reshape(-1).astype(bool)
                )


                valid_backward = (
                    back_status.reshape(-1).astype(bool)
                )


                valid_fb = (
                    fb_dist < FB_THRESHOLD
                )


                # IMPORTANT:
                # All three masks correspond to
                # the SAME original point array.
                # This prevents the indexing bug.
                # --------------------------------------------

                valid_mask = (
                    valid_forward
                    &
                    valid_backward
                    &
                    valid_fb
                )


                if np.any(valid_mask):

                    fb_error = float(
                        np.mean(
                            fb_dist[valid_mask]
                        )
                    )


                # ------------------------------------------------
                # CREATE CONSISTENT POINT ARRAYS
                # ------------------------------------------------

                old_pts = (
                    original_points[valid_mask]
                )


                new_pts = (
                    forward_points[valid_mask]
                )


                # ------------------------------------------------
                # RANSAC
                # ------------------------------------------------

                if len(old_pts) >= 4:

                    M, ransac_mask = (
                        cv2.estimateAffinePartial2D(
                            old_pts,
                            new_pts,
                            method=cv2.RANSAC,
                            ransacReprojThreshold=3.0,
                            maxIters=2000,
                            confidence=0.99
                        )
                    )


                    if (
                        M is not None
                        and
                        ransac_mask is not None
                    ):

                        # ----------------------------------------
                        # IMPORTANT:
                        # ransac_mask corresponds ONLY to
                        # old_pts/new_pts.
                        # ----------------------------------------

                        ransac_mask = (
                            ransac_mask.reshape(-1).astype(bool)
                        )


                        # Safety check
                        if (
                            len(ransac_mask)
                            ==
                            len(new_pts)
                        ):

                            reliable_old = (
                                old_pts[ransac_mask]
                            )


                            reliable_new = (
                                new_pts[ransac_mask]
                            )


                            ransac_inliers = len(
                                reliable_new
                            )


                            min_ransac_inliers_seen = min(
                                min_ransac_inliers_seen,
                                ransac_inliers
                            )


                            max_ransac_inliers_seen = max(
                                max_ransac_inliers_seen,
                                ransac_inliers
                            )


                            # ------------------------------------
                            # MOTION ESTIMATION
                            # ------------------------------------

                            if ransac_inliers >= 1:

                                old_center = np.mean(
                                    reliable_old,
                                    axis=0
                                )


                                new_center = np.mean(
                                    reliable_new,
                                    axis=0
                                )


                                motion = (
                                    new_center -
                                    old_center
                                )


                                dx = float(
                                    motion[0]
                                )


                                dy = float(
                                    motion[1]
                                )


                                measured_center = (
                                    previous_center +
                                    motion
                                )


                                # --------------------------------
                                # VELOCITY
                                # --------------------------------

                                velocity = float(
                                    np.linalg.norm(
                                        motion
                                    )
                                )


                                # --------------------------------
                                # CONFIDENCE
                                # --------------------------------

                                point_score = min(
                                    len(reliable_new) / 50.0,
                                    1.0
                                )


                                inlier_score = min(
                                    ransac_inliers / 30.0,
                                    1.0
                                )


                                fb_score = max(
                                    0.0,
                                    min(
                                        1.0,
                                        1.0 -
                                        fb_error / 3.0
                                    )
                                )


                                prediction_distance = (
                                    np.linalg.norm(
                                        measured_center -
                                        predicted_center
                                    )
                                )


                                prediction_score = max(
                                    0.0,
                                    min(
                                        1.0,
                                        1.0 -
                                        prediction_distance /
                                        MAX_CENTER_JUMP
                                    )
                                )


                                confidence = (
                                    0.30 *
                                    point_score
                                    +
                                    0.30 *
                                    inlier_score
                                    +
                                    0.20 *
                                    fb_score
                                    +
                                    0.20 *
                                    prediction_score
                                )


                                # --------------------------------
                                # ANTI-DRIFT
                                # --------------------------------

                                movement_from_previous = (
                                    np.linalg.norm(
                                        measured_center -
                                        previous_center
                                    )
                                )


                                if (
                                    confidence >= HIGH_CONFIDENCE
                                    and
                                    movement_from_previous
                                    <=
                                    MAX_CENTER_JUMP
                                ):

                                    # ACCEPT UPDATE
                                    center_x = float(
                                        measured_center[0]
                                    )


                                    center_y = float(
                                        measured_center[1]
                                    )


                                    accepted_updates += 1


                                    tracking_status = (
                                        "TRACKING"
                                    )


                                    measurement = np.array(
                                        [
                                            [center_x],
                                            [center_y]
                                        ],
                                        dtype=np.float32
                                    )


                                    kalman.correct(
                                        measurement
                                    )


                                    previous_center = (
                                        measured_center.copy()
                                    )


                                    # --------------------------------
                                    # KEEP ONLY RANSAC INLIERS
                                    # --------------------------------

                                    target_points = (
                                        reliable_new
                                        .reshape(-1, 1, 2)
                                        .astype(np.float32)
                                    )


                                else:

                                    # --------------------------------
                                    # REJECT UPDATE
                                    # --------------------------------

                                    rejected_updates += 1

                                    recovery_frames += 1

                                    tracking_status = (
                                        "REJECTED"
                                    )


                                    center_x = float(
                                        predicted_center[0]
                                    )


                                    center_y = float(
                                        predicted_center[1]
                                    )


                                    previous_center = (
                                        predicted_center.copy()
                                    )


                                    target_points = None


                        else:

                            tracking_status = (
                                "MASK_ERROR"
                            )

                            target_points = None


                    else:

                        tracking_status = (
                            "NO_RANSAC_MODEL"
                        )

                        target_points = None


                else:

                    tracking_status = (
                        "FEW_POINTS"
                    )

                    target_points = None


    # ========================================================
    # LOW CONFIDENCE
    # ========================================================

    if confidence < LOW_CONFIDENCE:

        center_x = float(
            predicted_center[0]
        )

        center_y = float(
            predicted_center[1]
        )


        if tracking_status == "PREDICTION":

            tracking_status = "RECOVERY"


        recovery_frames += 1


    # ========================================================
    # UPDATE FEATURE STATISTICS
    # ========================================================

    current_feature_count = (
        0
        if target_points is None
        else len(target_points)
    )


    min_target_points_seen = min(
        min_target_points_seen,
        current_feature_count
    )


    max_target_points_seen = max(
        max_target_points_seen,
        current_feature_count
    )


    # ========================================================
    # BOUNDING BOX
    # ========================================================

    new_x = int(
        center_x - w / 2
    )


    new_y = int(
        center_y - h / 2
    )


    new_x = max(
        0,
        min(
            new_x,
            width - w
        )
    )


    new_y = max(
        0,
        min(
            new_y,
            height - h
        )
    )


    # ========================================================
    # DRAW ROI
    # ========================================================

    cv2.rectangle(
        frame,
        (new_x, new_y),
        (new_x + w, new_y + h),
        (0, 255, 0),
        2
    )


    # ========================================================
    # DRAW TARGET FEATURES
    # ========================================================

    if target_points is not None:

        for point in target_points.reshape(-1, 2):

            px = int(point[0])

            py = int(point[1])


            if (
                0 <= px < width
                and
                0 <= py < height
            ):

                cv2.circle(
                    frame,
                    (px, py),
                    2,
                    (0, 255, 255),
                    -1
                )


    # ========================================================
    # DRAW CENTER
    # ========================================================

    cv2.circle(
        frame,
        (
            int(center_x),
            int(center_y)
        ),
        4,
        (0, 0, 255),
        -1
    )


    # ========================================================
    # TRAJECTORY
    # ========================================================

    trajectory.append(
        (
            int(center_x),
            int(center_y)
        )
    )


    for i in range(
        1,
        len(trajectory)
    ):

        cv2.line(
            frame,
            trajectory[i - 1],
            trajectory[i],
            (255, 0, 0),
            2
        )


    # ========================================================
    # INFORMATION
    # ========================================================

    cv2.putText(
        frame,
        f"Status: {tracking_status}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Features: {current_feature_count}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"RANSAC inliers: {ransac_inliers}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Velocity: {velocity:.2f} px/frame",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ========================================================
    # CSV
    # ========================================================

    csv_writer.writerow(
        [
            frame_number,
            new_x,
            new_y,
            w,
            h,
            center_x,
            center_y,
            dx,
            dy,
            velocity,
            current_feature_count,
            ransac_inliers,
            fb_error,
            confidence,
            tracking_status
        ]
    )


    # ========================================================
    # SAVE FRAME
    # ========================================================

    out.write(frame)


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "Stage 6A - Hybrid Adaptive Tracking",
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


    # ========================================================
    # RE-DETECT FEATURES IF NECESSARY
    # ========================================================

    if (
        target_points is None
        or
        len(target_points) < MIN_TARGET_POINTS
    ):

        detection_mask = np.zeros_like(
            gray
        )


        rx = max(
            0,
            new_x
        )


        ry = max(
            0,
            new_y
        )


        rw = min(
            w,
            width - rx
        )


        rh = min(
            h,
            height - ry
        )


        if (
            rw > 0
            and
            rh > 0
        ):

            detection_mask[
                ry:ry + rh,
                rx:rx + rw
            ] = 255


            new_points = (
                cv2.goodFeaturesToTrack(
                    gray,
                    mask=detection_mask,
                    **FEATURE_PARAMS_TARGET
                )
            )


            if new_points is not None:

                target_points = (
                    new_points
                )

                feature_redetections += 1


    # ========================================================
    # NEXT FRAME
    # ========================================================

    prev_gray = gray.copy()


# ============================================================
# FINISH
# ============================================================

elapsed = (
    time.time() -
    start_time
)


processing_fps = (
    (frame_number - 1) / elapsed
    if elapsed > 0
    else 0
)


cap.release()

out.release()

csv_file.close()

cv2.destroyAllWindows()


# ============================================================
# RESULTS
# ============================================================

print("\n======================================")

print("       STAGE 6A RESULTS")

print("======================================")

print(
    "Frames processed        :",
    frame_number - 1
)

print(
    "Original video FPS      :",
    f"{fps:.2f}"
)

print(
    "Average processing FPS  :",
    f"{processing_fps:.2f}"
)

print(
    "Initial target features :",
    len(
        cv2.goodFeaturesToTrack(
            first_gray,
            mask=target_mask,
            **FEATURE_PARAMS_TARGET
        )
    )
)

print(
    "Minimum target features :",
    min_target_points_seen
)

print(
    "Maximum target features :",
    max_target_points_seen
)

print(
    "Minimum RANSAC inliers  :",
    (
        min_ransac_inliers_seen
        if min_ransac_inliers_seen != 999999
        else 0
    )
)

print(
    "Maximum RANSAC inliers  :",
    max_ransac_inliers_seen
)

print(
    "Accepted updates        :",
    accepted_updates
)

print(
    "Rejected updates        :",
    rejected_updates
)

print(
    "Recovery frames         :",
    recovery_frames
)

print(
    "Feature re-detections   :",
    feature_redetections
)

print(
    "Output video saved to   :",
    OUTPUT_VIDEO
)

print(
    "Tracking data saved to  :",
    CSV_FILE
)

print("======================================")