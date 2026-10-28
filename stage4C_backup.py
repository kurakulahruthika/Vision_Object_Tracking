import cv2
import numpy as np
import time
import math

# ============================================================
# STAGE 4C
# ROBUST GLOBAL + LOCAL MOTION TRACKING
# ============================================================

VIDEO_PATH = "data/input.mp4"
OUTPUT_PATH = "results/stage4C_robust_tracking.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

print("Video opened successfully!")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Video FPS:", fps)
print("Resolution:", width, "x", height)
print("Total Frames:", total_frames)


# ============================================================
# READ FIRST FRAME
# ============================================================

ret, first_frame = cap.read()

if not ret:
    print("ERROR: Could not read first frame.")
    cap.release()
    exit()

old_gray = cv2.cvtColor(
    first_frame,
    cv2.COLOR_BGR2GRAY
)


# ============================================================
# SELECT TARGET
# ============================================================

print("\nSelect the vehicle you want to track.")

roi = cv2.selectROI(
    "Select Vehicle",
    first_frame,
    showCrosshair=True,
    fromCenter=False
)

cv2.destroyWindow("Select Vehicle")

x, y, w, h = map(int, roi)

if w == 0 or h == 0:
    print("No object selected.")
    cap.release()
    exit()

print("\nSelected object:")
print("x =", x)
print("y =", y)
print("width =", w)
print("height =", h)


# ============================================================
# FEATURE PARAMETERS
# ============================================================

target_params = dict(
    maxCorners=200,
    qualityLevel=0.01,
    minDistance=6,
    blockSize=7
)

global_params = dict(
    maxCorners=500,
    qualityLevel=0.01,
    minDistance=10,
    blockSize=7
)


# ============================================================
# LUCAS-KANADE PARAMETERS
# ============================================================

lk_params = dict(
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
# FEATURE DETECTION FUNCTION
# ============================================================

def detect_target_features(gray, x, y, w, h):

    mask = np.zeros_like(gray)

    margin = 10

    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(gray.shape[1], x + w + margin)
    y2 = min(gray.shape[0], y + h + margin)

    mask[y1:y2, x1:x2] = 255

    points = cv2.goodFeaturesToTrack(
        gray,
        mask=mask,
        **target_params
    )

    return points


def detect_global_features(gray, x, y, w, h):

    mask = np.ones_like(gray) * 255

    # Exclude target region
    x1 = max(0, x - 20)
    y1 = max(0, y - 20)
    x2 = min(gray.shape[1], x + w + 20)
    y2 = min(gray.shape[0], y + h + 20)

    mask[y1:y2, x1:x2] = 0

    points = cv2.goodFeaturesToTrack(
        gray,
        mask=mask,
        **global_params
    )

    return points


# ============================================================
# INITIAL FEATURES
# ============================================================

target_points = detect_target_features(
    old_gray,
    x,
    y,
    w,
    h
)

global_points = detect_global_features(
    old_gray,
    x,
    y,
    w,
    h
)

if target_points is None:
    print("ERROR: Could not detect target features.")
    cap.release()
    exit()

if global_points is None:
    print("ERROR: Could not detect global features.")
    cap.release()
    exit()

print(
    "Initial target feature points:",
    len(target_points)
)

print(
    "Initial global feature points:",
    len(global_points)
)


# ============================================================
# OUTPUT VIDEO
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)

if not out.isOpened():
    print("ERROR: Could not create output video.")
    cap.release()
    exit()


# ============================================================
# TRAJECTORY
# ============================================================

trajectory = np.zeros_like(first_frame)

center_x = x + w / 2
center_y = y + h / 2

center_history = []


# ============================================================
# STATISTICS
# ============================================================

frame_count = 0

processing_time = 0

min_target_features = 999999
max_target_features = 0

min_global_features = 999999
max_global_features = 0

min_global_inliers = 999999
max_global_inliers = 0

accepted_updates = 0
rejected_updates = 0

target_redetections = 0
global_redetections = 0


# ============================================================
# TRACKING CONFIDENCE
# ============================================================

confidence = 1.0


print("\n======================================")
print("STAGE 4C")
print("ROBUST GLOBAL + LOCAL TRACKING")
print("======================================")
print("Global motion compensation")
print("Local target motion")
print("RANSAC validation")
print("Confidence-based update")
print("Anti-drift protection")
print("Press Q to stop.")
print()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    start = time.time()

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # ========================================================
    # GLOBAL OPTICAL FLOW
    # ========================================================

    if (
        global_points is not None
        and len(global_points) >= 6
    ):

        new_global, status_global, error_global = (
            cv2.calcOpticalFlowPyrLK(
                old_gray,
                gray,
                global_points,
                None,
                **lk_params
            )
        )

    else:

        new_global = None
        status_global = None
        error_global = None


    good_global_old = np.empty(
        (0, 2),
        dtype=np.float32
    )

    good_global_new = np.empty(
        (0, 2),
        dtype=np.float32
    )


    if (
        new_global is not None
        and status_global is not None
        and error_global is not None
    ):

        status = status_global.ravel() == 1

        old_g = global_points[status]
        new_g = new_global[status]

        err_g = error_global[status].ravel()

        reliable = err_g < 20

        good_global_old = old_g[reliable]
        good_global_new = new_g[reliable]


    global_count = len(good_global_new)

    min_global_features = min(
        min_global_features,
        global_count
    )

    max_global_features = max(
        max_global_features,
        global_count
    )


    # ========================================================
    # GLOBAL RANSAC MODEL
    # ========================================================

    global_transform = None
    global_inliers = 0

    if len(good_global_new) >= 10:

        transform, inlier_mask = (
            cv2.estimateAffinePartial2D(
                good_global_old,
                good_global_new,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
                maxIters=2000,
                confidence=0.99
            )
        )

        if (
            transform is not None
            and inlier_mask is not None
        ):

            global_transform = transform

            global_inliers = int(
                np.sum(inlier_mask)
            )


    min_global_inliers = min(
        min_global_inliers,
        global_inliers
    )

    max_global_inliers = max(
        max_global_inliers,
        global_inliers
    )


    # ========================================================
    # VALIDATE GLOBAL MOTION
    # ========================================================

    global_valid = False

    if (
        global_transform is not None
        and global_inliers >= 20
    ):

        a = global_transform[0, 0]
        b = global_transform[0, 1]

        c = global_transform[1, 0]
        d = global_transform[1, 1]

        scale = math.sqrt(
            a * a + c * c
        )

        rotation = math.degrees(
            math.atan2(c, a)
        )

        dx = global_transform[0, 2]
        dy = global_transform[1, 2]

        if (
            0.85 <= scale <= 1.15
            and abs(rotation) <= 8
            and abs(dx) <= 120
            and abs(dy) <= 120
        ):

            global_valid = True


    # ========================================================
    # TARGET OPTICAL FLOW
    # ========================================================

    if (
        target_points is not None
        and len(target_points) >= 5
    ):

        new_target, status_target, error_target = (
            cv2.calcOpticalFlowPyrLK(
                old_gray,
                gray,
                target_points,
                None,
                **lk_params
            )
        )

    else:

        new_target = None
        status_target = None
        error_target = None


    good_target_old = np.empty(
        (0, 2),
        dtype=np.float32
    )

    good_target_new = np.empty(
        (0, 2),
        dtype=np.float32
    )


    if (
        new_target is not None
        and status_target is not None
        and error_target is not None
    ):

        status = status_target.ravel() == 1

        old_t = target_points[status]
        new_t = new_target[status]

        err_t = error_target[status].ravel()

        reliable = err_t < 20

        good_target_old = old_t[reliable]
        good_target_new = new_t[reliable]


    target_count = len(good_target_new)

    min_target_features = min(
        min_target_features,
        target_count
    )

    max_target_features = max(
        max_target_features,
        target_count
    )


    # ========================================================
    # LOCAL TARGET MOTION
    # ========================================================

    local_transform = None
    local_inliers = 0

    if len(good_target_new) >= 8:

        local_transform, local_mask = (
            cv2.estimateAffinePartial2D(
                good_target_old,
                good_target_new,
                method=cv2.RANSAC,
                ransacReprojThreshold=4.0,
                maxIters=1500,
                confidence=0.98
            )
        )

        if (
            local_transform is not None
            and local_mask is not None
        ):

            local_inliers = int(
                np.sum(local_mask)
            )


    # ========================================================
    # CALCULATE TARGET MOTION
    # ========================================================

    target_dx = 0.0
    target_dy = 0.0

    if local_transform is not None:

        target_dx = float(
            local_transform[0, 2]
        )

        target_dy = float(
            local_transform[1, 2]
        )

    elif len(good_target_new) >= 5:

        movement = (
            good_target_new -
            good_target_old
        )

        target_dx = float(
            np.median(
                movement[:, 0]
            )
        )

        target_dy = float(
            np.median(
                movement[:, 1]
            )
        )


    # ========================================================
    # GLOBAL MOTION AT TARGET
    # ========================================================

    global_dx_at_target = 0.0
    global_dy_at_target = 0.0

    if global_valid:

        center = np.array(
            [
                [[center_x, center_y]]
            ],
            dtype=np.float32
        )

        transformed = cv2.transform(
            center,
            global_transform
        )[0, 0]

        global_dx_at_target = (
            transformed[0] - center_x
        )

        global_dy_at_target = (
            transformed[1] - center_y
        )


    # ========================================================
    # RESIDUAL TARGET MOTION
    # ========================================================

    residual_dx = (
        target_dx -
        global_dx_at_target
    )

    residual_dy = (
        target_dy -
        global_dy_at_target
    )


    # ========================================================
    # CONFIDENCE CALCULATION
    # ========================================================

    confidence = 0.0

    # Target feature confidence
    if target_count >= 50:
        confidence += 0.30

    elif target_count >= 20:
        confidence += 0.20

    elif target_count >= 10:
        confidence += 0.10


    # Global model confidence
    if global_valid:

        if global_inliers >= 80:
            confidence += 0.30

        elif global_inliers >= 40:
            confidence += 0.20

        else:
            confidence += 0.10


    # Local geometric confidence
    if local_inliers >= 30:
        confidence += 0.30

    elif local_inliers >= 15:
        confidence += 0.20

    elif local_inliers >= 8:
        confidence += 0.10


    # ========================================================
    # MOTION LIMIT
    # ========================================================

    motion_magnitude = math.sqrt(
        residual_dx ** 2 +
        residual_dy ** 2
    )


    # Large unexplained motion = suspicious
    if motion_magnitude > 40:

        confidence -= 0.30

    elif motion_magnitude > 25:

        confidence -= 0.15


    confidence = max(
        0.0,
        min(1.0, confidence)
    )


    # ========================================================
    # UPDATE TRACKER ONLY WHEN RELIABLE
    # ========================================================

    update_allowed = (
        confidence >= 0.45
        and target_count >= 8
    )


    if update_allowed:

        # Global movement
        movement_x = 0
        movement_y = 0

        if global_valid:

            movement_x = global_dx_at_target
            movement_y = global_dy_at_target

        else:

            # If global model fails,
            # use local target motion conservatively.
            movement_x = target_dx
            movement_y = target_dy

            movement_x *= 0.5
            movement_y *= 0.5


        # Add target-specific residual
        movement_x += residual_dx
        movement_y += residual_dy


        # ====================================================
        # HARD ANTI-DRIFT LIMIT
        # ====================================================

        maximum_step = 35.0

        magnitude = math.sqrt(
            movement_x ** 2 +
            movement_y ** 2
        )

        if magnitude > maximum_step:

            factor = (
                maximum_step /
                magnitude
            )

            movement_x *= factor
            movement_y *= factor


        center_x += movement_x
        center_y += movement_y

        accepted_updates += 1

    else:

        rejected_updates += 1


    # ========================================================
    # KEEP CENTER INSIDE FRAME
    # ========================================================

    center_x = max(
        w / 2,
        min(center_x, width - w / 2)
    )

    center_y = max(
        h / 2,
        min(center_y, height - h / 2)
    )


    # ========================================================
    # UPDATE BOUNDING BOX
    # ========================================================

    x = int(
        center_x - w / 2
    )

    y = int(
        center_y - h / 2
    )


    # ========================================================
    # TRAJECTORY
    # ========================================================

    current_center = (
        int(center_x),
        int(center_y)
    )

    center_history.append(
        current_center
    )

    if len(center_history) >= 2:

        cv2.line(
            trajectory,
            center_history[-2],
            center_history[-1],
            (0, 255, 255),
            2
        )


    # ========================================================
    # DRAW TARGET FEATURES
    # ========================================================

    for p in good_target_new:

        px, py = p.ravel()

        cv2.circle(
            frame,
            (int(px), int(py)),
            3,
            (0, 255, 0),
            -1
        )


    # ========================================================
    # DRAW BOUNDING BOX
    # ========================================================

    if confidence >= 0.45:

        box_color = (0, 255, 0)

    else:

        box_color = (0, 0, 255)


    cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        box_color,
        3
    )


    # ========================================================
    # STATUS
    # ========================================================

    if confidence >= 0.70:

        status_text = "HIGH CONFIDENCE"

    elif confidence >= 0.45:

        status_text = "MEDIUM CONFIDENCE"

    else:

        status_text = "LOW CONFIDENCE - UPDATE REJECTED"


    cv2.putText(
        frame,
        "STAGE 4C",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        status_text,
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Target features: {target_count}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Global inliers: {global_inliers}",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Local inliers: {local_inliers}",
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Residual motion: {motion_magnitude:.2f}",
        (20, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ========================================================
    # ADD TRAJECTORY
    # ========================================================

    display = cv2.add(
        frame,
        trajectory
    )


    # ========================================================
    # SAVE
    # ========================================================

    out.write(display)

    cv2.imshow(
        "Stage 4C - Robust Tracking",
        display
    )


    # ========================================================
    # RE-DETECT TARGET FEATURES
    # ========================================================

    if (
        target_points is None
        or len(good_target_new) < 30
        or frame_count % 25 == 0
    ):

        new_points = detect_target_features(
            gray,
            x,
            y,
            w,
            h
        )

        if new_points is not None:

            target_points = new_points

            target_redetections += 1

        elif len(good_target_new) > 0:

            target_points = (
                good_target_new.reshape(
                    -1,
                    1,
                    2
                )
            )

    else:

        if len(good_target_new) > 0:

            target_points = (
                good_target_new.reshape(
                    -1,
                    1,
                    2
                )
            )


    # ========================================================
    # RE-DETECT GLOBAL FEATURES
    # ========================================================

    if (
        global_points is None
        or len(good_global_new) < 100
        or frame_count % 40 == 0
    ):

        new_global = detect_global_features(
            gray,
            x,
            y,
            w,
            h
        )

        if new_global is not None:

            global_points = new_global

            global_redetections += 1

        elif len(good_global_new) > 0:

            global_points = (
                good_global_new.reshape(
                    -1,
                    1,
                    2
                )
            )

    else:

        if len(good_global_new) > 0:

            global_points = (
                good_global_new.reshape(
                    -1,
                    1,
                    2
                )
            )


    # ========================================================
    # UPDATE PREVIOUS FRAME
    # ========================================================

    old_gray = gray.copy()

    processing_time += (
        time.time() - start
    )


    # ========================================================
    # QUIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# RELEASE
# ============================================================

cap.release()
out.release()
cv2.destroyAllWindows()


# ============================================================
# RESULTS
# ============================================================

average_processing_fps = (
    frame_count / processing_time
    if processing_time > 0
    else 0
)

print("\n======================================")
print("       STAGE 4C RESULTS")
print("======================================")

print(
    f"Frames processed        : {frame_count}"
)

print(
    f"Original video FPS      : {fps:.2f}"
)

print(
    f"Average processing FPS  : "
    f"{average_processing_fps:.2f}"
)

print(
    f"Minimum target features: "
    f"{min_target_features}"
)

print(
    f"Maximum target features: "
    f"{max_target_features}"
)

print(
    f"Minimum global features: "
    f"{min_global_features}"
)

print(
    f"Maximum global features: "
    f"{max_global_features}"
)

print(
    f"Minimum global inliers : "
    f"{min_global_inliers}"
)

print(
    f"Maximum global inliers : "
    f"{max_global_inliers}"
)

print(
    f"Accepted updates       : "
    f"{accepted_updates}"
)

print(
    f"Rejected updates       : "
    f"{rejected_updates}"
)

print(
    f"Target re-detections   : "
    f"{target_redetections}"
)

print(
    f"Global re-detections   : "
    f"{global_redetections}"
)

print(
    f"Output video saved to  : "
    f"{OUTPUT_PATH}"
)

print("======================================")