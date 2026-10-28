import cv2
import numpy as np
import csv
import os
import time

# ============================================================
# STAGE 9B: TRUSTED-STATE PRESERVATION + RECOVERY VERIFICATION
# Experimental improvement over Stage 9A.
# Important: feature consistency is NOT semantic target identity.
# This stage adds hard box-growth guards and requires a candidate
# to survive a short verification period before trusting it.
# ============================================================

INPUT_VIDEO = "data/input.mp4"
OUTPUT_VIDEO = "results/stage9B_trusted_recovery.mp4"
CSV_FILE = "results/stage9B_trusted_recovery_data.csv"
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

# Stage 9B safeguards.
MAX_AREA_GROWTH = 1.18        # maximum candidate area growth relative to trusted box
MAX_DIM_GROWTH = 1.20         # maximum per-frame width/height growth
MAX_DIM_SHRINK = 0.70         # minimum per-frame width/height ratio
MAX_TRUSTED_CENTER_JUMP = 85.0
VERIFY_FRAMES = 3             # candidate must persist through consecutive valid updates
MIN_VERIFY_INLIERS = 8
MAX_VERIFY_CENTER_JUMP = 70.0
MAX_BOX_AREA_FRACTION = 0.18  # hard limit vs full frame; never silently expand to huge box

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
    corners = np.array([[[x,y]], [[x+bw,y]], [[x+bw,y+bh]], [[x,y+bh]]], np.float32)
    pts = cv2.transform(corners, M).reshape(-1,2)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    return float((xmin+xmax)/2), float((ymin+ymax)/2), float(xmax-xmin), float(ymax-ymin)

def clamp_size(candidate_w, candidate_h, trusted_box, fw, fh):
    _, _, old_w, old_h = trusted_box
    new_w = float(np.clip(candidate_w, old_w * MAX_DIM_SHRINK, old_w * MAX_DIM_GROWTH))
    new_h = float(np.clip(candidate_h, old_h * MAX_DIM_SHRINK, old_h * MAX_DIM_GROWTH))
    max_area = fw * fh * MAX_BOX_AREA_FRACTION
    if new_w * new_h > max_area:
        factor = (max_area / (new_w * new_h)) ** 0.5
        new_w *= factor
        new_h *= factor
    return new_w, new_h

def box_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax,bx), max(ay,by)
    x2, y2 = min(ax+aw,bx+bw), min(ay+ah,by+bh)
    inter = max(0,x2-x1) * max(0,y2-y1)
    union = aw*ah + bw*bh - inter
    return inter/union if union > 0 else 0.0

def center_of(box):
    x,y,w,h = box
    return np.array([x+w/2.0, y+h/2.0], np.float32)

def detect_features_local(gray, center, trusted_box, fw, fh):
    _, _, bw, bh = trusted_box
    sw = min(fw*0.35, max(2.0*bw, 120.0))
    sh = min(fh*0.35, max(2.0*bh, 120.0))
    search = clip_box(center[0], center[1], sw, sh, fw, fh)
    pts = detect_features(gray, search)
    if pts is None:
        return None
    arr = pts.reshape(-1,2)
    radius = max(55.0, 0.9*max(bw,bh))
    keep = np.linalg.norm(arr-center.reshape(1,2), axis=1) <= radius
    arr = arr[keep]
    return arr.reshape(-1,1,2).astype(np.float32) if len(arr) >= MIN_TARGET_POINTS else None

cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise RuntimeError(f"Could not open input video: {INPUT_VIDEO}")
fps = cap.get(cv2.CAP_PROP_FPS)
fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
ret, first_frame = cap.read()
if not ret:
    cap.release()
    raise RuntimeError("Could not read first frame.")

first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
print("Select the target vehicle in the first frame; press ENTER/SPACE.")
roi = cv2.selectROI("Select Target", first_frame, False, False)
cv2.destroyWindow("Select Target")
x,y,w,h = map(int,roi)
if w <= 0 or h <= 0:
    cap.release()
    raise RuntimeError("Invalid target ROI.")
box = (x,y,w,h)
center = center_of(box)
points = detect_features(first_gray, box)
if points is None or len(points) < 4:
    cap.release()
    raise RuntimeError("Too few features. Select a textured part of the vehicle.")

kalman = cv2.KalmanFilter(4,2)
kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
kalman.processNoiseCov = np.eye(4,dtype=np.float32)*0.03
kalman.measurementNoiseCov = np.eye(2,dtype=np.float32)*2.0
kalman.errorCovPost = np.eye(4,dtype=np.float32)
kalman.statePost = np.array([[center[0]],[center[1]],[0],[0]],np.float32)

out = cv2.VideoWriter(OUTPUT_VIDEO,cv2.VideoWriter_fourcc(*"mp4v"),fps,(fw,fh))
if not out.isOpened():
    cap.release()
    raise RuntimeError(f"Could not create output video: {OUTPUT_VIDEO}")
csvf = open(CSV_FILE,"w",newline="",encoding="utf-8")
writer = csv.writer(csvf)
writer.writerow([
    "frame","x","y","width","height","center_x","center_y","dx","dy","velocity",
    "target_points","ransac_inliers","fb_error","confidence","tracking_status",
    "candidate_accepted","box_growth_rejected","verification_count","recovery_attempted",
    "box_area_fraction","center_prediction_distance"
])

prev_gray = first_gray.copy()
trusted_box = box
trusted_center = center.copy()
candidate_points = None
verification_count = 0
candidate_center = None
candidate_size = None
frame_no = 1
accepted = rejected = redetections = growth_rejections = 0
start = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_no += 1
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    pred = kalman.predict()
    pred_center = np.array([pred[0,0],pred[1,0]],np.float32)
    dx=dy=velocity=0.0
    fb_error=999.0
    confidence=0.0
    inliers=0
    status="PREDICTION"
    candidate_accepted=0
    box_growth_rejected=0
    recovery_attempted=0
    measured_box=None
    measured_center=None

    # Always estimate motion relative to the last trusted box, not an unbounded box.
    active_box = trusted_box

    if points is not None and len(points) >= 4:
        nxt, stf, _ = cv2.calcOpticalFlowPyrLK(prev_gray,gray,points,None,**LK_PARAMS)
        if nxt is not None:
            back, stb, _ = cv2.calcOpticalFlowPyrLK(gray,prev_gray,nxt,None,**LK_PARAMS)
            if back is not None:
                old=points.reshape(-1,2); new=nxt.reshape(-1,2); bck=back.reshape(-1,2)
                fb=np.linalg.norm(old-bck,axis=1)
                valid=stf.reshape(-1).astype(bool)&stb.reshape(-1).astype(bool)&(fb<FB_THRESHOLD)&np.isfinite(new).all(axis=1)
                oldv,newv=old[valid],new[valid]
                if len(oldv)>=4:
                    fb_error=float(np.mean(fb[valid]))
                    M,rm=cv2.estimateAffinePartial2D(oldv,newv,method=cv2.RANSAC,ransacReprojThreshold=3.0,maxIters=2000,confidence=0.99)
                    if M is not None and rm is not None:
                        mask=rm.reshape(-1).astype(bool)
                        inlier_pts=newv[mask]
                        inliers=len(inlier_pts)
                        if inliers>=MIN_RANSAC_INLIERS:
                            cx,cy,cw,ch=transform_box(M,active_box)
                            measured_center=np.array([cx,cy],np.float32)
                            pred_dist=float(np.linalg.norm(measured_center-pred_center))
                            point_score=min(inliers/30.0,1.0)
                            fb_score=float(np.clip(1.0-fb_error/FB_THRESHOLD,0,1))
                            pred_score=float(np.clip(1.0-pred_dist/MAX_CENTER_JUMP,0,1))
                            confidence=0.45*point_score+0.30*fb_score+0.25*pred_score

                            # Hard candidate validation against trusted state.
                            proposed_w,proposed_h=clamp_size(cw,ch,active_box,fw,fh)
                            old_area=active_box[2]*active_box[3]
                            proposed_area=proposed_w*proposed_h
                            area_ratio=proposed_area/max(1.0,old_area)
                            dim_ratio_w=proposed_w/max(1.0,active_box[2])
                            dim_ratio_h=proposed_h/max(1.0,active_box[3])
                            center_shift=float(np.linalg.norm(measured_center-trusted_center))
                            growth_ok=(area_ratio <= MAX_AREA_GROWTH and
                                       dim_ratio_w <= MAX_DIM_GROWTH and
                                       dim_ratio_h <= MAX_DIM_GROWTH and
                                       proposed_area <= fw*fh*MAX_BOX_AREA_FRACTION)
                            motion_ok=center_shift <= MAX_TRUSTED_CENTER_JUMP
                            prediction_ok=pred_dist <= MAX_CENTER_JUMP
                            geometry_ok=inliers >= MIN_RANSAC_INLIERS and fb_error < FB_THRESHOLD
                            candidate_ok=growth_ok and motion_ok and prediction_ok and geometry_ok and confidence >= 0.30

                            if not growth_ok:
                                growth_rejections += 1
                                box_growth_rejected=1
                                status="BOX_GROWTH_REJECTED"
                                candidate_points=None
                                verification_count=0
                            elif candidate_ok:
                                # Require several consecutive plausible candidates before
                                # updating the trusted state after a recovery/transition.
                                if candidate_center is not None and np.linalg.norm(measured_center-candidate_center) <= MAX_VERIFY_CENTER_JUMP:
                                    verification_count += 1
                                else:
                                    verification_count=1
                                candidate_center=measured_center.copy()
                                candidate_size=(proposed_w,proposed_h)
                                candidate_points=inlier_pts.reshape(-1,1,2).astype(np.float32)

                                if verification_count >= VERIFY_FRAMES and inliers >= MIN_VERIFY_INLIERS:
                                    new_box=clip_box(cx,cy,proposed_w,proposed_h,fw,fh)
                                    old_center=trusted_center.copy()
                                    trusted_box=new_box
                                    trusted_center=center_of(trusted_box)
                                    points=candidate_points
                                    dx=float(trusted_center[0]-old_center[0])
                                    dy=float(trusted_center[1]-old_center[1])
                                    velocity=float(np.hypot(dx,dy))
                                    kalman.correct(np.array([[trusted_center[0]],[trusted_center[1]]],np.float32))
                                    accepted += 1
                                    candidate_accepted=1
                                    status="TRACKING_VERIFIED"
                                    candidate_center=None
                                    candidate_size=None
                                    candidate_points=None
                                    verification_count=0
                                else:
                                    status="VERIFYING_CANDIDATE"
                                    points=candidate_points
                            else:
                                rejected += 1
                                status="CANDIDATE_REJECTED"
                                candidate_center=None
                                candidate_size=None
                                candidate_points=None
                                verification_count=0
                                points=None
                        else:
                            status="FEW_INLIERS"
                            points=None
                            verification_count=0
                    else:
                        status="NO_RANSAC_MODEL"
                        points=None
                        verification_count=0
                else:
                    status="FEW_VALID_POINTS"
                    points=None
                    verification_count=0
            else:
                status="BACKWARD_FLOW_FAILED"
                points=None
                verification_count=0
        else:
            status="FORWARD_FLOW_FAILED"
            points=None
            verification_count=0
    else:
        points=None
        status="FEW_POINTS"

    # If candidate failed/lost, search near the predicted location but do not move
    # trusted box until verified tracking measurements become consistent.
    if points is None or len(points)<MIN_TARGET_POINTS:
        recovery_attempted=1
        recovered=detect_features_local(gray,pred_center,trusted_box,fw,fh)
        if recovered is not None:
            points=recovered
            redetections+=1
            status="LOCAL_RECOVERY_VERIFY"
            candidate_center=None
            verification_count=0
        elif status in ("PREDICTION","FEW_POINTS","FEW_INLIERS","NO_RANSAC_MODEL","CANDIDATE_REJECTED","BOX_GROWTH_REJECTED"):
            status="RECOVERY_FAILED"

    # Render only the trusted box; never render an unverified candidate box.
    x,y,w,h=trusted_box
    center_x,center_y=trusted_center
    area_fraction=(w*h)/float(fw*fh)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    if points is not None:
        for p in points.reshape(-1,2):
            px,py=int(p[0]),int(p[1])
            if 0<=px<fw and 0<=py<fh:
                cv2.circle(frame,(px,py),2,(0,255,255),-1)
    cv2.circle(frame,(int(center_x),int(center_y)),4,(0,0,255),-1)
    cv2.putText(frame,f"Status: {status}",(20,30),cv2.FONT_HERSHEY_SIMPLEX,0.58,(255,255,255),2)
    cv2.putText(frame,f"Confidence: {confidence:.2f}  Verify: {verification_count}/{VERIFY_FRAMES}",(20,60),cv2.FONT_HERSHEY_SIMPLEX,0.58,(255,255,255),2)
    cv2.putText(frame,f"Points: {0 if points is None else len(points)}  Inliers: {inliers}",(20,90),cv2.FONT_HERSHEY_SIMPLEX,0.58,(255,255,255),2)
    cv2.putText(frame,f"Box area: {area_fraction:.3f}",(20,120),cv2.FONT_HERSHEY_SIMPLEX,0.58,(255,255,255),2)

    writer.writerow([frame_no,x,y,w,h,center_x,center_y,dx,dy,velocity,
                     0 if points is None else len(points),inliers,fb_error,confidence,status,
                     candidate_accepted,box_growth_rejected,verification_count,recovery_attempted,
                     area_fraction, float(np.linalg.norm(measured_center-pred_center)) if measured_center is not None else -1])
    out.write(frame)
    cv2.imshow("Stage 9B - Trusted-State Recovery Verification",frame)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
    prev_gray=gray.copy()

elapsed=time.time()-start
processed=frame_no-1
cap.release(); out.release(); csvf.close(); cv2.destroyAllWindows()
print("\n======================================")
print("STAGE 9B RESULTS")
print("======================================")
print("Frames processed       :",processed)
print("Original video FPS     :",f"{fps:.2f}")
print("Average processing FPS :",f"{processed/elapsed:.2f}" if elapsed>0 else "N/A")
print("Verified updates       :",accepted)
print("Rejected candidates    :",rejected)
print("Box-growth rejections  :",growth_rejections)
print("Local re-detections    :",redetections)
print("Output video saved to  :",OUTPUT_VIDEO)
print("Tracking data saved to:",CSV_FILE)
print("NOTE: Recovery features are not semantic identity verification.")
print("======================================")
