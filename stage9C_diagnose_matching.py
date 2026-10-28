
import cv2
import csv
import os

INPUT = "data/input.mp4"
OUT_DIR = "results/stage9C_diagnostics"

os.makedirs(OUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(INPUT)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open {INPUT}")

ret, first = cap.read()
if not ret:
    raise RuntimeError("Cannot read first frame")

# Select a tight box around the vehicle.
roi = cv2.selectROI("Select vehicle", first, False, False)
cv2.destroyAllWindows()

x, y, w, h = map(int, roi)
if w <= 0 or h <= 0:
    raise RuntimeError("Invalid selection")

template_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)[y:y+h, x:x+w]
template_gray = template_gray.copy()
template_edges = cv2.Canny(template_gray, 50, 150)

# Test smaller sizes as well as the original size.
scales = [0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00]

# Sample beginning, shrinking, edge, and return phases.
sample_frames = {1, 21, 101, 201, 260, 270, 280, 290,
                 295, 300, 305, 310, 315, 320, 325, 330}

csv_path = os.path.join(OUT_DIR, "matching_scores.csv")
rows = []
frame_no = 1

while True:
    if frame_no == 1:
        frame = first.copy()
    else:
        ret, frame = cap.read()
        if not ret:
            break

    if frame_no in sample_frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        preview = frame.copy()
        best = {
            "gray": (-1.0, None),
            "edge": (-1.0, None)
        }

        for scale in scales:
            tw = max(12, int(w * scale))
            th = max(12, int(h * scale))

            if tw >= gray.shape[1] or th >= gray.shape[0]:
                continue

            tg = cv2.resize(template_gray, (tw, th))
            te = cv2.Canny(tg, 50, 150)

            # Grayscale matching
            result_g = cv2.matchTemplate(
                gray, tg, cv2.TM_CCOEFF_NORMED
            )
            _, score_g, _, loc_g = cv2.minMaxLoc(result_g)

            if score_g > best["gray"][0]:
                best["gray"] = (
                    float(score_g),
                    (loc_g[0], loc_g[1], tw, th, scale)
                )

            # Edge matching
            if cv2.countNonZero(te) > 0:
                result_e = cv2.matchTemplate(
                    edges, te, cv2.TM_CCOEFF_NORMED
                )
                _, score_e, _, loc_e = cv2.minMaxLoc(result_e)

                if score_e > best["edge"][0]:
                    best["edge"] = (
                        float(score_e),
                        (loc_e[0], loc_e[1], tw, th, scale)
                    )

        for method, (score, box) in best.items():
            if box is not None:
                bx, by, bw, bh, scale = box
                color = (0, 255, 0) if method == "gray" else (255, 180, 0)
                cv2.rectangle(
                    preview, (bx, by), (bx + bw, by + bh), color, 2
                )
                cv2.putText(
                    preview, f"{method}: {score:.3f} scale={scale:.2f}",
                    (20, 30 if method == "gray" else 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
                )

                rows.append([
                    frame_no, method, score,
                    bx, by, bw, bh, scale
                ])
            else:
                rows.append([frame_no, method, -1, "", "", "", "", ""])

        cv2.imwrite(
            os.path.join(OUT_DIR, f"frame_{frame_no:03d}.jpg"),
            preview
        )

    frame_no += 1

cap.release()

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "frame", "method", "score",
        "x", "y", "width", "height", "scale"
    ])
    writer.writerows(rows)

print("Diagnostic complete.")
print("Scores:", csv_path)
print("Candidate images:", OUT_DIR)

