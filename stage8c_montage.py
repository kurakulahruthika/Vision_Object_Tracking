
import os
import cv2
import math

folder = "results/stage8/overlays"
output = "results/stage8/overlay_montage.jpg"

files = sorted(
    [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".png"))
    ]
)

if not files:
    raise RuntimeError("No overlay images found.")

thumb_width = 640
thumb_height = 360
label_height = 30
columns = 2
rows = math.ceil(len(files) / columns)

canvas = 255 * (
    __import__("numpy").ones(
        (rows * (thumb_height + label_height),
         columns * thumb_width, 3),
        dtype="uint8"
    )
)

for i, filename in enumerate(files):
    path = os.path.join(folder, filename)
    image = cv2.imread(path)

    if image is None:
        continue

    image = cv2.resize(image, (thumb_width, thumb_height))

    row = i // columns
    col = i % columns
    x = col * thumb_width
    y = row * (thumb_height + label_height)

    canvas[y:y + thumb_height, x:x + thumb_width] = image

    cv2.putText(
        canvas,
        filename,
        (x + 8, y + thumb_height + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2
    )

cv2.imwrite(output, canvas)
print("Montage saved to:", output)
