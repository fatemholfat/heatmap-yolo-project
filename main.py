from ultralytics import YOLO
import cv2
import numpy as np
import os

# -----------------------------
# تنظیمات
# -----------------------------

VIDEO_PATH = "input.mp4"
OUTPUT_DIR = "output"

# مدل YOLO
MODEL = "yolo11n.pt"

# فقط Person
PERSON_CLASS = 0

# -----------------------------
# آماده سازی
# -----------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("ویدیو باز نشد!")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 25

# نقشه حرارتی
heatmap = np.zeros((height, width), dtype=np.float32)

# -----------------------------
# خروجی ویدیو
# -----------------------------

output_video = os.path.join(
    OUTPUT_DIR,
    "tracked.mp4"
)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height)
)

# -----------------------------
# پردازش
# -----------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Tracking
    results = model.track(
        frame,
        persist=True,
        classes=[PERSON_CLASS],
        verbose=False
    )

    result = results[0]

    if result.boxes is not None:

        boxes = result.boxes

        # مختصات افراد
        xyxy = boxes.xyxy.cpu().numpy()

        # ID هر فرد
        if boxes.id is not None:
            track_ids = boxes.id.cpu().numpy().astype(int)
        else:
            track_ids = []

        for i, box in enumerate(xyxy):

            x1, y1, x2, y2 = map(int, box)

            # نقطه کف پای فرد
            cx = int((x1 + x2) / 2)
            cy = int(y2)

            # افزایش تراکم در محل حضور/حرکت
            cv2.circle(
                heatmap,
                (cx, cy),
                15,
                1,
                -1
            )

            # رسم شخص
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # ID
            if i < len(track_ids):

                cv2.putText(
                    frame,
                    f"ID {track_ids[i]}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    writer.write(frame)

    cv2.imshow(
        "Store Tracking",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
writer.release()

cv2.destroyAllWindows()

# -----------------------------
# ساخت Heatmap نهایی
# -----------------------------

# نرم کردن نقشه
heatmap = cv2.GaussianBlur(
    heatmap,
    (0, 0),
    sigmaX=25,
    sigmaY=25
)

# نرمال سازی
heatmap = cv2.normalize(
    heatmap,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

heatmap = heatmap.astype(np.uint8)

# تبدیل به رنگ
heatmap_color = cv2.applyColorMap(
    heatmap,
    cv2.COLORMAP_JET
)

# تصویر پس زمینه
cap = cv2.VideoCapture(VIDEO_PATH)
ret, background = cap.read()
cap.release()

if ret:

    background = cv2.resize(
        background,
        (width, height)
    )

    # ترکیب تصویر فروشگاه و Heatmap
    result = cv2.addWeighted(
        background,
        0.55,
        heatmap_color,
        0.45,
        0
    )

    output_image = os.path.join(
        OUTPUT_DIR,
        "store_heatmap.jpg"
    )

    cv2.imwrite(
        output_image,
        result
    )

    print("Heatmap ساخته شد:")
    print(output_image)

print("پردازش تمام شد.")