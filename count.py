from ultralytics import YOLO
import cv2

# -----------------------------
# تنظیمات
# -----------------------------

VIDEO_PATH = "input.mp4"

# مدل YOLO
model = YOLO("yolo11n.pt")

# -----------------------------
# باز کردن ویدیو
# -----------------------------

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("ویدیو باز نشد!")

# ابعاد تصویر
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# -----------------------------
# خط شمارش
# -----------------------------

# خط افقی در وسط تصویر
LINE_Y = height // 2

# -----------------------------
# شمارنده‌ها
# -----------------------------

people_in = 0
people_out = 0

# ID هایی که قبلاً شمارش شده‌اند
counted_in = set()
counted_out = set()

# آخرین موقعیت هر فرد
previous_positions = {}

# -----------------------------
# پردازش ویدیو
# -----------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Tracking افراد
    results = model.track(
        frame,
        persist=True,
        classes=[0],       # فقط person
        tracker="bytetrack.yaml",
        verbose=False
    )

    result = results[0]

    # رسم خط شمارش
    cv2.line(
        frame,
        (0, LINE_Y),
        (width, LINE_Y),
        (255, 0, 0),
        3
    )

    if result.boxes is not None and result.boxes.id is not None:

        boxes = result.boxes

        xyxy = boxes.xyxy.cpu().numpy()
        track_ids = boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(xyxy, track_ids):

            x1, y1, x2, y2 = map(int, box)

            # مرکز شخص
            cx = int((x1 + x2) / 2)

            # نقطه پایین شخص
            cy = int(y2)

            # -----------------------------
            # موقعیت قبلی
            # -----------------------------

            previous_y = previous_positions.get(track_id)

            # -----------------------------
            # تشخیص عبور از خط
            # -----------------------------

            if previous_y is not None:

                # حرکت از بالا به پایین
                if (
                    previous_y < LINE_Y
                    and cy >= LINE_Y
                    and track_id not in counted_in
                    and track_id not in counted_out
                ):

                    people_in += 1
                    counted_in.add(track_id)

                # حرکت از پایین به بالا
                elif (
                    previous_y > LINE_Y
                    and cy <= LINE_Y
                    and track_id not in counted_out
                    and track_id not in counted_in
                ):

                    people_out += 1
                    counted_out.add(track_id)

            # ذخیره موقعیت فعلی
            previous_positions[track_id] = cy

            # -----------------------------
            # رسم شخص
            # -----------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # نمایش ID
            cv2.putText(
                frame,
                f"ID: {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # -----------------------------
    # نمایش آمار
    # -----------------------------

    cv2.rectangle(
        frame,
        (10, 10),
        (280, 110),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"IN: {people_in}",
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"OUT: {people_out}",
        (25, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2
    )

    # نمایش
    cv2.imshow(
        "Store People Counter",
        frame
    )

    # خروج با Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------
# پایان
# -----------------------------

cap.release()
cv2.destroyAllWindows()

print("--------------------------------")
print("نتیجه نهایی")
print("--------------------------------")
print(f"افراد وارد شده: {people_in}")
print(f"افراد خارج شده: {people_out}")
print(f"تعداد خالص افراد داخل: {people_in - people_out}")