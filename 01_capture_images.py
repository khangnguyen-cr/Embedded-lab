import cv2
import os
from datetime import datetime

OUT_DIR = "face_detection_dataset/images"
os.makedirs(OUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Khong mo duoc camera. Kiem tra webcam va quyen truy cap.")

count = 0
print("Nhan phim 'c' de chup anh, 'q' de thoat.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Khong doc duoc frame tu camera.")
        break

    cv2.imshow("Capture Images", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"img_{ts}.jpg"
        path = os.path.join(OUT_DIR, filename)
        cv2.imwrite(path, frame)
        count += 1
        print("Saved:", path)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Tong so anh da chup:", count)
