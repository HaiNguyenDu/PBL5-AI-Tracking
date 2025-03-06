import cv2
import numpy as np
import pyfirmata
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from filterpy.kalman import KalmanFilter

# Kết nối Arduino
port = "/dev/ttyUSB0"
board = pyfirmata.Arduino(port)
servo_pinX = board.get_pin('d:9:s')
servo_pinY = board.get_pin('d:10:s')

# Load YOLO model
model = YOLO("yolo11n.pt")

# Deep SORT Tracker
tracker = DeepSort(max_age=30, max_iou_distance=0.5)

# Kalman Filter
kf = KalmanFilter(dim_x=4, dim_z=2)
kf.F = np.array([[1, 1, 0, 0], 
                 [0, 1, 0, 0], 
                 [0, 0, 1, 1], 
                 [0, 0, 0, 1]])
kf.H = np.array([[1, 0, 0, 0], 
                 [0, 0, 1, 0]])
kf.P *= 1000
kf.x = np.array([0, 0, 0, 0])

frame_step = 5  # Số frame giữa các lần YOLO chạy 
ws, hs = 1280, 720 # Kích thước khung hình mong muốn
selected_id = None  # ID của đối tượng qđang theo dõi

# Mở Camera và set kích thước khung hình
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, ws)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hs)

# Tạo cửa sổ hiển thị có thể thay đổi kích thước
cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)

frame_count = 0  # Đếm số frame
detections = []  # Lưu kết quả YOLO giữa các lần chạy

# Hàm kiểm tra IoU để tránh trùng ID
def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2
    xi1, yi1, xi2, yi2 = max(x1, x1_), max(y1, y1_), min(x2, x2_), min(y2, y2_)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# Sự kiện click chuột để chọn đối tượng
def select_object(event, x, y, flags, param):
    global selected_id
    if event == cv2.EVENT_LBUTTONDOWN:
        for track in param:
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            if x1 <= x <= x2 and y1 <= y <= y2:
                selected_id = track.track_id
                print(f"Chọn đối tượng ID: {selected_id}")

cv2.setMouseCallback("Tracking", select_object, param=[])

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame_count += 1

    if frame_count % frame_step == 0:
        results = model(frame)
        new_detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            if class_id == 0:  # Chỉ giữ lại người
                new_detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, class_id))
        detections = new_detections

    # Cập nhật tracker
    tracks = tracker.update_tracks(detections, frame=frame)
    cv2.setMouseCallback("Tracking", select_object, param=tracks)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if selected_id is not None and track_id == selected_id:
            if frame_count % frame_step == 0:
                kf.update([cx, cy])
            else:
                kf.predict()
            predicted = kf.x[:2]
            # servo_x = np.clip(np.interp(predicted[0], [0, ws], [180, 0]), 0, 180)
            # servo_y = np.clip(np.interp(predicted[1], [0, hs], [180, 0]), 0, 180)
            servo_x = np.interp(predicted[0], [0, ws], [180, 0])
            servo_y = np.interp(predicted[1], [0, hs], [180, 0])
            servo_pinX.write(servo_x)
            servo_pinY.write(servo_y)

        color = (0, 255, 0) if track_id == selected_id else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
