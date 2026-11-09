import cv2
import time
from ultralytics import YOLO
import os

print("加载 YOLO 模型...")
car_model = YOLO('yolov8n.pt')
plate_model = YOLO('best.pt')

# 摄像头
cap = cv2.VideoCapture(1)  # iVCam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("【进程A】YOLO 检测启动，按 'q' 退出...")
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    frame_count += 1

    # 软件地感线圈（检测是否有车）
    if frame_count % 3 == 0:
        car_results = car_model(frame, verbose=False, classes=[2, 5, 7])
        has_car = False
        for r in car_results:
            for box in r.boxes:
                if (box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1]) > 20000:
                    has_car = True
                    break
            if has_car: break

        if has_car:
            plate_results = plate_model(frame, verbose=False)
            for result in plate_results:
                for box in result.boxes:
                    if box.conf[0].item() < 0.6: continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    if (x2 - x1) * (y2 - y1) < 1500: continue

                    # 裁剪车牌并放大
                    pad = 5
                    plate_img = frame[
                        max(0, y1 - pad):min(frame.shape[0], y2 + pad), max(0, x1 - pad):min(frame.shape[1], x2 + pad)]
                    if plate_img.size == 0: continue

                    plate_img_resized = cv2.resize(plate_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

                    # 保存为临时图片，交给进程B处理
                    cv2.imwrite('temp_plate.jpg', plate_img_resized)

                    # 画框显示
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Sending to OCR...", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                                2)

    cv2.imshow("Process A - YOLO Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()