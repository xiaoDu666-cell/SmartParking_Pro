import cv2
import time
from ultralytics import YOLO
import hyperlpr3 as lpr3
import parking_system as db

print("正在加载模型与数据库...")
db.init_db()
car_model = YOLO('yolov8n.pt')
cathode = lpr3.LicensePlateCatcher()

current_mode = 1

# ================= 核心：模拟地感线圈与道闸的状态机 =================
gate_locked = False  # 道闸是否锁死（代表车正在过闸机）
lock_timer = 0  # 锁死计时器
LOCK_DURATION = 5.0  # 闸机锁死时间（5秒内不识别第二辆车，相当于车开过去的时间）
last_logged_plate = ""  # 记录刚刚放行的车牌，防止同一辆车反复触发


def process_plate(plate_number):
    """处理计费逻辑"""
    if current_mode == 1:
        if db.check_in_parking(plate_number):
            print(f"⏳ 车辆 {plate_number} 已入场，忽略。")
            return False
        db.enter_parking(plate_number)
        return True
    else:
        fee = db.exit_parking(plate_number)
        if fee is not None:
            print(f"✅ 车辆结算成功，共收费 {fee} 元")
            return True
        return False


def main():
    global current_mode, gate_locked, lock_timer, last_logged_plate

    # 如果你用手机摄像头，这里可能需要改成 1 或 2
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("无法打开摄像头！")
        return

    print("系统已启动。按 '1' 入场模式，按 '2' 出场模式，按 'q' 退出。")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1

        # 1. 状态机计时更新：如果道闸锁死了，检查是否到了解锁时间
        if gate_locked:
            if time.time() - lock_timer > LOCK_DURATION:
                gate_locked = False
                print("🔓 道闸已重置，等待下一辆车...")

        # 2. 画面 UI 提示
        mode_text = "ENTRY MODE" if current_mode == 1 else "EXIT MODE"
        mode_color = (0, 255, 0) if current_mode == 1 else (0, 0, 255)
        cv2.putText(frame, mode_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, mode_color, 2)

        if gate_locked:
            cv2.putText(frame, "GATE LOCKED (WAIT)", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 3. 只有在道闸没锁死的时候，才进行推理
        if not gate_locked and frame_count % 3 == 0:
            # 软件地感线圈：先看有没有车
            car_results = car_model(frame, verbose=False, classes=[2, 5, 7])
            has_car = False
            for r in car_results:
                for box in r.boxes:
                    if (box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1]) > 20000:
                        has_car = True
                        break
                if has_car: break

            if has_car:
                # 调用 HyperLPR3 识别
                results = cathode(frame)
                if results:
                    # 取置信度最高的结果
                    best_result = max(results, key=lambda x: x[1])
                    plate_number = best_result[0]
                    confidence = best_result[1]
                    box = best_result[3]

                    if confidence > 0.75:  # 提高一点点置信度要求，减少闪烁
                        x1, y1, x2, y2 = box

                        # 如果识别出了车牌，且不是刚刚才放过的那辆
                        if plate_number != last_logged_plate:
                            # 执行入库/出库
                            success = process_plate(plate_number)

                            if success:
                                # 核心动作：触发道闸，锁死！
                                gate_locked = True
                                lock_timer = time.time()
                                last_logged_plate = plate_number

                                # 画大绿框
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                cv2.putText(frame, f"PASS: {plate_number}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.9, (0, 255, 0), 2)
                        else:
                            # 虽然有车，但就是刚刚过去的那辆，画面不闪，直接画绿框即可
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, plate_number, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),
                                        2)

        cv2.imshow("Smart Parking System (Gate Simulator)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            current_mode = 1
            print("已切换为：入场模式")
        elif key == ord('2'):
            current_mode = 2
            print("已切换为：出场模式")
            # 切换模式时重置状态
            gate_locked = False
            last_logged_plate = ""

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()