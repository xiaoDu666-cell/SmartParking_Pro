import cv2
import numpy as np
import re
import sqlite3
from datetime import datetime
from ultralytics import YOLO
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont


# ================= 1. 数据库与计费模块 =================
def init_db():
    conn = sqlite3.connect('parking.db')
    cursor = conn.cursor()
    # 在场车辆表
    cursor.execute('''CREATE TABLE IF NOT EXISTS current_vehicles (
                        plate_number TEXT PRIMARY KEY, entry_time TEXT NOT NULL)''')
    # 历史流水表
    cursor.execute('''CREATE TABLE IF NOT EXISTS parking_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, plate_number TEXT NOT NULL, 
                        entry_time TEXT NOT NULL, exit_time TEXT NOT NULL, 
                        duration_hours REAL NOT NULL, fee REAL NOT NULL)''')
    conn.commit()
    return conn


def process_vehicle(conn, plate_text):
    """处理车辆进出场逻辑"""
    cursor = conn.cursor()
    cursor.execute("SELECT entry_time FROM current_vehicles WHERE plate_number = ?", (plate_text,))
    result = cursor.fetchone()

    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")

    if result:
        # --- 车辆已存在，执行【出场】 ---
        entry_time = datetime.strptime(result[0], "%Y-%m-%d %H:%M:%S")
        duration_hours = max(1, round((now - entry_time).total_seconds() / 3600))
        fee = duration_hours * 5.0  # 假设 5元/小时

        # 从在场表删除，写入流水表
        cursor.execute("DELETE FROM current_vehicles WHERE plate_number = ?", (plate_text,))
        cursor.execute('''INSERT INTO parking_records (plate_number, entry_time, exit_time, duration_hours, fee)
                          VALUES (?, ?, ?, ?, ?)''', (plate_text, result[0], now_str, duration_hours, fee))
        conn.commit()
        return f"出场成功！停车{duration_hours}小时，收费{fee}元", (0, 0, 255)  # 红字
    else:
        # --- 车辆不存在，执行【入场】 ---
        cursor.execute("INSERT INTO current_vehicles (plate_number, entry_time) VALUES (?, ?)", (plate_text, now_str))
        conn.commit()
        return f"欢迎入场！车牌：{plate_text}", (0, 255, 0)  # 绿字


# ================= 2. 视觉 AI 模型加载 =================
print("正在加载 AI 模型，请稍候...")
yolo_model = YOLO('best.pt')
ocr = PaddleOCR(lang="ch", enable_mkldnn=False)
conn = init_db()
print("系统启动成功！按 '空格键' 抓拍识别，按 'Esc' 退出。")

# 记录上一次识别结果（防止一帧内重复处理）
last_plate = ""
last_msg = "等待识别..."
last_color = (255, 255, 255)

# ================= 3. 开启摄像头主循环 =================
cap = cv2.VideoCapture(1)  # 0 代表默认摄像头

while True:
    ret, frame = cap.read()
    if not ret:
        print("摄像头读取失败！")
        break

    # 实时显示画面（不卡顿）
    display_frame = frame.copy()

    # 将当前系统提示消息写在左上角
    cv2.putText(display_frame, last_msg, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, last_color, 2)

    cv2.imshow("Smart Parking System", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # 按 Esc 退出
        break
    elif key == 32:  # 按 空格 键触发识别
        print("\n触发识别...")
        # 使用 YOLO 检测当前帧
        results = yolo_model(frame)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_img = frame[y1:y2, x1:x2]

                if plate_img.size == 0:
                    continue

                # OCR 识别
                ocr_result = ocr.predict(plate_img)
                plate_text = ""
                if ocr_result:
                    res = ocr_result[0]
                    if 'rec_texts' in res and 'rec_polys' in res:
                        text_blocks = []
                        for text, poly in zip(res['rec_texts'], res['rec_polys']):
                            min_x = int(min(p[0] for p in poly))
                            text_blocks.append((min_x, text))
                        text_blocks.sort(key=lambda x: x[0])
                        raw_text = "".join([t[1] for t in text_blocks])
                        plate_text = re.sub(r'[^\u4e00-\u9fa5A-Za-z0-9]', '', raw_text)

                        # 智能补全（针对新能源绿牌）
                        if len(plate_text) == 7 and plate_text[
                            0] in "京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新":
                            plate_text = plate_text[:1] + "B" + plate_text[1:]
                        elif len(plate_text) == 7 and not re.search(r'[\u4e00-\u9fa5]', plate_text):
                            plate_text = "粤" + plate_text

                # 只有当车牌长度大于 6 时才认为识别有效
                if len(plate_text) >= 7:
                    # 存入数据库并返回消息
                    last_msg, last_color = process_vehicle(conn, plate_text)
                    print(last_msg)
                else:
                    last_msg = "车牌不清晰，请靠近一点"
                    last_color = (0, 255, 255)  # 黄字

# 清理资源
cap.release()
cv2.destroyAllWindows()
conn.close()