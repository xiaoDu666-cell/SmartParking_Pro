import cv2
import time
import re
import os
from paddleocr import PaddleOCR
import parking_system as db
from collections import deque
import logging

logging.getLogger("ppocr").setLevel(logging.ERROR)

print("加载 PaddleOCR 模型...")
ocr_model = PaddleOCR(lang="ch",device="cpu")

PROVINCES = ['京', '津', '冀', '晋', '蒙', '辽', '吉', '黑', '沪', '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘',
             '粤', '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁', '新']
recent_plates = deque(maxlen=3)
last_action_time = 0
COOLDOWN_SECONDS = 8
current_mode = 1  # 1=入场, 2=出场（暂时写死为1）


# 提取你之前写的 clean_plate_text 和 process_plate 函数放这里
def clean_plate_text(text):
    text = re.sub(r'[^\w\u4e00-\u9fa5]', '', text).upper()
    text = text.replace('O', '0').replace('I', '1')
    if len(text) < 7 or len(text) > 8: return None
    if text[0] not in PROVINCES: text = '粤' + text[1:]
    if len(text) == 8:
        if text[2] == '0':
            text = text[:2] + 'D' + text[3:]
        elif text[2] not in ['D', 'F'] and text[-1] not in ['D', 'F']:
            text = text[:2] + 'D' + text[3:]
    match = re.search(r'[\u4e00-\u9fa5][A-Z][A-Z0-9]{5,6}', text)
    if match: return match.group(0)
    return None


def process_plate(plate_number):
    global last_action_time
    current_time = time.time()
    if current_time - last_action_time < COOLDOWN_SECONDS: return
    if db.check_in_parking(plate_number):
        print(f"⏳ [拦截] {plate_number} 已入场")
        return
    db.enter_parking(plate_number)
    last_action_time = current_time


print("【进程B】PaddleOCR 监听启动...")
last_mtime = 0

while True:
    if os.path.exists('temp_plate.jpg'):
        mtime = os.path.getmtime('temp_plate.jpg')
        # 只有当图片被更新过才去识别
        if mtime > last_mtime:
            last_mtime = mtime
            img = cv2.imread('temp_plate.jpg')
            # 图像增强
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            img_for_paddle = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

            ocr_result = ocr_model.ocr(img_for_paddle)
            if ocr_result and len(ocr_result) > 0:
                res = ocr_result[0]
                raw_text = ""
                if isinstance(res, dict) and 'rec_texts' in res:
                    raw_text = "".join(res['rec_texts'])

                if raw_text:
                    print(f"【识别原始】: {raw_text}")
                    valid_plate = clean_plate_text(raw_text)
                    if valid_plate:
                        recent_plates.append(valid_plate)
                        if len(recent_plates) >= 3 and len(set(recent_plates)) == 1:
                            process_plate(valid_plate)
    time.sleep(0.5)  # 稍微休眠，防止CPU 100%