import cv2
import os
import re
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont

SAVE_DIR = "ppt_materials"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

print("正在加载 AI 模型...")
yolo_model = YOLO('best.pt')
ocr = PaddleOCR(lang="ch", enable_mkldnn=False)
print("模型加载完毕！")


def put_chinese_text(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    try:
        fontStyle = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", textSize, encoding="utf-8")
    except:
        fontStyle = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", textSize, encoding="utf-8")
    draw.text(position, text, textColor, font=fontStyle)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


video_path = "test_images/test_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"❌ 错误：无法打开视频文件 {video_path}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0 or fps > 60: fps = 25
delay = int(1000 / fps)

# 读取第一帧用于手动框选
ret, first_frame = cap.read()
if not ret: exit()

print("请用鼠标在第一帧画面中框出车牌（框选后按空格/回车确认）...")
# 让用户手动框选车牌
roi = cv2.selectROI("Select Plate ROI (Press Enter/Space to confirm)", first_frame, showCrosshair=True,
                    fromCenter=False)
cv2.destroyWindow("Select Plate ROI (Press Enter/Space to confirm)")

# 解包手动框选的坐标
tx, ty, tw, th = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
print(f"已锁定坐标: X={tx}, Y={ty}, 宽={tw}, 高={th}")

last_plate_text = ""

# 重新读取视频开头
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_count = 0

while True:
    success, frame = cap.read()
    if not success: break

    # 每一帧都在你框定的区域内进行裁剪识别
    # 加入一点边界外扩，确保车牌边缘不被切掉
    pad = 10
    cx1 = max(0, tx - pad)
    cy1 = max(0, ty - pad)
    cx2 = min(frame.shape[1], tx + tw + pad)
    cy2 = min(frame.shape[0], ty + th + pad)

    plate_img = frame[cy1:cy2, cx1:cx2]

    # 每隔 3 帧识别一次，保证流畅
    if frame_count % 3 == 0 and plate_img.size > 0:
        plate_img_large = cv2.resize(plate_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        ocr_result = ocr.predict(plate_img_large)

        if ocr_result and 'rec_texts' in ocr_result[0]:
            res = ocr_result[0]
            text_blocks = []
            if 'rec_polys' in res:
                for text, poly in zip(res['rec_texts'], res['rec_polys']):
                    min_x = int(min(p[0] for p in poly))
                    text_blocks.append((min_x, text))
                text_blocks.sort(key=lambda x: x[0])
                raw_text = "".join([t[1] for t in text_blocks])
            else:
                raw_text = "".join(res['rec_texts'])

            plate_text = re.sub(r'[^\u4e00-\u9fa5A-Za-z0-9]', '', raw_text)
            if len(plate_text) >= 7:
                last_plate_text = plate_text
                print(f"[帧 {frame_count}] 识别: {last_plate_text}")

    # 画图展示
    if last_plate_text != "":
        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
        frame = put_chinese_text(frame, last_plate_text, (cx1, cy1 - 35), textColor=(0, 255, 0))

        # 保存 PPT 素材
        if frame_count % 30 == 0:
            cv2.imwrite(f"{SAVE_DIR}/video_frame_{frame_count}.jpg", frame)

    cv2.imshow("Plate Recognition Video", frame)
    if cv2.waitKey(delay) & 0xFF == 27:
        break
    frame_count += 1

cap.release()
cv2.destroyAllWindows()
print("处理完毕！")