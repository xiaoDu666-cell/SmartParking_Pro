import cv2
import os
import time
import re
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont

# 创建保存素材的文件夹
SAVE_DIR = "ppt_materials"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

print("正在加载 AI 模型...")
yolo_model = YOLO('best.pt')
ocr = PaddleOCR(lang="ch", enable_mkldnn=False)


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


# 修改为你本地视频的路径
video_path = "car_video.mp4"
cap = cv2.VideoCapture(video_path)

save_count = 0
while True:
    success, frame = cap.read()
    if not success: break

    # 降低检测频率
    if int(time.time() * 10) % 3 == 0:
        results = yolo_model(frame, verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                pad_x = int((x2 - x1) * 0.08)
                pad_y = int((y2 - y1) * 0.15)
                cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                cx2, cy2 = min(frame.shape[1], x2 + pad_x), min(frame.shape[0], y2 + pad_y)

                plate_img = frame[cy1:cy2, cx1:cx2]
                if plate_img.size == 0: continue

                # 模糊度检测
                gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                if cv2.Laplacian(gray, cv2.CV_64F).var() < 40: continue

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
                        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
                        frame = put_chinese_text(frame, plate_text, (cx1, cy1 - 35), textColor=(0, 255, 0))

                        # 🌟 每隔一段保存一帧作为 PPT 素材
                        if save_count % 15 == 0:
                            cv2.imwrite(f"{SAVE_DIR}/frame_yolo_{save_count}.jpg", frame)  # 包含全车和框
                            cv2.imwrite(f"{SAVE_DIR}/frame_plate_{save_count}.jpg", plate_img)  # 只有纯车牌
                            with open(f"{SAVE_DIR}/frame_text_{save_count}.txt", "w") as f:
                                f.write(f"原始输出: {res['rec_texts']}\n最终车牌: {plate_text}")
                        save_count += 1

    cv2.imshow("Video Processing", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # 按 ESC 退出
        break

cap.release()
cv2.destroyAllWindows()
print(f"处理完毕！所有素材已存入 {SAVE_DIR} 文件夹。")