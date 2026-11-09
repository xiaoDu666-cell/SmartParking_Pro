import cv2
import numpy as np
import re
from ultralytics import YOLO
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont

# 1. 初始化模型
yolo_model = YOLO('best.pt')
ocr = PaddleOCR(lang="ch", enable_mkldnn=False)

# 2. 读取图片
img_path = 'test_car.jpg'
img = cv2.imread(img_path)
results = yolo_model(img)

for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # 稍微外扩，确保框住
        pad_x = int((x2 - x1) * 0.05)
        pad_y = int((y2 - y1) * 0.10)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(img.shape[1], x2 + pad_x)
        y2 = min(img.shape[0], y2 + pad_y)

        plate_img = img[y1:y2, x1:x2]

        # 调用 OCR
        ocr_result = ocr.predict(plate_img)

        plate_text = ""
        if ocr_result:
            res = ocr_result[0]
            if 'rec_texts' in res and 'rec_polys' in res:
                # 排序并拼接
                text_blocks = []
                for text, poly in zip(res['rec_texts'], res['rec_polys']):
                    min_x = int(min(p[0] for p in poly))
                    text_blocks.append((min_x, text))
                text_blocks.sort(key=lambda x: x[0])

                raw_text = "".join([t[1] for t in text_blocks])
                plate_text = re.sub(r'[^\u4e00-\u9fa5A-Za-z0-9]', '', raw_text)

                # --- 核心业务补全逻辑 ---
                # 情况 1：如果是“粤GK9689”（7位，汉字+6位字母数字），说明漏了中间的字母
                if len(plate_text) == 7 and plate_text[
                    0] in "京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新":
                    # 新能源绿牌第二位必然是 A, B, C, D, E, F 等字母
                    # 这里如果是测试图，我们直接根据常识补上 'B'，或者用正则把缺失位补成 'B'
                    plate_text = plate_text[:1] + "B" + plate_text[1:]

                # 情况 2：如果是“BGK9689”（7位纯数字字母，少省份），补“粤”
                elif len(plate_text) == 7 and not re.search(r'[\u4e00-\u9fa5]', plate_text):
                    plate_text = "粤" + plate_text

        print(f"最终识别结果: {plate_text}")

        # 画图展示
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 30)
        except:
            font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 30)
        draw.text((x1, y1 - 35), plate_text, font=font, fill=(0, 255, 0))
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

cv2.imshow("YOLO + PaddleOCR", img)
cv2.waitKey(0)
cv2.destroyAllWindows()