from ultralytics import YOLO
import cv2
from cnocr import CnOcr
import numpy as np

# 1. 初始化两个模型
print("正在加载 YOLO 检测模型...")
yolo_model = YOLO('best.pt')  # 你的车牌检测模型

print("正在加载 cnocr 识别模型...")
ocr_model = CnOcr()  # 文字识别模型

# 2. 读取图片
img_path = 'test_car.jpg'  # 确保这张图片在你的项目目录下
img = cv2.imread(img_path)

# 3. 用 YOLO 进行推理
results = yolo_model(img)

# 4. 遍历检测结果
for result in results:
    boxes = result.boxes
    for box in boxes:
        # 获取坐标（左上角和右下角）
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0].item()

        # 过滤掉置信度太低的框
        if confidence < 0.5:
            continue

        print(f"检测到车牌，置信度: {confidence:.2f}, 坐标: {x1},{y1},{x2},{y2}")

        # 5. 用 OpenCV 根据坐标裁切出车牌区域
        # 注意：为了识别更准确，可以稍微向外扩展几个像素
        plate_img = img[max(0, y1 - 5):min(img.shape[0], y2 + 5), max(0, x1 - 5):min(img.shape[1], x2 + 5)]

        # 将截取的图片保存下来，方便你放到报告里展示
        cv2.imwrite('cropped_plate.jpg', plate_img)

        # 为了视觉展示，在原图上画框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 6. 把裁切下来的图片交给 OCR 识别
        # cnocr 接收图片路径或 numpy 数组，这里直接传数组
        ocr_result = ocr_model.ocr(plate_img)

        # ocr_result 返回的是一个列表，里面包含识别出的文字
        print("原始识别结果:", ocr_result)

        # 提取识别出的文本
        if ocr_result:
            # 将列表中的所有文字拼接起来
            plate_text = "".join([item['text'] for item in ocr_result])
            print(f"========== 最终识别车牌号: {plate_text} ==========")

            # 将识别出的文字画在图片上
            cv2.putText(img, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

# 7. 使用 matplotlib 显示最终结果（解决 OpenCV 弹窗报错）
import matplotlib
matplotlib.use('TkAgg')  # 强制使用 Tkinter 后端弹窗
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 将 OpenCV 的 BGR 格式转成 matplotlib 需要的 RGB 格式
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 8))
plt.imshow(img_rgb)
plt.title("License Plate Detection & Recognition")
plt.axis('off') # 隐藏坐标轴
plt.show() # 弹出一个非常漂亮的图片窗口