from ultralytics import YOLO
import cv2

# 1. 加载 YOLOv8 官方预训练模型（第一次运行会自动下载，只有 6MB）
model = YOLO('best.pt')

# 2. 对图片进行推理
img_path = 'test_car.jpg'  # 请确保该图片在你的项目目录下
results = model(img_path)

# 3. 提取并显示结果
for result in results:
    # 将检测框绘制在图片上
    im_array = result.plot()
    cv2.imshow("Plate Detection", im_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()