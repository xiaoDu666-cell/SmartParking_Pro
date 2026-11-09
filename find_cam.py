import cv2
print("正在扫描可用摄像头...")
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ 发现摄像头索引: {i}")
        cap.release()
    else:
        print(f"❌ 索引 {i} 不可用")