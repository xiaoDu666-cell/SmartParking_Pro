import cv2
import numpy as np
from ultralytics import YOLO
import torch
import sys
import os

# 引入 LPRNet 模型构建器 (确保你的 model 文件夹和 LPRNet.py 路径正确)
from model.LPRNet import build_lprnet

# --- 1. 参数与模型加载 ---
# 根据你的实际路径修改
YOLO_WEIGHT = 'weights/best.pt'
LPRNET_WEIGHT = 'weights/Final_LPRNet_model.pth'

print("正在加载模型...")
yolo_model = YOLO(YOLO_WEIGHT)

# 初始化 LPRNet
lprnet = build_lprnet(lpr_max_len=8, phase=False, class_num=68, dropout_rate=0)
lprnet.load_state_dict(torch.load(LPRNET_WEIGHT, map_location='cpu'))
lprnet.eval()

# 字符字典（CCPD标准）
CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑', '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂',
         '湘', '粤', '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁', '新', '警', '学', 'O', 'A', 'B', 'C',
         'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0',
         '1', '2', '3', '4', '5', '6', '7', '8', '9']


def get_plate_number(img_path):
    """输入图片路径，返回车牌号和裁剪下来的车牌小图"""
    img = cv2.imread(img_path)
    if img is None:
        return "图片读取失败", None

    # 1. YOLO 检测车牌位置
    results = yolo_model(img, verbose=False)

    plates_data = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # 提取坐标并转换为整数
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])

            # 裁剪车牌区域
            plate_img = img[y1:y2, x1:x2]
            if plate_img.size == 0:
                continue

            # 2. LPRNet 识别字符
            # 预处理：缩放到 LPRNet 要求的 24x94
            img_resized = cv2.resize(plate_img, (94, 24))
            img_resized = img_resized.astype('float32')
            img_resized = (img_resized - 127.5) / 127.5  # 归一化
            img_resized = np.transpose(img_resized, (2, 0, 1))  # HWC -> CHW
            img_tensor = torch.from_numpy(img_resized).unsqueeze(0)  # 增加 batch 维度

            # 推理
            with torch.no_grad():
                prebs = lprnet(img_tensor)

            # 解码 CTC 结果
            prebs = prebs.argmax(dim=2)  # 获取最大概率索引
            prebs = prebs.view(-1).cpu().numpy()

            # 去重并去除 blank (索引0或占位符，根据你 LPRNet 实现而定，一般为 class_num-1 或空白)
            # 这里用最基础的相邻去重
            plate_str = ""
            last_char = -1
            for i in prebs:
                if i != last_char and i < len(CHARS):
                    plate_str += CHARS[i]
                    last_char = i

            plates_data.append({
                "number": plate_str,
                "conf": conf,
                "crop": plate_img
            })

    return plates_data


if __name__ == "__main__":
    # 测试一张图片
    test_img = "test_images/1.jpg"
    print(f"正在处理: {test_img}")

    results = get_plate_number(test_img)

    if isinstance(results, str):
        print(results)
    else:
        for item in results:
            print(f"✅ 识别成功: 车牌号 -> {item['number']} (置信度: {item['conf']:.2f})")
            # 把裁剪出来的车牌小图保存下来看看
            cv2.imwrite(f"plate_crop_{item['number']}.jpg", item['crop'])
            print(f"   (车牌截图已保存为 plate_crop_{item['number']}.jpg)")