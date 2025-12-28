# 机器视觉实验 - 依赖安装说明

## 必需的Python包

```bash
# 基础依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 图像处理
pip install numpy opencv-python matplotlib

# 实验4需要的额外依赖
pip install pandas
```

## 验证GPU是否可用

```python
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

## 各实验使用说明

### 实验1：图像滤波与特征提取
1. 将自拍图像命名为 `input_image.jpg` 放在当前目录
2. 运行 `python exp1_image_filtering.py`
3. 输出在 `exp1_output/` 目录

### 实验2：车道线检测
1. 将道路图像命名为 `road_image.jpg` 放在当前目录
2. 运行 `python exp2_lane_detection.py`
3. 输出在 `exp2_output/` 目录

### 实验3：手写数字识别
1. 将学号照片命名为 `student_id.jpg` 放在当前目录
2. 运行 `python exp3_digit_recognition.py`
3. 首次运行会下载MNIST数据集并训练模型
4. 输出在 `exp3_output/` 目录

### 实验4：共享单车检测
1. 将共享单车照片命名为 `bike_image.jpg` 放在当前目录
2. 运行 `python exp4_bike_detection.py`
3. 首次运行会下载YOLOv5模型
4. 输出在 `exp4_output/` 目录
