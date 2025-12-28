"""
实验4：共享单车检测
使用 torchvision 的 Faster R-CNN 模型进行目标检测 - GPU加速版本
支持使用预训练COCO模型检测自行车 (bicycle类别)
"""

import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import numpy as np
import cv2
import os
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# COCO数据集类别 (91类，包含背景)
# bicycle 的类别ID是 2
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# 目标类别ID (我们关心的类别)
BICYCLE_CLASS_ID = 2
MOTORCYCLE_CLASS_ID = 4


def load_detection_model():
    """加载 Faster R-CNN 预训练模型 (torchvision自带，无需额外安装)"""
    print("加载 Faster R-CNN 模型...")
    try:
        # 使用预训练权重
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights)
        model.to(device)
        model.eval()
        print(f"模型加载成功！设备: {device}")
        return model, weights.transforms()
    except Exception as e:
        print(f"加载失败: {e}")
        return None, None


def detect_objects(model, transform, image, conf_threshold=0.5):
    """
    使用模型检测图像中的目标
    
    参数:
        model: Faster R-CNN 模型
        transform: 图像预处理变换
        image: BGR格式的图像 (numpy array)
        conf_threshold: 置信度阈值
    
    返回:
        检测结果列表
    """
    # BGR -> RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 转换为tensor
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.to(device)
    
    # 推理
    with torch.no_grad():
        predictions = model([image_tensor])
    
    pred = predictions[0]
    boxes = pred['boxes'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    
    # 筛选结果
    detections = []
    for box, label, score in zip(boxes, labels, scores):
        if score >= conf_threshold:
            detections.append({
                'box': box,  # [x1, y1, x2, y2]
                'label': label,
                'class_name': COCO_CLASSES[label] if label < len(COCO_CLASSES) else 'unknown',
                'score': score
            })
    
    return detections


def filter_bicycles(detections):
    """筛选自行车和摩托车类别"""
    bicycle_detections = []
    for det in detections:
        if det['label'] in [BICYCLE_CLASS_ID, MOTORCYCLE_CLASS_ID]:
            bicycle_detections.append(det)
    return bicycle_detections


def draw_detections(image, detections, color=(0, 255, 0), thickness=2):
    """在图像上绘制检测框和标签"""
    result = image.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det['box'].astype(int)
        score = det['score']
        class_name = det['class_name']
        
        # 绘制边界框
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
        
        # 绘制标签背景
        label = f"{class_name}: {score:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(result, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0] + 5, y1), color, -1)
        
        # 绘制标签文字
        cv2.putText(result, label, (x1 + 2, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return result


def process_single_image(model, transform, image_path, output_dir, conf_threshold=0.5):
    """处理单张图像"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法加载图像: {image_path}")
        return None
    
    print(f"处理图像: {image_path}")
    print(f"图像尺寸: {image.shape}")
    
    # 检测所有目标
    all_detections = detect_objects(model, transform, image, conf_threshold)
    print(f"检测到 {len(all_detections)} 个目标")
    
    # 筛选自行车
    bicycle_detections = filter_bicycles(all_detections)
    print(f"其中自行车/摩托车: {len(bicycle_detections)} 辆")
    
    # 绘制所有检测结果 (蓝色)
    result_all = draw_detections(image, all_detections, color=(255, 0, 0))
    cv2.imwrite(os.path.join(output_dir, "all_detections.png"), result_all)
    
    # 绘制自行车检测结果 (绿色)
    result_bicycle = draw_detections(image, bicycle_detections, color=(0, 255, 0), thickness=3)
    cv2.imwrite(os.path.join(output_dir, "bicycle_detection.png"), result_bicycle)
    
    # 打印详细信息
    for i, det in enumerate(bicycle_detections):
        x1, y1, x2, y2 = det['box'].astype(int)
        print(f"  {det['class_name']} {i+1}: 位置({x1},{y1}) 到 ({x2},{y2}), 置信度: {det['score']:.2%}")
    
    return bicycle_detections


def batch_process(model, transform, image_dir, output_dir, conf_threshold=0.5):
    """批量处理目录中的所有图像"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(image_dir).glob(f'*{ext}'))
        image_files.extend(Path(image_dir).glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"在 {image_dir} 中未找到图像文件")
        return []
    
    results_summary = []
    for img_path in image_files:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
            
        print(f"\n处理: {img_path.name}")
        
        all_detections = detect_objects(model, transform, image, conf_threshold)
        bicycle_detections = filter_bicycles(all_detections)
        
        # 保存结果
        result_image = draw_detections(image, bicycle_detections, color=(0, 255, 0), thickness=3)
        output_path = os.path.join(output_dir, f"detected_{img_path.name}")
        cv2.imwrite(output_path, result_image)
        
        results_summary.append({
            'image': img_path.name,
            'total_objects': len(all_detections),
            'bicycle_count': len(bicycle_detections)
        })
        
        print(f"  检测到 {len(bicycle_detections)} 辆自行车/摩托车")
    
    return results_summary


def main():
    OUTPUT_DIR = "exp4_output"
    IMAGE_PATH = "bike_image.jpg"  # 单张图像
    IMAGE_DIR = "bike_images"       # 批量处理目录
    CONF_THRESHOLD = 0.5  # 置信度阈值
    
    print("=" * 50)
    print("实验4：共享单车检测 (Faster R-CNN)")
    print(f"设备: {device}")
    print("=" * 50)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载模型
    model, transform = load_detection_model()
    
    if model is None:
        print("模型加载失败，请检查PyTorch和torchvision安装")
        return
    
    # 处理单张图像
    if os.path.exists(IMAGE_PATH):
        print(f"\n{'='*50}")
        print("处理单张图像")
        print('='*50)
        process_single_image(model, transform, IMAGE_PATH, OUTPUT_DIR, CONF_THRESHOLD)
    else:
        print(f"\n未找到 {IMAGE_PATH}，请将共享单车照片放在当前目录")
    
    # 批量处理
    if os.path.exists(IMAGE_DIR):
        print(f"\n{'='*50}")
        print(f"批量处理目录: {IMAGE_DIR}")
        print('='*50)
        results = batch_process(model, transform, IMAGE_DIR, OUTPUT_DIR, CONF_THRESHOLD)
        print(f"\n共处理 {len(results)} 张图像")
        
        # 汇总统计
        total_bicycles = sum(r['bicycle_count'] for r in results)
        print(f"共检测到 {total_bicycles} 辆自行车/摩托车")
    
    print("\n" + "=" * 50)
    print("实验4完成！")
    print(f"结果保存在: {OUTPUT_DIR}/")
    print("=" * 50)


if __name__ == "__main__":
    main()
