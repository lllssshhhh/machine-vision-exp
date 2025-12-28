"""
实验3：手写数字识别
基于MNIST数据集训练CNN模型，识别手写学号照片
GPU加速版本 (PyTorch)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# 配置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ===================== CNN模型定义 =====================
class DigitCNN(nn.Module):
    """用于手写数字识别的CNN模型"""
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 28->14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 14->7
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 7->3
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ===================== 训练与评估 =====================
def train_model(model, train_loader, epochs=5):
    """训练模型"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("开始训练...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        acc = 100. * correct / total
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.2f}%')
    
    return model


def evaluate_model(model, test_loader):
    """评估模型"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total


# ===================== 学号图像处理 =====================
def segment_digits(image):
    """
    从学号照片中分割出单个数字
    """
    # 转灰度
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 自适应二值化（处理光照不均）
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31, C=10
    )
    
    # 形态学处理：先开运算去噪，再闭运算连接
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digit_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # 过滤条件：面积、高度、宽高比
        if area > 100 and h > 20 and w > 5:
            digit_regions.append((x, y, w, h, binary[y:y+h, x:x+w]))
    
    # 按x坐标从左到右排序
    digit_regions.sort(key=lambda r: r[0])
    
    return digit_regions, binary


def preprocess_digit(digit_img):
    """
    将单个数字图像预处理为MNIST格式 (28x28)
    """
    h, w = digit_img.shape
    
    # 保持宽高比缩放到20x20以内
    scale = 20.0 / max(h, w)
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))
    
    resized = cv2.resize(digit_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 放入28x28画布中央
    canvas = np.zeros((28, 28), dtype=np.uint8)
    y_offset = (28 - new_h) // 2
    x_offset = (28 - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas


def recognize_student_id(model, image_path, output_dir, conf_threshold=0.5):
    """
    识别学号照片中的数字
    
    参数:
        conf_threshold: 置信度阈值，低于此值的数字会被过滤
    """
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：无法加载图像 {image_path}")
        return None, None
    
    print(f"图像尺寸: {image.shape}")
    
    # 分割数字
    digit_regions, binary = segment_digits(image)
    print(f"检测到 {len(digit_regions)} 个数字区域")
    
    if len(digit_regions) == 0:
        print("警告：未检测到数字，请检查图像质量")
        cv2.imwrite(os.path.join(output_dir, "debug_binary.png"), binary)
        return None, None
    
    # 保存调试图像
    debug_img = image.copy()
    for x, y, w, h, _ in digit_regions:
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, "debug_segments.png"), debug_img)
    cv2.imwrite(os.path.join(output_dir, "debug_binary.png"), binary)
    
    # 识别每个数字
    model.eval()
    all_results = []
    preprocessed_imgs = []
    
    with torch.no_grad():
        for x, y, w, h, digit_img in digit_regions:
            # 预处理
            processed = preprocess_digit(digit_img)
            preprocessed_imgs.append(processed)
            
            # 转换为tensor并归一化 (MNIST标准化参数)
            tensor = torch.from_numpy(processed).float().unsqueeze(0).unsqueeze(0) / 255.0
            tensor = (tensor - 0.1307) / 0.3081
            tensor = tensor.to(device)
            
            # 预测
            output = model(tensor)
            prob = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1).item()
            conf = prob[0, pred].item()
            
            all_results.append((pred, conf))
    
    # 过滤低置信度的结果
    filtered_results = [(pred, conf) for pred, conf in all_results if conf >= conf_threshold]
    filtered_imgs = [img for img, (_, conf) in zip(preprocessed_imgs, all_results) if conf >= conf_threshold]
    
    print(f"过滤后保留 {len(filtered_results)} 个有效数字 (置信度 >= {conf_threshold:.0%})")
    
    # 可视化预处理后的数字
    if filtered_imgs:
        n_digits = len(filtered_imgs)
        fig, axes = plt.subplots(1, n_digits, figsize=(n_digits * 1.5, 2))
        if n_digits == 1:
            axes = [axes]
        
        for i, (img, (pred, conf)) in enumerate(zip(filtered_imgs, filtered_results)):
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'{pred}\n{conf:.1%}', fontsize=10)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "digit_recognition.png"), dpi=150)
        plt.close()
    
    # 拼接学号 (只使用高置信度的结果)
    student_id = ''.join([str(r[0]) for r in filtered_results])
    return student_id, filtered_results


# ===================== 主程序 =====================
def main():
    # 配置
    OUTPUT_DIR = "exp3_output"
    STUDENT_ID_IMAGE = "student_id.jpg"
    MODEL_PATH = "digit_model.pth"
    EPOCHS = 5
    BATCH_SIZE = 128
    
    print("=" * 50)
    print("实验3：手写数字识别 (MNIST + CNN)")
    print(f"设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 50)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ===== 1. 加载MNIST数据集 =====
    print("\n[1/4] 加载MNIST数据集...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"训练集: {len(train_dataset)} 张图像")
    print(f"测试集: {len(test_dataset)} 张图像")
    
    # ===== 2. 模型训练/加载 =====
    print("\n[2/4] 初始化CNN模型...")
    model = DigitCNN().to(device)
    
    if os.path.exists(MODEL_PATH):
        print(f"加载已有模型: {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print("训练新模型...")
        model = train_model(model, train_loader, epochs=EPOCHS)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"模型已保存到: {MODEL_PATH}")
    
    # ===== 3. 评估模型 =====
    print("\n[3/4] 评估模型...")
    acc = evaluate_model(model, test_loader)
    print(f"MNIST测试集准确率: {acc:.2f}%")
    
    # ===== 4. 识别学号 =====
    print("\n[4/4] 识别学号照片...")
    if os.path.exists(STUDENT_ID_IMAGE):
        student_id, results = recognize_student_id(model, STUDENT_ID_IMAGE, OUTPUT_DIR)
        
        if student_id:
            print(f"\n{'='*50}")
            print(f"识别结果: {student_id}")
            print(f"{'='*50}")
            print("\n各位数字详情:")
            for i, (digit, conf) in enumerate(results):
                print(f"  第{i+1}位: {digit} (置信度: {conf:.2%})")
            
            # 保存结果
            with open(os.path.join(OUTPUT_DIR, "result.txt"), 'w', encoding='utf-8') as f:
                f.write(f"识别学号: {student_id}\n\n")
                for i, (digit, conf) in enumerate(results):
                    f.write(f"第{i+1}位: {digit} (置信度: {conf:.2%})\n")
    else:
        print(f"未找到学号照片: {STUDENT_ID_IMAGE}")
        print("请将学号照片命名为 student_id.jpg 放在当前目录")
    
    # ===== 演示：测试集样本 =====
    print("\n生成测试集演示...")
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    model.eval()
    
    with torch.no_grad():
        for i in range(10):
            img, label = test_dataset[i]
            pred = model(img.unsqueeze(0).to(device)).argmax(1).item()
            ax = axes[i // 5, i % 5]
            ax.imshow(img.squeeze().cpu(), cmap='gray')
            color = 'green' if pred == label else 'red'
            ax.set_title(f'预测:{pred} 真实:{label}', color=color)
            ax.axis('off')
    
    plt.suptitle('MNIST测试集样本识别结果', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "mnist_demo.png"), dpi=150)
    plt.close()
    
    print(f"\n实验3完成！结果保存在: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
