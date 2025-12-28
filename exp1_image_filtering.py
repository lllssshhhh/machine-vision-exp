"""
实验1：图像滤波与特征提取
使用GPU加速 (PyTorch + CuPy)

功能：
1. Sobel算子滤波
2. 自定义卷积核滤波
3. 颜色直方图计算
4. LBP纹理特征提取

注意：滤波、直方图、纹理特征均为手动实现，未调用cv2.filter2D或cv2.calcHist
"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import os

# 配置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


def load_image(image_path):
    """加载图像"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法加载图像: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


def manual_convolution_2d_gpu(image, kernel):
    """
    手动实现2D卷积 - GPU加速版本
    使用PyTorch张量操作实现，不调用现成的卷积函数
    
    参数:
        image: 输入图像 (H, W) 或 (H, W, C)
        kernel: 卷积核 (kH, kW)
    返回:
        卷积结果
    """
    # 转换为torch张量
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image.astype(np.float32)).to(device)
    if isinstance(kernel, np.ndarray):
        kernel = torch.from_numpy(kernel.astype(np.float32)).to(device)
    
    # 处理多通道图像
    if len(image.shape) == 3:
        # 对每个通道分别进行卷积
        channels = []
        for c in range(image.shape[2]):
            channel_result = manual_convolution_2d_gpu(image[:, :, c], kernel)
            channels.append(channel_result)
        return torch.stack(channels, dim=2)
    
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    # 手动填充
    padded = torch.zeros((h + 2*pad_h, w + 2*pad_w), dtype=torch.float32, device=device)
    padded[pad_h:pad_h+h, pad_w:pad_w+w] = image
    
    # 手动卷积 - 使用展开操作加速（仍是手动实现逻辑）
    output = torch.zeros((h, w), dtype=torch.float32, device=device)
    
    # 为了GPU效率，使用批量操作而非逐像素循环
    # 展开成滑动窗口视图
    windows = torch.zeros((h, w, kh, kw), dtype=torch.float32, device=device)
    for i in range(kh):
        for j in range(kw):
            windows[:, :, i, j] = padded[i:i+h, j:j+w]
    
    # 逐元素相乘后求和
    output = (windows * kernel.view(1, 1, kh, kw)).sum(dim=(2, 3))
    
    return output


def sobel_filter_gpu(image):
    """
    使用Sobel算子进行边缘检测 - GPU版本
    手动实现，不调用cv2.Sobel
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        # 手动实现RGB到灰度的转换
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    else:
        gray = image.copy()
    
    # Sobel算子
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)
    
    sobel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=np.float32)
    
    # 应用卷积
    gx = manual_convolution_2d_gpu(gray, sobel_x)
    gy = manual_convolution_2d_gpu(gray, sobel_y)
    
    # 计算梯度幅值
    magnitude = torch.sqrt(gx**2 + gy**2)
    
    # 归一化到0-255
    magnitude = magnitude - magnitude.min()
    magnitude = magnitude / magnitude.max() * 255
    
    return magnitude.cpu().numpy().astype(np.uint8)


def custom_kernel_filter_gpu(image, kernel):
    """
    使用自定义卷积核进行滤波 - GPU版本
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    else:
        gray = image.copy()
    
    # 应用卷积
    result = manual_convolution_2d_gpu(gray, kernel)
    
    # 归一化
    result = result.cpu().numpy()
    result = np.abs(result)
    result = result / result.max() * 255
    
    return result.astype(np.uint8)


def compute_color_histogram_gpu(image, bins=256):
    """
    手动计算颜色直方图 - GPU加速版本
    不调用cv2.calcHist
    
    参数:
        image: RGB图像 (H, W, 3)
        bins: 直方图的bin数量
    返回:
        三个通道的直方图
    """
    image_tensor = torch.from_numpy(image.astype(np.float32)).to(device)
    
    histograms = []
    for c in range(3):
        channel = image_tensor[:, :, c].flatten()
        
        # 手动计算直方图
        hist = torch.zeros(bins, dtype=torch.float32, device=device)
        
        # 将像素值映射到bin索引
        indices = (channel / 256 * bins).long()
        indices = torch.clamp(indices, 0, bins - 1)
        
        # 使用scatter_add进行计数
        ones = torch.ones_like(indices, dtype=torch.float32)
        hist.scatter_add_(0, indices, ones)
        
        histograms.append(hist.cpu().numpy())
    
    return histograms


def compute_lbp_texture_gpu(image, radius=1, n_points=8):
    """
    计算LBP(局部二值模式)纹理特征 - GPU加速版本
    手动实现
    
    参数:
        image: 输入图像
        radius: LBP半径
        n_points: 采样点数量
    返回:
        LBP特征图和直方图
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2])
    else:
        gray = image.astype(np.float32)
    
    gray_tensor = torch.from_numpy(gray).to(device)
    h, w = gray_tensor.shape
    
    # 计算采样点的相对位置
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    sample_points = []
    for angle in angles:
        dy = -radius * np.cos(angle)
        dx = radius * np.sin(angle)
        sample_points.append((dy, dx))
    
    # 填充图像
    padded = torch.zeros((h + 2*radius, w + 2*radius), dtype=torch.float32, device=device)
    padded[radius:radius+h, radius:radius+w] = gray_tensor
    
    # 计算LBP
    lbp = torch.zeros((h, w), dtype=torch.float32, device=device)
    
    for i, (dy, dx) in enumerate(sample_points):
        # 双线性插值获取邻域值
        y_floor = int(np.floor(dy)) + radius
        x_floor = int(np.floor(dx)) + radius
        y_ceil = y_floor + 1
        x_ceil = x_floor + 1
        
        # 简化处理：使用最近邻
        y_idx = radius + round(dy)
        x_idx = radius + round(dx)
        
        neighbor = padded[y_idx:y_idx+h, x_idx:x_idx+w]
        center = gray_tensor
        
        # 比较并设置位
        bit = (neighbor >= center).float()
        lbp = lbp + bit * (2 ** i)
    
    # 计算LBP直方图
    lbp_flat = lbp.flatten().long()
    num_bins = 2 ** n_points
    lbp_hist = torch.zeros(num_bins, dtype=torch.float32, device=device)
    ones = torch.ones_like(lbp_flat, dtype=torch.float32)
    lbp_hist.scatter_add_(0, lbp_flat, ones)
    
    # 归一化
    lbp_hist = lbp_hist / lbp_hist.sum()
    
    return lbp.cpu().numpy(), lbp_hist.cpu().numpy()


def visualize_results(original, sobel_result, custom_result, histograms, lbp_image, save_dir):
    """可视化并保存结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建大图
    fig = plt.figure(figsize=(16, 12))
    
    # 原图
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(original)
    ax1.set_title('原始图像', fontsize=12)
    ax1.axis('off')
    
    # Sobel结果
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(sobel_result, cmap='gray')
    ax2.set_title('Sobel边缘检测', fontsize=12)
    ax2.axis('off')
    
    # 自定义卷积核结果
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(custom_result, cmap='gray')
    ax3.set_title('自定义卷积核结果', fontsize=12)
    ax3.axis('off')
    
    # 颜色直方图
    ax4 = fig.add_subplot(2, 3, 4)
    colors = ['red', 'green', 'blue']
    labels = ['R通道', 'G通道', 'B通道']
    for hist, color, label in zip(histograms, colors, labels):
        ax4.plot(hist, color=color, label=label, alpha=0.7)
    ax4.set_title('颜色直方图', fontsize=12)
    ax4.set_xlabel('像素值')
    ax4.set_ylabel('频次')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # LBP纹理特征图
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(lbp_image, cmap='gray')
    ax5.set_title('LBP纹理特征图', fontsize=12)
    ax5.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'experiment1_results.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"结果图像已保存到: {os.path.join(save_dir, 'experiment1_results.png')}")


def main():
    # ========== 配置参数 ==========
    # 请修改为你自己的图像路径
    IMAGE_PATH = "input_image.jpg"  # 输入图像路径
    OUTPUT_DIR = "exp1_output"       # 输出目录
    
    # 给定的卷积核（用于检测垂直边缘）
    CUSTOM_KERNEL = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ], dtype=np.float32)
    
    # ========== 执行实验 ==========
    print("=" * 50)
    print("实验1：图像滤波与特征提取")
    print("=" * 50)
    
    # 检查GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("警告：未检测到GPU，将使用CPU运行")
    
    # 加载图像
    print("\n[1/5] 加载图像...")
    try:
        image = load_image(IMAGE_PATH)
        print(f"图像尺寸: {image.shape}")
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请将输入图像放置在当前目录下，或修改IMAGE_PATH变量")
        # 创建示例图像用于演示
        print("\n创建示例图像进行演示...")
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Sobel滤波
    print("[2/5] 执行Sobel边缘检测...")
    sobel_result = sobel_filter_gpu(image)
    
    # 自定义卷积核滤波
    print("[3/5] 执行自定义卷积核滤波...")
    custom_result = custom_kernel_filter_gpu(image, CUSTOM_KERNEL)
    
    # 颜色直方图
    print("[4/5] 计算颜色直方图...")
    histograms = compute_color_histogram_gpu(image)
    
    # LBP纹理特征
    print("[5/5] 提取LBP纹理特征...")
    lbp_image, lbp_hist = compute_lbp_texture_gpu(image)
    
    # 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 保存处理后的图像
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'sobel_result.png'), sobel_result)
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'custom_kernel_result.png'), custom_result)
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'lbp_result.png'), (lbp_image / lbp_image.max() * 255).astype(np.uint8))
    
    # 保存纹理特征为npy格式
    texture_features = {
        'lbp_image': lbp_image,
        'lbp_histogram': lbp_hist
    }
    np.save(os.path.join(OUTPUT_DIR, 'texture_features.npy'), texture_features)
    print(f"\n纹理特征已保存到: {os.path.join(OUTPUT_DIR, 'texture_features.npy')}")
    
    # 可视化结果
    print("\n生成结果可视化...")
    visualize_results(image, sobel_result, custom_result, histograms, lbp_image, OUTPUT_DIR)
    
    print("\n" + "=" * 50)
    print("实验1完成！")
    print("=" * 50)
    print(f"\n输出文件:")
    print(f"  - {os.path.join(OUTPUT_DIR, 'sobel_result.png')}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'custom_kernel_result.png')}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'lbp_result.png')}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'texture_features.npy')}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'experiment1_results.png')}")


if __name__ == "__main__":
    main()
