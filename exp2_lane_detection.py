import cv2
import numpy as np
import os

def select_lane_colors(bgr):
    """保留白色/黄色车道线（HLS 更稳一些）"""
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)

    # 白色：亮度高、饱和度低（S 低能压掉很多反光/彩色噪声）
    lower_white = np.array([0, 200, 0], dtype=np.uint8)
    upper_white = np.array([255, 255, 80], dtype=np.uint8)
    mask_white = cv2.inRange(hls, lower_white, upper_white)

    # 黄色：H 在 15~40，S/L 适中偏高
    lower_yellow = np.array([15, 30, 80], dtype=np.uint8)
    upper_yellow = np.array([40, 220, 255], dtype=np.uint8)
    mask_yellow = cv2.inRange(hls, lower_yellow, upper_yellow)

    mask = cv2.bitwise_or(mask_white, mask_yellow)
    return cv2.bitwise_and(bgr, bgr, mask=mask), mask

def roi_mask(binary, top_width=0.70, top_height=0.62, bottom_width=0.95):
    """
    梯形 ROI：关键是上边不要太窄（否则必然“聚中间”）
    top_width: ROI 上边宽度占比（建议 0.65~0.80）
    top_height: ROI 上边高度（越大越往下）
    bottom_width: ROI 下边宽度占比
    """
    h, w = binary.shape[:2]
    mask = np.zeros_like(binary)

    tw = top_width
    bw = bottom_width
    th = top_height

    pts = np.array([[
        (int((1-bw)*w/2), h),
        (int((1-tw)*w/2), int(h*th)),
        (int((1+tw)*w/2), int(h*th)),
        (int((1+bw)*w/2), h),
    ]], dtype=np.int32)

    cv2.fillPoly(mask, pts, 255)
    return cv2.bitwise_and(binary, mask), pts

def weighted_lane_fit(lines, img_shape):
    """把 HoughLinesP 的线段分为左右，并按长度加权拟合成两条线"""
    if lines is None:
        return None, None

    h, w = img_shape[:2]
    left, right = [], []

    for (x1, y1, x2, y2) in lines.reshape(-1, 4):
        if x1 == x2:
            continue

        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < 0.5 or abs(slope) > 4.0:
            # 太平的线（接近水平）& 太陡的线（噪声/护栏）都丢掉
            continue

        intercept = y1 - slope * x1
        length = np.hypot(x2 - x1, y2 - y1)

        # 线在图像底部的交点（用来判断左右 & 去掉中间杂线）
        x_bottom = (h - intercept) / slope

        if slope < 0 and x_bottom < w * 0.48:
            left.append((slope, intercept, length))
        elif slope > 0 and x_bottom > w * 0.52:
            right.append((slope, intercept, length))

    def merge(group):
        if len(group) == 0:
            return None
        slopes = np.array([g[0] for g in group])
        intercepts = np.array([g[1] for g in group])
        weights = np.array([g[2] for g in group])  # 长度权重
        slope = np.sum(slopes * weights) / np.sum(weights)
        intercept = np.sum(intercepts * weights) / np.sum(weights)
        return slope, intercept

    return merge(left), merge(right)

def draw_lane(img, left_lane, right_lane, y_top_ratio=0.62, color=(0, 255, 0), thickness=10):
    """画左右两条拟合直线"""
    h, w = img.shape[:2]
    overlay = np.zeros_like(img)
    y1 = h
    y2 = int(h * y_top_ratio)

    def draw_one(lane):
        if lane is None:
            return
        slope, intercept = lane
        if abs(slope) < 1e-6:
            return
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        cv2.line(overlay, (x1, y1), (x2, y2), color, thickness)

    draw_one(left_lane)
    draw_one(right_lane)
    return cv2.addWeighted(img, 0.8, overlay, 1.0, 0)

def main():
    IMAGE_PATH = "road_image.jpg"
    OUTPUT_DIR = "exp2_output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"找不到图片：{IMAGE_PATH}")

    # 1) 颜色过滤
    color_img, color_mask = select_lane_colors(img)
    cv2.imwrite(f"{OUTPUT_DIR}/1_color_mask.png", color_mask)

    # 2) 灰度 + 高斯 + Canny（用 OpenCV 的完整 Canny）
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.5)
    edges = cv2.Canny(blur, 50, 150)  # 两阈值可调
    cv2.imwrite(f"{OUTPUT_DIR}/2_edges.png", edges)

    # 3) ROI（上边加宽，避免“聚中间”）
    masked, roi_pts = roi_mask(edges, top_width=0.75, top_height=0.60, bottom_width=0.98)
    cv2.imwrite(f"{OUTPUT_DIR}/3_roi_edges.png", masked)

    # 4) HoughLinesP（线段更适合车道线后处理）
    lines = cv2.HoughLinesP(
        masked,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=40,
        maxLineGap=120
    )

    # debug：画出所有线段候选（蓝色）
    dbg = img.copy()
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            cv2.line(dbg, (x1, y1), (x2, y2), (255, 0, 0), 2)
    # ROI 轮廓
    cv2.polylines(dbg, roi_pts, isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.imwrite(f"{OUTPUT_DIR}/4_hough_segments.png", dbg)

    # 5) 左右拟合 + 绘制
    left_lane, right_lane = weighted_lane_fit(lines, img.shape)
    result = draw_lane(img, left_lane, right_lane, y_top_ratio=0.60)
    cv2.imwrite(f"{OUTPUT_DIR}/result.png", result)

    print("Done.")
    print("输出目录：", OUTPUT_DIR)
if __name__ == "__main__":
    main()
