import cv2
import numpy as np
from pathlib import Path

# 行列表（按你的分类器输出）
PLATE_LAYOUT = {
    "1": (1, 1), "6": (2, 3), "12": (3, 4), "24": (4, 6),
    "48": (6, 8), "96": (8, 12), "384": (16, 24)
}

def order_corners(pts4):
    # 将 4 点按 TL, TR, BR, BL 排序
    pts = np.array(pts4, dtype=np.float32)
    s = pts.sum(axis=1); d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]; br = pts[np.max(np.where(s==s.max()))]
    tr = pts[np.argmin(d)]; bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def largest_rotated_rect(gray):
    # OTSU 取最大轮廓的旋转矩形
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th = cv2.bitwise_not(th)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None, None
    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)  # (center, (w,h), angle)
    box = cv2.boxPoints(rect)  # 4x2
    return rect, box

def fit_plate_grid(image_path, plate_type, pad_ratio=0.08, debug=True):
    """
    image_path: 输入图片路径
    plate_type: "96" 等（来自你的分类器）
    pad_ratio : 内边距比例，默认 8%
    return:
      centers_xy: (rows*cols, 2) 原图坐标的网格中心
      radius_px : 建议半径（像素）
      debug_path: 调试图路径（若 debug=True）
    """
    rows, cols = PLATE_LAYOUT[plate_type]
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(image_path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    rect, box = largest_rotated_rect(gray)
    if rect is None:
        raise RuntimeError("Plate rectangle not found")

    box = order_corners(box)  # TL, TR, BR, BL

    # 规范平面尺寸：按板的长宽比来定，给足分辨率
    W = 2000
    # 用旋转矩形长宽比估计 H
    (w_rect, h_rect) = rect[1]
    if w_rect < 1 or h_rect < 1: h_rect = w_rect = 1
    aspect = h_rect / w_rect
    H = int(max(800, W * aspect))

    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(box, dst)
    M_inv = cv2.getPerspectiveTransform(dst, box)

    top = cv2.warpPerspective(bgr, M, (W, H))
    # 生成内部网格（留边）
    x0 = int(W*pad_ratio); x1 = int(W*(1-pad_ratio))
    y0 = int(H*pad_ratio); y1 = int(H*(1-pad_ratio))
    xs = np.linspace(x0, x1, cols)
    ys = np.linspace(y0, y1, rows)
    grid_norm = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2).astype(np.float32)

    # （可选）局部细化，减少 1-2 像素误差
    # 这里先给基础版；若需细化，可以在 top 上对每个近似中心做局部 NCC/质心微调

    # 反投影回原图
    grid_norm_h = cv2.convertPointsToHomogeneous(grid_norm).reshape(-1,3).T  # 3xN
    grid_orig_h = M_inv @ grid_norm_h
    grid_orig = (grid_orig_h[:2] / grid_orig_h[2]).T  # Nx2

    # 半径估计：相邻中心最小间距的一半（再打个折扣）
    centers = grid_orig.reshape(rows, cols, 2)
    dx = np.min(np.linalg.norm(centers[:,1:,:]-centers[:,:-1,:], axis=2))
    dy = np.min(np.linalg.norm(centers[1:,:,:]-centers[:-1,:,:], axis=2))
    radius = int(max(6, 0.45 * min(dx, dy)))

    debug_path = None
    if debug:
        dbg = bgr.copy()
        for (x,y) in grid_orig.astype(int):
            cv2.circle(dbg, (x,y), radius, (0,255,0), 2)
        # 画矩形边框
        for i in range(4):
            p1 = tuple(box[i].astype(int)); p2 = tuple(box[(i+1)%4].astype(int))
            cv2.line(dbg, p1, p2, (255,0,0), 2)
        debug_path = f"{Path(image_path).stem}_gridfit_debug.jpg"
        cv2.imwrite(debug_path, dbg)

    return grid_orig, radius, debug_path

from Predict import predict_image, classes

img = "PlateImages/48.jpg"
probs = predict_image(img)
plate_type = classes[int(np.argmax(probs))]   # e.g. "96"

centers, r, dbg = fit_plate_grid(img, plate_type, pad_ratio=0.08, debug=True)
print("centers shape:", centers.shape, "radius:", r, "debug:", dbg)
# centers 是 (rows*cols, 2) 原图坐标，按行优先（从上到下、左到右）
