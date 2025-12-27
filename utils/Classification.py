import cv2
import numpy as np
import pandas as pd
from Predict import predict_image, classes
from skimage import color
from pathlib import Path

# ========== 参数 ==========
DEBUG = True
PLATE_LAYOUT = {
    "1": (1, 1),
    "6": (2, 3),
    "12": (3, 4),
    "24": (4, 6),
    "48": (6, 8),
    "96": (8, 12),
    "384": (16, 24)
}

# ========== 通用函数 ==========
def rgb_to_lab(rgb_uint8):
    rgb = np.clip(rgb_uint8.astype(np.float32) / 255.0, 0, 1)
    return color.rgb2lab(rgb)

def deltaE2000(lab1, lab2):
    return color.deltaE_ciede2000(lab1, lab2)

def nearest_color_name(rgb):
    css3 = {
        "red": (255, 0, 0), "green": (0, 128, 0), "blue": (0, 0, 255),
        "yellow": (255, 255, 0), "orange": (255, 165, 0),
        "purple": (128, 0, 128), "brown": (165, 42, 42),
        "pink": (255, 192, 203), "gray": (128, 128, 128),
        "black": (0, 0, 0), "white": (255, 255, 255),
    }
    lab_target = rgb_to_lab(np.array([[rgb]], dtype=np.uint8)/255.0).reshape(3,)
    best, best_name = 1e9, None
    for name, rgb_ref in css3.items():
        lab = rgb_to_lab(np.array([[rgb_ref]], dtype=np.uint8)/255.0).reshape(3,)
        d = deltaE2000(lab_target[None, :], lab[None, :])[0]
        if d < best:
            best, best_name = d, name
    return best_name

def detect_wells_hough(img_gray, dp=1.2, min_dist=20, p1=120, p2=30, min_r=8, max_r=80):
    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, dp, minDist=min_dist,
                               param1=p1, param2=p2, minRadius=min_r, maxRadius=max_r)
    if circles is None:
        return []
    return np.round(circles[0, :]).astype(int)

def robust_well_color(img_bgr, cx, cy, r):
    h, w = img_bgr.shape[:2]
    Y, X = np.ogrid[:h, :w]
    mask = (X - cx)**2 + (Y - cy)**2 <= int(0.85*r)**2
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]
    valid = (mask) & (v < np.percentile(v[mask], 98))
    if valid.sum() < 10: valid = mask
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb_valid = rgb[valid]
    return np.median(rgb_valid, axis=0).astype(np.uint8)

# ========== 主流程 ==========
def analyze_plate(img_path):
    # Step 1: 分类模型预测
    probs = predict_image(img_path)
    plate_pred = classes[int(np.argmax(probs))]
    print(f"[INFO] Predicted plate type: {plate_pred}")

    rows, cols = PLATE_LAYOUT.get(plate_pred, (0, 0))
    if rows == 0:
        print("[WARN] Unknown plate layout")
        return None

    # Step 2: 读取图像 + 灰度化
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # Step 3: 检测圆
    circles = detect_wells_hough(gray, dp=1.2, min_dist=gray.shape[0]//rows//2,
                                 p1=100, p2=28, min_r=10, max_r=gray.shape[1]//cols//2)
    if len(circles) == 0:
        print("[WARN] No wells detected.")
        return None

    # Step 4: 网格排序
    centers = circles[:, :2]
    centers = centers[np.argsort(centers[:, 1])]
    row_chunks = np.array_split(centers, rows)
    grid = [r[np.argsort(r[:, 0])] for r in row_chunks]

    # Step 5: 取颜色
    wells = []
    for i, row in enumerate(grid):
        for j, (cx, cy) in enumerate(row):
            r = circles[:, 2].mean()
            rgb = robust_well_color(img, int(cx), int(cy), int(r))
            cname = nearest_color_name(rgb)
            wells.append({"row": i+1, "col": j+1,
                          "RGB": tuple(rgb.tolist()), "Color": cname})
            if DEBUG:
                cv2.circle(img, (int(cx), int(cy)), int(r), (0,255,0), 2)
                cv2.putText(img, f"{i+1},{j+1}", (int(cx)-10, int(cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

    # Step 6: Debug 输出
    if DEBUG:
        debug_path = Path(img_path).stem + "_debug.jpg"
        cv2.imwrite(debug_path, img)
        print(f"[DEBUG] saved circle preview -> {debug_path}")

    df = pd.DataFrame(wells)
    return plate_pred, probs, df


# ========== 用法示例 ==========
if __name__ == "__main__":
    img_path = r"C:\Users\mercu\Desktop\ipys\ASRfinal\OT2ColorMatchingEnhanced\PlateImages\Colors\WIN_20251031_20_17_24_Pro.jpg"
    plate_pred, probs, df = analyze_plate(img_path)

    print("\n[PROBABILITIES]")
    for c, p in zip(classes, probs):
        print(f"{c:8s}: {p:.4f}")
    print("\n[WELL COLORS]")
    print(df)
