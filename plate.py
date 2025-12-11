import os
import sys
import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from pupil_apriltags import Detector
from PIL import Image

"""
Full pipeline (normalized design coordinates):
1) CNN classify plate type from whole image, output probabilities.
2) Detect 4 AprilTags (tag36h11, ids 0/1/2/3) on the color ring.
3) Use known ring geometry (in normalized [0,1]) to compute homography:
      design space (x in [0,1], y in [0,1]) -> image pixels.
4) From tag centers compute:
   - outer plate cutout rectangle (green box)
   - inner well area rectangle with margins (cyan box)
   - uniform grid of well centers inside inner rectangle
5) Draw debug image: tags (red), outer rect (green),
   inner rect (cyan), grid (blue dots)
6) Use color-card patches (top/bottom colors + side grays) to
   fit an affine color calibration in BGR.
7) Apply calibration at each well center, print and save BGR to CSV.
"""

# ==============================================================
# CNN & transforms
# ==============================================================

CLASSES = ["nothing", "1", "6", "12", "24", "48", "96", "384"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

MODEL_PATH = "plate_model_resnet18_colorRing.pth"  # ← 改成你的模型路径
IMG_SIZE = 400

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

inference_tf = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.2)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])


def load_cnn_model(model_path: str):
    """
    Same architecture as your training script:

        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, len(classes))
        )
    """
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, len(CLASSES)),
    )
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def classify_plate(model, img_bgr):
    """
    Run CNN on a BGR image, return predicted label and probability vector.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    x = inference_tf(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    idx = int(np.argmax(probs))
    label = CLASSES[idx]
    return label, probs


# ==============================================================
# Normalized design geometry
# ==============================================================

# 设计坐标系：
#   cutout 矩形: x∈[0,1], y∈[0,1]  (0,0) = 左下角, (1,1) = 右上角
#   顶部色块: 6 等分, 每块宽 s = 1/6, 高 s
#   底部同理, 左右灰条高 1/4 一段
W_DES = 1.0
H_DES = 1.0
S_DES = W_DES / 6.0         # 顶部/底部色块宽度
SIDE_H_DES = H_DES / 4.0    # 侧面灰条每块高度

# 内缩 margin（用比例表示，更好调）
INNER_MARGIN_LEFT = 0.05
INNER_MARGIN_RIGHT = 0.05
INNER_MARGIN_TOP = 0.10
INNER_MARGIN_BOTTOM = 0.075


def get_design_tag_centers():
    """
    四个 tag 托盘中心在设计坐标系中的位置 (normalized)。
    参照我们生成 PDF 的布局：
      cutout: [0,1]x[0,1]
      TL tray: bottom-left at (-S, 1)  -> center(-S/2, 1+S/2)
      TR tray: bottom-left at ( 1, 1)  -> center(1+S/2, 1+S/2)
      BL tray: bottom-left at (-S,-S)  -> center(-S/2,-S/2)
      BR tray: bottom-left at ( 1,-S)  -> center(1+S/2,-S/2)
    """
    s = S_DES
    tl = np.array([-s / 2.0, 1.0 + s / 2.0], dtype=np.float32)
    tr = np.array([1.0 + s / 2.0, 1.0 + s / 2.0], dtype=np.float32)
    bl = np.array([-s / 2.0, -s / 2.0], dtype=np.float32)
    br = np.array([1.0 + s / 2.0, -s / 2.0], dtype=np.float32)
    return {0: tl, 1: tr, 2: bl, 3: br}


def get_design_plate_corners():
    """
    cutout 外矩形 (TL, TR, BR, BL) in design space.
    """
    tl = np.array([0.0, 1.0], dtype=np.float32)
    tr = np.array([1.0, 1.0], dtype=np.float32)
    br = np.array([1.0, 0.0], dtype=np.float32)
    bl = np.array([0.0, 0.0], dtype=np.float32)
    return np.stack([tl, tr, br, bl], axis=0)


def get_design_well_corners():
    """
    真正放孔中心的“内矩形” (扣掉四个 margin) (TL, TR, BR, BL).
    """
    x0 = INNER_MARGIN_LEFT
    x1 = 1.0 - INNER_MARGIN_RIGHT
    y0 = INNER_MARGIN_BOTTOM
    y1 = 1.0 - INNER_MARGIN_TOP

    tl = np.array([x0, y1], dtype=np.float32)
    tr = np.array([x1, y1], dtype=np.float32)
    br = np.array([x1, y0], dtype=np.float32)
    bl = np.array([x0, y0], dtype=np.float32)
    return np.stack([tl, tr, br, bl], axis=0)


def get_design_color_patches():
    """
    Color card patches in design space.
    Return list of (name, (x0,y0,x1,y1), target_bgr).
    """
    s = S_DES
    side_h = SIDE_H_DES

    patches = []

    # top 6 colors
    top_names = ["top_red", "top_orange", "top_yellow",
                 "top_green", "top_cyan", "top_blue"]
    top_colors_bgr = [
        np.array([0,   0,   255], dtype=np.float32),  # red
        np.array([0,   128, 255], dtype=np.float32),  # orange (rough)
        np.array([0,   255, 255], dtype=np.float32),  # yellow
        np.array([0,   255, 0],   dtype=np.float32),  # green
        np.array([255, 255, 0],   dtype=np.float32),  # cyan
        np.array([255, 0,   0],   dtype=np.float32),  # blue
    ]
    for i, (name, tgt) in enumerate(zip(top_names, top_colors_bgr)):
        x0 = i * s
        x1 = (i + 1) * s
        y0 = 1.0
        y1 = 1.0 + s
        patches.append((name, (x0, y0, x1, y1), tgt))

    # bottom 6 colors
    bottom_names = ["bot_cyan", "bot_magenta", "bot_yellow",
                    "bot_black", "bot_red", "bot_green"]
    bottom_colors_bgr = [
        np.array([255, 255, 0],   dtype=np.float32),  # cyan
        np.array([255, 0,   255], dtype=np.float32),  # magenta
        np.array([0,   255, 255], dtype=np.float32),  # yellow
        np.array([0,   0,   0],   dtype=np.float32),  # black
        np.array([0,   0,   255], dtype=np.float32),  # red
        np.array([0,   255, 0],   dtype=np.float32),  # green
    ]
    for i, (name, tgt) in enumerate(zip(bottom_names, bottom_colors_bgr)):
        x0 = i * s
        x1 = (i + 1) * s
        y0 = -s
        y1 = 0.0
        patches.append((name, (x0, y0, x1, y1), tgt))

    # left/right gray bars
    gray_levels = [0.2, 0.45, 0.7, 0.9]
    for j, g in enumerate(gray_levels):
        val = int(round(g * 255))
        tgt = np.array([val, val, val], dtype=np.float32)

        # left
        x0 = -s
        x1 = 0.0
        y0 = j * side_h
        y1 = (j + 1) * side_h
        patches.append((f"left_gray_{j}", (x0, y0, x1, y1), tgt))

        # right
        x0 = 1.0
        x1 = 1.0 + s
        y0 = j * side_h
        y1 = (j + 1) * side_h
        patches.append((f"right_gray_{j}", (x0, y0, x1, y1), tgt))

    return patches


# ==============================================================
# AprilTag detection + homography
# ==============================================================

def detect_tags_36h11(gray):
    """
    Detect tag36h11 on the whole grayscale image.
    Return dict: id -> center(np.array[2]) for ids 0,1,2,3.
    """
    det = Detector(
        families="tag36h11",
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=True,
        decode_sharpening=0.25,
    )
    results = det.detect(gray)
    centers = {}
    for r in results:
        tid = int(r.tag_id)
        if tid in (0, 1, 2, 3):
            centers[tid] = np.array(r.center, dtype=np.float32)
    if len(centers) < 4:
        raise RuntimeError(f"Detected only {len(centers)} tag(s) among 0/1/2/3, need all 4.")
    return centers


def compute_homography(centers_px):
    """
    centers_px: dict id->(cx,cy) in pixels.
    Return:
      H: design -> image homography
      outer_plate_corners_px: 4x2 pixels (TL,TR,BR,BL)
    """
    design_centers = get_design_tag_centers()
    src = []
    dst = []
    for tid in (0, 1, 2, 3):
        src.append(design_centers[tid])
        dst.append(centers_px[tid])
    src = np.stack(src, axis=0)
    dst = np.stack(dst, axis=0)

    H, _ = cv2.findHomography(src, dst)
    if H is None:
        raise RuntimeError("findHomography failed.")

    plate_corners_design = get_design_plate_corners()
    ones = np.ones((4, 1), dtype=np.float32)
    pts_h = np.concatenate([plate_corners_design, ones], axis=1).T
    pts_img_h = H @ pts_h
    pts_img = (pts_img_h[:2, :] / pts_img_h[2:, :]).T

    return H, pts_img


def project_points(H, points_design):
    pts = np.asarray(points_design, dtype=np.float32)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts, ones], axis=1).T
    pts_img_h = H @ pts_h
    pts_img = (pts_img_h[:2, :] / pts_img_h[2:, :]).T
    return pts_img


# ==============================================================
# Plate grid in design space
# ==============================================================

def plate_grid_design(plate_label):
    """
    Return design-space well centers for the given plate type.
    Uniform grid inside the inner rectangle (margin-clipped).
    """
    mapping = {
        "1":   (1, 1),
        "6":   (2, 3),
        "12":  (3, 4),
        "24":  (4, 6),
        "48":  (6, 8),
        "96":  (8, 12),
        "384": (16, 24),
    }
    if plate_label not in mapping:
        return np.zeros((0, 2), dtype=np.float32)

    rows, cols = mapping[plate_label]

    x0 = INNER_MARGIN_LEFT
    x1 = 1.0 - INNER_MARGIN_RIGHT
    y0 = INNER_MARGIN_BOTTOM
    y1 = 1.0 - INNER_MARGIN_TOP

    w_inner = x1 - x0
    h_inner = y1 - y0

    xs = x0 + (np.arange(cols) + 0.5) * (w_inner / cols)
    ys = y0 + (np.arange(rows) + 0.5) * (h_inner / rows)

    grid = []
    for r in range(rows):
        for c in range(cols):
            grid.append([xs[c], ys[r]])

    return np.array(grid, dtype=np.float32)


# ==============================================================
# Color calibration
# ==============================================================

def sample_patch_mean_bgr(img_bgr, H, rect_design):
    """
    rect_design: (x0,y0,x1,y1) in design space.
    Approximate patch as bounding box in image, take mean BGR.
    """
    x0, y0, x1, y1 = rect_design
    corners = np.array([
        [x0, y0],
        [x1, y0],
        [x1, y1],
        [x0, y1],
    ], dtype=np.float32)
    pts_img = project_points(H, corners)
    xs = pts_img[:, 0]
    ys = pts_img[:, 1]

    h, w, _ = img_bgr.shape
    xmin = max(0, int(np.floor(xs.min())))
    xmax = min(w - 1, int(np.ceil(xs.max())))
    ymin = max(0, int(np.floor(ys.min())))
    ymax = min(h - 1, int(np.ceil(ys.max())))

    if xmax <= xmin or ymax <= ymin:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    roi = img_bgr[ymin:ymax + 1, xmin:xmax + 1, :]
    mean_bgr = roi.reshape(-1, 3).mean(axis=0)
    return mean_bgr.astype(np.float32)


def fit_color_calibration(patches_meas, patches_target):
    """
    Fit affine color matrix W (4x3) such that:
        [b,g,r,1] @ W ≈ [b',g',r']
    """
    X = np.hstack([patches_meas,
                   np.ones((patches_meas.shape[0], 1), dtype=np.float32)])
    Y = patches_target
    W, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    return W


def apply_color_calibration(bgr, W):
    vec = np.append(bgr.astype(np.float32), 1.0)
    out = vec @ W
    return np.clip(out, 0, 255)


def calibrate_well_colors(img_bgr, H, grid_design_pts):
    """
    Use color-card patches to fit calibration, then apply to each well center.
    Return calibrated_colors (N,3) in BGR.
    """
    patches = get_design_color_patches()

    meas = []
    tgt = []
    for name, rect, tgt_bgr in patches:
        m = sample_patch_mean_bgr(img_bgr, H, rect)
        meas.append(m)
        tgt.append(tgt_bgr)
    meas = np.stack(meas, axis=0)
    tgt = np.stack(tgt, axis=0)

    W = fit_color_calibration(meas, tgt)

    wells_img = project_points(H, grid_design_pts)
    h, w, _ = img_bgr.shape
    colors = []
    for p in wells_img:
        x = int(round(p[0]))
        y = int(round(p[1]))
        if 0 <= x < w and 0 <= y < h:
            bgr = img_bgr[y, x, :].astype(np.float32)
        else:
            bgr = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        calib = apply_color_calibration(bgr, W)
        colors.append(calib)
    return np.stack(colors, axis=0)


# ==============================================================
# Debug drawing
# ==============================================================

def draw_debug(img_bgr, tag_centers_px, outer_corners_px,
               inner_corners_px, grid_pts_px, out_path):
    img = img_bgr.copy()

    # tag centers (red)
    for tid, c in tag_centers_px.items():
        cv2.circle(img, (int(c[0]), int(c[1])), 10, (0, 0, 255), 3)
        cv2.putText(img, f"id{tid}", (int(c[0]) + 5, int(c[1]) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # outer plate rectangle (green)
    pts = outer_corners_px.astype(int)
    for i in range(4):
        p1 = tuple(pts[i])
        p2 = tuple(pts[(i + 1) % 4])
        cv2.line(img, p1, p2, (0, 255, 0), 3)

    # inner well rectangle (cyan)
    pts_in = inner_corners_px.astype(int)
    for i in range(4):
        p1 = tuple(pts_in[i])
        p2 = tuple(pts_in[(i + 1) % 4])
        cv2.line(img, p1, p2, (255, 255, 0), 2)

    # grid points (blue)
    for p in grid_pts_px:
        cv2.circle(img, (int(p[0]), int(p[1])), 3, (255, 0, 0), -1)

    cv2.imwrite(out_path, img)
    print(f"[INFO] Debug image saved: {out_path}")


# ==============================================================
# Main
# ==============================================================

def main(img_path):
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Cannot find CNN model file: {MODEL_PATH}")

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    print("[INFO] Image shape:", img_bgr.shape)

    # CNN classification
    model = load_cnn_model(MODEL_PATH)
    plate_label, probs = classify_plate(model, img_bgr)
    print("[INFO] CNN plate type prediction:", plate_label)
    print("[INFO] Probabilities:")
    for cls, p in zip(CLASSES, probs):
        print(f"  {cls:>7s}: {p:.3f}")

    # AprilTag + homography
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    tag_centers_px = detect_tags_36h11(gray)
    H, plate_corners_px = compute_homography(tag_centers_px)

    # inner rect + grid in design space -> image
    well_corners_design = get_design_well_corners()
    inner_corners_px = project_points(H, well_corners_design)

    grid_design = plate_grid_design(plate_label)
    grid_pts_px = project_points(H, grid_design)

    # debug image
    debug_path = os.path.splitext(img_path)[0] + "_debug_grid.jpg"
    draw_debug(img_bgr, tag_centers_px, plate_corners_px,
               inner_corners_px, grid_pts_px, debug_path)

    # color calibration + well colors
    calib_colors = calibrate_well_colors(img_bgr, H, grid_design)

    print("\n[INFO] Calibrated well BGR (first 10):")
    for i, c in enumerate(calib_colors[:10]):
        b, g, r = c
        print(f"  well {i:3d}: BGR=({b:6.1f}, {g:6.1f}, {r:6.1f})")

    out_csv = os.path.splitext(img_path)[0] + "_well_colors.csv"
    np.savetxt(out_csv, calib_colors, delimiter=",", header="B,G,R", comments="")
    print(f"[INFO] All well colors saved to: {out_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plate.py your_image.jpg")
        sys.exit(0)
    main(sys.argv[1])
