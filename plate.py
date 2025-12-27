import os
import sys
import argparse
import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from pupil_apriltags import Detector
from PIL import Image

# ============================================================
# Config
# ============================================================

CLASSES = ["nothing", "1", "6", "12", "24", "48", "96", "384"]
MODEL_PATH = "plate_model_resnet18_colorRing.pth"
IMG_SIZE = 400
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TAG_FAMILY = "tag36h11"
TAG_IDS = (0, 1, 2, 3)

# design geometry
S_DES = 1.0 / 6.0

# ratio margins (design space)
MARGIN_LEFT   = 0.05
MARGIN_RIGHT  = 0.05
MARGIN_TOP    = 0.085
MARGIN_BOTTOM = 0.085

# plate mapping: rows, cols
PLATE_MAP = {
    "1":   (1, 1),
    "6":   (2, 3),
    "12":  (3, 4),
    "24":  (4, 6),
    "48":  (6, 8),
    "96":  (8, 12),
    "384": (16, 24),
}

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]
inference_tf = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.2)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

# ============================================================
# CNN
# ============================================================

def load_cnn_model(model_path: str):
    """
    Must match your training head:
        model.fc = nn.Sequential(Dropout, Linear)
    """
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, len(CLASSES)),
    )
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model

def classify_plate(model, img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    x = inference_tf(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return CLASSES[idx], probs

# ============================================================
# AprilTag + homography (center-based, stable)
# ============================================================

def detect_tags_centers(gray):
    det = Detector(
        families=TAG_FAMILY,
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
        if tid in TAG_IDS:
            centers[tid] = np.array(r.center, dtype=np.float32)

    missing = [t for t in TAG_IDS if t not in centers]
    if missing:
        raise RuntimeError(f"Missing tags: {missing}. Found: {list(centers.keys())}")
    return centers

def design_tag_centers():
    s = S_DES
    return {
        0: np.array([-s/2, 1.0 + s/2], dtype=np.float32),
        1: np.array([1.0 + s/2, 1.0 + s/2], dtype=np.float32),
        2: np.array([-s/2, -s/2], dtype=np.float32),
        3: np.array([1.0 + s/2, -s/2], dtype=np.float32),
    }

def compute_homography_from_centers(tag_centers_px):
    src, dst = [], []
    des = design_tag_centers()
    for tid in TAG_IDS:
        src.append(des[tid])
        dst.append(tag_centers_px[tid])
    src = np.stack(src, axis=0)
    dst = np.stack(dst, axis=0)

    H, _ = cv2.findHomography(src, dst, method=0)
    if H is None:
        raise RuntimeError("findHomography failed with tag centers.")
    return H

def project_points(H, pts_design):
    pts = np.asarray(pts_design, dtype=np.float32)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts, ones], axis=1)      # Nx3
    out = (H @ pts_h.T).T                            # Nx3
    out = out[:, :2] / out[:, 2:3]
    return out.astype(np.float32)

# ============================================================
# Design shapes: cutout, inner, grid (dynamic by plate type)
# ============================================================

def design_cutout_corners():
    # TL, TR, BR, BL
    return np.array([[0,1],[1,1],[1,0],[0,0]], dtype=np.float32)

def design_inner_corners():
    x0 = MARGIN_LEFT
    x1 = 1.0 - MARGIN_RIGHT
    y0 = MARGIN_BOTTOM
    y1 = 1.0 - MARGIN_TOP
    return np.array([[x0,y1],[x1,y1],[x1,y0],[x0,y0]], dtype=np.float32)

def design_grid_centers(rows, cols):
    x0 = MARGIN_LEFT
    x1 = 1.0 - MARGIN_RIGHT
    y0 = MARGIN_BOTTOM
    y1 = 1.0 - MARGIN_TOP

    w = x1 - x0
    h = y1 - y0
    xs = x1 - (np.arange(cols) + 0.5) * (w / cols)
    ys = y0 + (np.arange(rows) + 0.5) * (h / rows)
    return np.array([[x, y] for y in ys for x in xs], dtype=np.float32)

# ============================================================
# Color calibration
# ============================================================

def design_color_patches():
    s = S_DES
    side_h = 1.0 / 4.0
    patches = []

    top_names = ["top_red","top_orange","top_yellow","top_green","top_cyan","top_blue"]
    top_colors_bgr = [
        np.array([0,0,255], dtype=np.float32),
        np.array([0,128,255], dtype=np.float32),
        np.array([0,255,255], dtype=np.float32),
        np.array([0,255,0], dtype=np.float32),
        np.array([255,255,0], dtype=np.float32),
        np.array([255,0,0], dtype=np.float32),
    ]
    for i,(nm,tgt) in enumerate(zip(top_names, top_colors_bgr)):
        patches.append((nm, (i*s, 1.0, (i+1)*s, 1.0+s), tgt))

    bot_names = ["bot_cyan","bot_magenta","bot_yellow","bot_black","bot_red","bot_green"]
    bot_colors_bgr = [
        np.array([255,255,0], dtype=np.float32),
        np.array([255,0,255], dtype=np.float32),
        np.array([0,255,255], dtype=np.float32),
        np.array([0,0,0], dtype=np.float32),
        np.array([0,0,255], dtype=np.float32),
        np.array([0,255,0], dtype=np.float32),
    ]
    for i,(nm,tgt) in enumerate(zip(bot_names, bot_colors_bgr)):
        patches.append((nm, (i*s, -s, (i+1)*s, 0.0), tgt))

    gray_levels = [0.2,0.45,0.7,0.9]
    for j,g in enumerate(gray_levels):
        v = int(round(g*255))
        tgt = np.array([v,v,v], dtype=np.float32)
        patches.append((f"left_gray_{j}", (-s, j*side_h, 0.0, (j+1)*side_h), tgt))
        patches.append((f"right_gray_{j}", (1.0, j*side_h, 1.0+s, (j+1)*side_h), tgt))

    return patches

def sample_patch_mean_bgr(img_bgr, H, rect_design):
    x0,y0,x1,y1 = rect_design
    corners = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], dtype=np.float32)
    pts = project_points(H, corners)
    xs, ys = pts[:,0], pts[:,1]
    h,w = img_bgr.shape[:2]
    xmin = max(0, int(np.floor(xs.min())))
    xmax = min(w-1, int(np.ceil(xs.max())))
    ymin = max(0, int(np.floor(ys.min())))
    ymax = min(h-1, int(np.ceil(ys.max())))
    if xmax <= xmin or ymax <= ymin:
        return np.array([0,0,0], dtype=np.float32)
    roi = img_bgr[ymin:ymax+1, xmin:xmax+1].reshape(-1,3).astype(np.float32)
    return roi.mean(axis=0)

def fit_color_affine(meas_bgr, tgt_bgr):
    X = np.hstack([meas_bgr, np.ones((meas_bgr.shape[0],1), dtype=np.float32)])
    Y = tgt_bgr.astype(np.float32)
    W, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    return W.astype(np.float32)

def apply_color_affine(bgr, W):
    v = np.array([bgr[0], bgr[1], bgr[2], 1.0], dtype=np.float32)
    out = v @ W
    return np.clip(out, 0, 255)

def calibrate_wells(img_bgr, H, grid_design):
    patches = design_color_patches()
    meas, tgt = [], []
    for _, rect, tgt_bgr in patches:
        meas.append(sample_patch_mean_bgr(img_bgr, H, rect))
        tgt.append(tgt_bgr)
    meas = np.stack(meas, axis=0)
    tgt  = np.stack(tgt, axis=0)
    W = fit_color_affine(meas, tgt)

    wells_px = project_points(H, grid_design)
    h,w = img_bgr.shape[:2]
    out = []
    for (x,y) in wells_px:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            bgr = img_bgr[yi, xi].astype(np.float32)
        else:
            bgr = np.array([0,0,0], dtype=np.float32)
        out.append(apply_color_affine(bgr, W))
    return np.stack(out, axis=0).astype(np.float32)

# ============================================================
# Debug drawing
# ============================================================

def draw_poly(img, pts4, color, thickness):
    pts = pts4.astype(np.int32)
    for i in range(4):
        cv2.line(img, tuple(pts[i]), tuple(pts[(i+1)%4]), color, thickness)

# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("img", help="input image path")
    ap.add_argument("--force", default="", help="force plate type: 48/96/etc (skip CNN decision)")
    args = ap.parse_args()

    img_path = args.img
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise RuntimeError(f"Cannot read image: {img_path}")
    print("[INFO] image:", img_bgr.shape)

    # Decide plate type
    plate_label = ""
    probs = None

    if args.force:
        if args.force not in PLATE_MAP:
            raise RuntimeError(f"--force must be one of {list(PLATE_MAP.keys())}")
        plate_label = args.force
        print(f"[INFO] Forced plate type: {plate_label}")
    else:
        if os.path.isfile(MODEL_PATH):
            model = load_cnn_model(MODEL_PATH)
            plate_label, probs = classify_plate(model, img_bgr)
            print("[INFO] CNN pred:", plate_label)
            for c,p in zip(CLASSES, probs):
                print(f"  {c:>7s}: {float(p):.3f}")
        else:
            raise RuntimeError(f"CNN model not found: {MODEL_PATH}. Use --force to proceed.")

    # Fallback if CNN outputs 'nothing' or invalid
    if plate_label not in PLATE_MAP:
        print(f"[WARN] CNN predicted '{plate_label}', not in {list(PLATE_MAP.keys())}. Fallback to 96.")
        plate_label = "96"

    rows, cols = PLATE_MAP[plate_label]
    print(f"[INFO] Using grid: {plate_label} => rows={rows}, cols={cols}")

    # Tags + homography
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    tag_centers_px = detect_tags_centers(gray)
    H = compute_homography_from_centers(tag_centers_px)

    # Project shapes
    cutout_px = project_points(H, design_cutout_corners())
    inner_px  = project_points(H, design_inner_corners())
    grid_des  = design_grid_centers(rows, cols)
    grid_px   = project_points(H, grid_des)

    # Debug draw
    dbg = img_bgr.copy()
    for tid,c in tag_centers_px.items():
        cv2.circle(dbg, (int(c[0]), int(c[1])), 10, (0,0,255), 3)
        cv2.putText(dbg, f"id{tid}", (int(c[0])+6, int(c[1])+6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    draw_poly(dbg, cutout_px, (0,255,0), 3)
    draw_poly(dbg, inner_px,  (255,255,0), 2)

    for (x,y) in grid_px:
        cv2.circle(dbg, (int(round(x)), int(round(y))), 3, (255,0,0), -1)

    dbg_path = os.path.splitext(img_path)[0] + "_debug.jpg"
    cv2.imwrite(dbg_path, dbg)
    print("[INFO] debug saved:", dbg_path)

    # Calibration + output
    colors_bgr = calibrate_wells(img_bgr, H, grid_des)
    out_csv = os.path.splitext(img_path)[0] + "_well_colors.csv"
    np.savetxt(out_csv, colors_bgr, delimiter=",", header="B,G,R", comments="")
    print("[INFO] colors saved:", out_csv)

if __name__ == "__main__":
    main()
