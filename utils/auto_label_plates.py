import cv2
import csv
import os
from pathlib import Path

# ================= CONFIG =================
IMG_DIR = Path("PlateImages")
OUT_CSV = Path("dataset_labels.csv")
DEBUG_DIR = IMG_DIR / "debug"
DEBUG_DIR.mkdir(exist_ok=True)
# ==========================================

# 键盘输入映射
plate_map = {
    "0": "0",
    "1": "1",
    "2": "6",
    "3": "12",
    "4": "24",
    "5": "48",
    "6": "96",
    "7": "384"
}

points = []
current_plate_type = None

def order_points(pts):
    pts = sorted(pts, key=lambda x: x[1])  # sort by y
    top = sorted(pts[:2], key=lambda x: x[0])
    bottom = sorted(pts[2:], key=lambda x: x[0])
    return [top[0], top[1], bottom[1], bottom[0]]  # TL, TR, BR, BL

def mouse_click(event, x, y, flags, param):
    global points, img_copy
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            cv2.circle(img_copy, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(img_copy, str(len(points)), (x+10, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.imshow("Label", img_copy)

def draw_rect(img, pts):
    for i in range(4):
        p1 = tuple(map(int, pts[i]))
        p2 = tuple(map(int, pts[(i+1)%4]))
        cv2.line(img, p1, p2, (0, 255, 0), 2)

def label_images():
    global img_copy, points, current_plate_type
    images = sorted([f for f in IMG_DIR.glob("*.jpg")])
    if not images:
        print("❌ No images found in", IMG_DIR)
        return

    # 初始化 CSV
    if not OUT_CSV.exists():
        with open(OUT_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file_name", "plate_type", "x1","y1","x2","y2","x3","y3","x4","y4"])

    for img_path in images:
        print(f"\n📷 Labeling {img_path.name} ...")
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_copy = img.copy()
        points = []
        current_plate_type = None

        cv2.imshow("Label", img_copy)
        cv2.setMouseCallback("Label", mouse_click)

        while True:
            display = img_copy.copy()
            if current_plate_type:
                cv2.putText(display, f"Current: {current_plate_type}-well", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            cv2.imshow("Label", display)

            key = cv2.waitKey(1) & 0xFF

            # 数字键选择板型
            if key in range(ord('0'), ord('7') + 1):
                current_plate_type = plate_map[chr(key)]
                print(f"✅ Plate type selected: {current_plate_type}")

            elif key == ord('r'):  # reset
                img_copy = img.copy()
                points = []
                print("↩️ Reset points")

            elif key == 13 and len(points) == 4 and current_plate_type:  # Enter to save
                pts = order_points(points)
                draw_rect(img_copy, pts)
                debug_path = DEBUG_DIR / f"{img_path.stem}_debug.jpg"
                cv2.imwrite(str(debug_path), img_copy)

                with open(OUT_CSV, "a", newline="") as f:
                    writer = csv.writer(f)
                    row = [img_path.name, current_plate_type] + [v for xy in pts for v in xy]
                    writer.writerow(row)

                print(f"💾 Saved to {OUT_CSV}")
                break

            elif key == 32:  # Space skip
                print("⏭️ Skipped.")
                break

            elif key == ord('q'):  # Quit
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("\n✅ All images processed!")

if __name__ == "__main__":
    label_images()
