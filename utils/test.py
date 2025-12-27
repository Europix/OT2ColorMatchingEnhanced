import sys
import cv2
from pupil_apriltags import Detector

# 你的板子原图路径
IMG_PATH = r"3.jpg"   # 改成实际路径

# 常见的 AprilTag 家族，全都扫一遍
FAMILIES = [
    "tag36h11",

]

# 一些 detector 参数组合
SCALES = [0.5, 0.75, 1.0, 1.5]
DECIMATES = [0.5, 1.0, 2.0]

def main():
    img = cv2.imread(IMG_PATH)
    if img is None:
        raise FileNotFoundError(f"读不到图像: {IMG_PATH}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    print("gray.shape:", gray.shape)

    any_hit = False

    for fam in FAMILIES:
        print(f"\n=== family = {fam} ===")
        for s in SCALES:
            if s == 1.0:
                g = gray_eq
            else:
                g = cv2.resize(
                    gray_eq, None, fx=s, fy=s,
                    interpolation=cv2.INTER_LINEAR
                )

            print(f"  scale={s} shape={g.shape}")
            for d in DECIMATES:
                det = Detector(
                    families=fam,
                    nthreads=4,
                    quad_decimate=d,
                    quad_sigma=0.0,
                    refine_edges=True,
                    decode_sharpening=0.25,
                )
                results = det.detect(g)
                n = len(results)
                print(f"    decimate={d}: Detected={n}")
                if n > 0:
                    any_hit = True
                    r0 = results[0]
                    print(f"      sample id={r0.tag_id}, center={r0.center}")

    if not any_hit:
        print("\n>>> 所有 family/scale/decimate 组合都没 detect 到 tag。")


if __name__ == "__main__":
    main()
