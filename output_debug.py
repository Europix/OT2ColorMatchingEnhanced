import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROWS = 8
COLS = 12
ROW_LABELS = list("ABCDEFGH")

def clamp01(x):
    return np.clip(x, 0.0, 1.0)

def main(csv_path, out_path="plate_96well_vis.png"):
    df = pd.read_csv(csv_path)

    # 允许两种格式：
    # 1) 只有 B,G,R 三列（你现在就是这种）
    # 2) 有 well/row/col + B,G,R（我下面给你 plate.py 会输出这种）
    if set(["B", "G", "R"]).issubset(df.columns):
        colors = df[["R", "G", "B"]].to_numpy(dtype=np.float32)  # matplotlib 用 RGB
        if len(colors) != ROWS * COLS:
            raise ValueError(f"Expected {ROWS*COLS} rows for 96-well, got {len(colors)}")
        # 关键：不要 start=1，不要额外偏移
        grid_rgb = colors.reshape(ROWS, COLS, 3)
    else:
        raise ValueError("CSV must contain columns B,G,R (and optionally well/row/col).")

    # normalize to 0..1
    grid_rgb = clamp01(grid_rgb / 255.0)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_title(f"96-well Plate Visualization\n{csv_path}", fontsize=16)

    # draw grid
    ax.set_xlim(0.5, COLS + 0.5)
    ax.set_ylim(ROWS + 0.5, 0.5)  # invert y so A is top
    ax.set_xticks(range(1, COLS + 1))
    ax.set_yticks(range(1, ROWS + 1))
    ax.set_yticklabels(ROW_LABELS)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.grid(True, linewidth=1)

    # draw wells as circles
    for r in range(ROWS):
        for c in range(COLS):
            rgb = grid_rgb[r, c]
            circ = plt.Circle((c + 1, r + 1), 0.33, color=rgb, ec="black", lw=1.2)
            ax.add_patch(circ)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"[OK] Saved: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python vis_96well_from_csv.py Fixed3_well_colors.csv [out.png]")
        sys.exit(0)
    csv_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) >= 3 else "plate_96well_vis.png"
    main(csv_path, out_path)
