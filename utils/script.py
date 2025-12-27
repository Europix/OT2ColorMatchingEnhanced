import shutil, random, re
from pathlib import Path
from tqdm import tqdm

src_dir = Path("PlateImages")          # 原始图片目录
dst_dir = Path("dataset_simple")        # 输出目录
train_ratio = 0.8


TOKENS = ["384", "96", "48", "24", "12", "6", "1", "nothing"]

def normalize_name(p: Path) -> str:
    s = p.stem.lower()
    s = s.replace(" ", "")
    s = s.replace("(", "_").replace(")", "_")
    s = s.replace("__", "_")
    # 去掉 lid on/off 的字样（不区分 lid）
    s = re.sub(r"_?on$", "", s)
    s = re.sub(r"_on_", "_", s)
    return s

def extract_label(p: Path) -> str:
    name = normalize_name(p)
    for tok in TOKENS:
        if tok == "nothing":
            # nothing 单独判断
            if re.search(r"(?:^|[^a-z0-9])nothing(?:[^a-z0-9]|$)", name):
                return "nothing"
        else:
            # 只匹配完整数字，不被其它数字夹带
            if re.search(rf"(?:^|[^0-9]){tok}(?:[^0-9]|$)", name):
                return tok
    return "unknown"

# 收集
images = [p for p in src_dir.glob("*.jpg")]
print(f"Found {len(images)} images")

# 分组
groups = {}
for img in images:
    label = extract_label(img)
    groups.setdefault(label, []).append(img)

# 打印一下分组统计
for k, v in groups.items():
    print(f"{k:8s}: {len(v)}")

# 拆分并复制
if dst_dir.exists():
    shutil.rmtree(dst_dir)
for label, files in groups.items():
    random.shuffle(files)
    n_train = int(len(files) * train_ratio)
    splits = [("train", files[:n_train]), ("val", files[n_train:])]
    for split, flist in splits:
        outdir = dst_dir / split / label
        outdir.mkdir(parents=True, exist_ok=True)
        for f in tqdm(flist, desc=f"{label}-{split}"):
            shutil.copy2(f, outdir / f.name)

print("✅ Done:", dst_dir)
