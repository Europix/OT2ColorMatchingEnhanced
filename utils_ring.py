from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import Color, white

# mm to points
def mm(x):
    return x * 72.0 / 25.4

W, H = letter
pdf_path = "plate_color_ring_with_tags_safe.pdf"
c = canvas.Canvas(pdf_path, pagesize=letter)

# 96-well plate outer footprint (mm)
plate_w_mm = 127.76  # horizontal length
plate_h_mm = 85.48   # vertical width

cut_w = mm(plate_w_mm)
cut_h = mm(plate_h_mm)

# center cutout
cut_x = (W - cut_w) / 2.0
cut_y = (H - cut_h) / 2.0

# side length for color blocks & tag“托盘”
s = cut_w / 6.0

# side rectangles height (4 per side)
side_h = cut_h / 4.0

# draw inner cutout border
c.setStrokeColor(Color(0, 0, 0))
c.setLineWidth(1.5)
c.rect(cut_x, cut_y, cut_w, cut_h, stroke=1, fill=0)

# ---------------- top 6 color squares ----------------
top_colors = [
    Color(1, 0, 0),       # red
    Color(1, 0.5, 0),     # orange
    Color(1, 1, 0),       # yellow
    Color(0, 1, 0),       # green
    Color(0, 1, 1),       # cyan
    Color(0, 0, 1),       # blue
]

for i, col in enumerate(top_colors):
    c.setFillColor(col)
    x0 = cut_x + i * s
    y0 = cut_y + cut_h
    c.rect(x0, y0, s, s, stroke=0, fill=1)

# ---------------- bottom 6 color squares ----------------
bottom_colors = [
    Color(0, 1, 1),       # cyan
    Color(1, 0, 1),       # magenta
    Color(1, 1, 0),       # yellow
    Color(0, 0, 0),       # black
    Color(1, 0, 0),       # red
    Color(0, 1, 0),       # green
]

for i, col in enumerate(bottom_colors):
    c.setFillColor(col)
    x0 = cut_x + i * s
    y0 = cut_y - s
    c.rect(x0, y0, s, s, stroke=0, fill=1)

# ---------------- side grayscale bars ----------------
side_grays = [
    Color(0.2, 0.2, 0.2),
    Color(0.45, 0.45, 0.45),
    Color(0.7, 0.7, 0.7),
    Color(0.9, 0.9, 0.9),
]

for i, col in enumerate(side_grays):
    c.setFillColor(col)
    x0 = cut_x - s
    y0 = cut_y + i * side_h
    c.rect(x0, y0, s, side_h, stroke=0, fill=1)

for i, col in enumerate(side_grays):
    c.setFillColor(col)
    x0 = cut_x + cut_w
    y0 = cut_y + i * side_h
    c.rect(x0, y0, s, side_h, stroke=0, fill=1)

# ---------------- AprilTag images with extra white margin ----------------
tag_paths = [
    "tag36_11_00000.png",  # TL
    "tag36_11_00001.png",  # TR
    "tag36_11_00002.png",  # BL
    "tag36_11_00003.png",  # BR
]

# tag 实际边长比托盘略小，给外面留一圈白底
tag_scale = 0.8
tag_side = s * tag_scale
offset = (s - tag_side) / 2.0

# 先画白色托盘，再画 tag 图像（保证 tag 周围永远有白背景）
c.setFillColor(white)

# TL
tl_tray_x = cut_x - s
tl_tray_y = cut_y + cut_h
c.rect(tl_tray_x, tl_tray_y, s, s, stroke=0, fill=1)
c.drawImage(
    tag_paths[0],
    tl_tray_x + offset,
    tl_tray_y + offset,
    width=tag_side,
    height=tag_side,
    preserveAspectRatio=True,
    mask="auto",
)

# TR
tr_tray_x = cut_x + cut_w
tr_tray_y = cut_y + cut_h
c.rect(tr_tray_x, tr_tray_y, s, s, stroke=0, fill=1)
c.drawImage(
    tag_paths[1],
    tr_tray_x + offset,
    tr_tray_y + offset,
    width=tag_side,
    height=tag_side,
    preserveAspectRatio=True,
    mask="auto",
)

# BL
bl_tray_x = cut_x - s
bl_tray_y = cut_y - s
c.rect(bl_tray_x, bl_tray_y, s, s, stroke=0, fill=1)
c.drawImage(
    tag_paths[2],
    bl_tray_x + offset,
    bl_tray_y + offset,
    width=tag_side,
    height=tag_side,
    preserveAspectRatio=True,
    mask="auto",
)

# BR
br_tray_x = cut_x + cut_w
br_tray_y = cut_y - s
c.rect(br_tray_x, br_tray_y, s, s, stroke=0, fill=1)
c.drawImage(
    tag_paths[3],
    br_tray_x + offset,
    br_tray_y + offset,
    width=tag_side,
    height=tag_side,
    preserveAspectRatio=True,
    mask="auto",
)

c.showPage()
c.save()

print("Saved to", pdf_path)
