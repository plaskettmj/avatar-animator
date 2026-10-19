#!/usr/bin/env python3
"""Generate test-images/space-planet.png: an original procedural test graphic (starfield, nebula,
and a ringed planet with two cartoon eyes). Released under CC0 1.0 along with this file."""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H = 1600, 1200
rng = np.random.default_rng(7)
y, x = np.mgrid[0:H, 0:W].astype(np.float32)
# nebula: a few soft coloured blobs on deep blue
img = np.zeros((H, W, 3), np.float32) + (6, 8, 22)
for cx, cy, r, col in [(400, 300, 520, (90, 30, 140)), (1250, 850, 600, (20, 70, 150)), (900, 200, 380, (150, 40, 90))]:
    g = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * r ** 2))[..., None]
    img += g * np.array(col, np.float32) * 0.9
im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
d = ImageDraw.Draw(im)
for _ in range(900):  # stars
    sx, sy = rng.integers(0, W), rng.integers(0, H); b = int(rng.integers(140, 256)); s = rng.choice([1, 1, 1, 2])
    d.ellipse((sx - s, sy - s, sx + s, sy + s), fill=(b, b, min(255, b + 20)))
# planet with ring
cx, cy, R = 800, 620, 260
layer = Image.new("RGBA", (W, H)); ld = ImageDraw.Draw(layer)
ld.ellipse((cx - 470, cy - 95, cx + 470, cy + 95), outline=(230, 190, 120, 255), width=26)  # back ring
pl = np.zeros((H, W, 4), np.uint8)
dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
shade = np.clip(1.1 - ((x - cx + 90) ** 2 + (y - cy + 90) ** 2) ** 0.5 / (2.2 * R), 0.35, 1.0)
band = 0.85 + 0.15 * np.sin((y - cy) / 22.0)
for k, c in enumerate((235, 140, 70)):
    pl[..., k] = np.clip(c * shade * band, 0, 255)
pl[..., 3] = (dist < R) * 255
planet = Image.fromarray(pl)
layer = Image.alpha_composite(layer, planet)
ld = ImageDraw.Draw(layer)
ld.arc((cx - 470, cy - 95, cx + 470, cy + 95), 0, 180, fill=(245, 205, 135, 255), width=26)  # front ring
for ex in (cx - 85, cx + 85):  # cartoon eyes
    ld.ellipse((ex - 42, cy - 90, ex + 42, cy + 10), fill=(255, 255, 255, 255))
    ld.ellipse((ex - 20, cy - 62, ex + 20, cy - 10), fill=(25, 20, 40, 255))
im = Image.alpha_composite(im.convert("RGBA"), layer).convert("RGB").filter(ImageFilter.SMOOTH)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test-images", "space-planet.png")
im.save(out, optimize=True); print("wrote", os.path.normpath(out))
