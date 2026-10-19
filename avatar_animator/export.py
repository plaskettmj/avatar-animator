"""Encoding (GIF/WebP under a size budget), seam verification, contact sheets."""
import os, shutil, struct, subprocess, tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageSequence

LIMIT = 4.5e6

def _gif(frames, path, dur, lossy):
    ims = [Image.fromarray(f) for f in frames]
    W = ims[0].width
    idx = np.linspace(0, len(ims) - 1, 8).astype(int)
    mos = Image.new("RGB", (W * 4, W * 2))
    for j, k in enumerate(idx):
        mos.paste(ims[k], ((j % 4) * W, (j // 4) * W))
    pal = mos.quantize(256, method=Image.Quantize.MEDIANCUT)
    q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in ims]
    q[0].save(path, save_all=True, append_images=q[1:], duration=dur, loop=0, disposal=1)
    if lossy and shutil.which("gifsicle"):
        tmp = path + ".tmp.gif"
        subprocess.run(["gifsicle", "-O3", f"--lossy={lossy}", path, "-o", tmp], check=True)
        os.replace(tmp, path)

def save_gif(frames, path, fps):
    dur = int(round(1000 / fps))
    for lossy in (40, 80, 140):
        _gif(frames, path, dur, lossy)
        if os.path.getsize(path) < LIMIT:
            return dict(lossy=lossy, frames=len(frames), size=os.path.getsize(path), dropped=False)
    # still too big: shrink to 448, then 384 (sidebar shows avatars small anyway)
    for side in (448, 384):
        small = [np.asarray(Image.fromarray(f).resize((side, side), Image.LANCZOS)) for f in frames]
        for lossy in (100, 180):
            _gif(small, path, dur, lossy)
            if os.path.getsize(path) < LIMIT:
                return dict(lossy=lossy, frames=len(frames), size=os.path.getsize(path), resized=side)
    half = small[::2]   # last resort: halve the frame rate
    _gif(half, path, dur * 2, 100)
    return dict(lossy=100, frames=len(half), size=os.path.getsize(path), resized=384, dropped=True)

def save_webp(frames, path, fps):
    dur = int(round(1000 / fps))
    ims = [Image.fromarray(f) for f in frames]
    for q in (88, 75, 60, 45):
        ims[0].save(path, save_all=True, append_images=ims[1:], duration=dur, loop=0, quality=q, method=6)
        if os.path.getsize(path) < LIMIT:
            break
    return dict(quality=q, frames=len(frames), size=os.path.getsize(path))

def webp_info(path):
    d = open(path, "rb").read(); i = 12; durs = []; loop = None
    while i < len(d):
        tag = d[i:i + 4]; n = struct.unpack("<I", d[i + 4:i + 8])[0]; p = d[i + 8:i + 8 + n]
        if tag == b"ANIM": loop = struct.unpack("<H", p[4:6])[0]
        if tag == b"ANMF": durs.append(int.from_bytes(p[12:15], "little"))
        i += 8 + n + (n & 1)
    return dict(frames=len(durs), loop=loop, total_ms=sum(durs))

def check_file(path):
    im = Image.open(path)
    fs = [np.asarray(f.convert("RGB")).astype(np.float32) for f in ImageSequence.Iterator(im)]
    consec = float(np.mean([np.abs(fs[i + 1] - fs[i]).mean() for i in range(len(fs) - 1)]))
    seam = float(np.abs(fs[0] - fs[-1]).mean())
    info = dict(size_mb=round(os.path.getsize(path) / 1e6, 2), dims=im.size, frames=len(fs),
                seam_diff=round(seam, 2), avg_frame_diff=round(consec, 2),
                seam_ok=seam < 2.0 * consec + 1.0)
    if path.endswith(".gif"):
        durs = [f.info.get("duration", 0) for f in ImageSequence.Iterator(Image.open(path))]
        info.update(loop=im.info.get("loop"), total_ms=sum(durs))
    else:
        w = webp_info(path); info.update(loop=w["loop"], total_ms=w["total_ms"])
    return info

def contact_sheet(frames, path, idx, labels=None):
    W = frames[0].shape[1]; t = 192
    sheet = Image.new("RGB", (t * 3, t * 2 + 0), "black")
    for j, k in enumerate(idx[:6]):
        im = Image.fromarray(frames[k]).resize((t, t), Image.LANCZOS)
        d = ImageDraw.Draw(im); d.rectangle((0, 0, 92, 13), fill=(0, 0, 0))
        d.text((3, 1), labels[j] if labels else f"frame {k}", fill=(255, 230, 0))
        sheet.paste(im, ((j % 3) * t, (j // 3) * t))
    big = Image.new("RGB", (W * 3, W * 2))
    for j, k in enumerate(idx[:6]):
        big.paste(Image.fromarray(frames[k]), ((j % 3) * W, (j // 3) * W))
    d = ImageDraw.Draw(big)
    for j, k in enumerate(idx[:6]):
        x, y = (j % 3) * W, (j // 3) * W
        d.rectangle((x, y, x + 110, y + 16), fill=(0, 0, 0)); d.text((x + 4, y + 3), labels[j] if labels else f"frame {k}", fill=(255, 230, 0))
    big.save(path)
