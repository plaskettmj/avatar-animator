#!/usr/bin/env python3
"""Avatar animator CLI: any image -> seamless looping animated avatar (GIF + WebP).

Example:
  python animate.py photo.jpg --style portrait -o out/portrait
  python animate.py photo.jpg --style portrait -o out/portrait --prepare-only   # slow step only: crop + eyes
  python animate.py photo.jpg --style portrait -o out/portrait --eyes "0.40,0.42,0.08,0.05;0.58,0.42,0.08,0.05"
    (re-export with the same -o reuses out/portrait/prep/, so segmentation is skipped)
  python animate.py mascot.png -o out/m --eyes-at "D5,G5"            # chat path: cells of out/m/prep/grid.png
  python animate.py mascot.png -o out/m --eyes-src "365,470;605,470"  # clicked points in source px (shape is grabbed)
"""
import argparse, json, os, re, sys, time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from avatar_animator.core import (Analysis, Animator, STYLES, SUBJECT_MOTIONS, BG_MOTIONS, DEFAULTS, EYE_KINDS,
                                  parse_eye_spec, format_eye_specs)

MODELS = ["birefnet-general-lite", "birefnet-general", "u2net", "isnet-general-use"]
GRID = 10   # grid.png: columns A-J, rows 1-10 over the visible square
from avatar_animator import export as export_mod

def parse_list(s, allowed):
    if s is None: return None
    if s.strip().lower() in ("", "none"): return []
    items = [x.strip() for x in s.split(",") if x.strip()]
    bad = [x for x in items if x not in allowed]
    if bad: raise SystemExit(f"unknown motion(s) {bad}; choose from {allowed}")
    return items

def _load(path):
    return np.asarray(Image.open(path).convert("RGB"))

def _analysis(src, outdir, cache, log):
    """Slow analysis for this image, cached in <outdir>/prep/ unless cache=False."""
    return Analysis(src, cache_dir=os.path.join(outdir, "prep") if cache else None, log=log)

def eyes_arg(boxes):
    """Format eye boxes for --eyes: 'cx,cy,w,h;cx,cy,w,h'."""
    return ";".join(",".join(f"{v:.4f}" for v in b) for b in boxes)

def _split(s):
    if s is None: return None
    if s.strip().lower() in ("", "none"): return []
    return [e.strip() for e in s.split(";") if e.strip()]

def parse_eyes(s):
    """Parse --eyes 'cx,cy,w,h[,kind];...' (fractions of the output square) into tuples. '' or 'none' -> []."""
    items = _split(s)
    if items is None: return None
    eyes = []
    for e in items:
        toks = [t.strip() for t in e.replace(":", ",").split(",") if t.strip()]
        kind = toks.pop().lower() if toks and toks[-1].lower() in EYE_KINDS else None
        try:
            v = tuple(map(float, toks))
        except ValueError:
            raise SystemExit(f"--eyes needs cx,cy,w,h per eye (plus an optional kind {EYE_KINDS}), got {e!r}")
        if len(v) != 4: raise SystemExit(f"--eyes needs cx,cy,w,h per eye, got {e!r}")
        eyes.append(v + (kind,) if kind else v)
    return eyes

def parse_eyes_src(s):
    """Parse --eyes-src 'x,y[,w,h][,kind];...' (source-image px) into eye specs. Without w,h the eye's shape
    is found automatically around the point."""
    items = _split(s)
    if items is None: return None
    try:
        return [parse_eye_spec(e, "src") for e in items]
    except ValueError as err:
        raise SystemExit(f"--eyes-src: {err}")

def parse_cells(s, n=GRID):
    """Parse --eyes-at 'D5,G5' (cells of prep/grid.png, optional ':cartoon' / ':photo' each) into eye specs
    at the cell centres, in output fractions. The eye shape is then found around each point."""
    if s is None: return None
    if s.strip().lower() in ("", "none"): return []
    eyes = []
    for c in [c.strip() for c in s.replace(";", ",").split(",") if c.strip()]:
        cell, _, kind = c.partition(":")
        m = re.fullmatch(r"([A-Za-z])\s*([0-9]{1,2})", cell.strip())
        kind = (kind.strip().lower() or "auto")
        if not m or kind not in EYE_KINDS:
            raise SystemExit(f"--eyes-at: use cells like D5 or D5:cartoon (columns A-{chr(64 + n)}, rows 1-{n}), got {c!r}")
        col, row = ord(m.group(1).upper()) - 65, int(m.group(2)) - 1
        if not (0 <= col < n and 0 <= row < n):
            raise SystemExit(f"--eyes-at: {c!r} is outside the grid (A-{chr(64 + n)}, 1-{n})")
        eyes.append(dict(x=(col + 0.5) / n, y=(row + 0.5) / n, w=None, h=None, kind=kind, space="out"))
    return eyes

def collect_eyes(eyes=None, eyes_src=None, eyes_at=None):
    """Merge the three manual-eye inputs (any may be a string or already parsed) into one list, or None."""
    out = []
    for val, fn in ((eyes, parse_eyes), (eyes_src, parse_eyes_src), (eyes_at, parse_cells)):
        if isinstance(val, str): val = fn(val)
        out += list(val or [])
    return out or None

def prepare(path, outdir, style="portrait", subject=None, bg=None, size=512, model="birefnet-general-lite",
            center=None, zoom=1.0, eyes=None, blink="auto", params=None, cache=True, log=print, eye_kind="auto"):
    """Slow step only: subject mask, crop, background fill and eye detection. No frames are rendered or encoded.

    Writes <outdir>/prep/: the cached analysis (analysis.json, mask-<model>.png), prep.json, subject.png
    (RGBA subject layer on the padded canvas), background.png (filled background on the same canvas) and
    preview.png (the cropped square with the eyes outlined, numbered and labelled with their blink style),
    crop.png (the clean square, for editors), grid.png (the square with a labelled A-J / 1-10 grid, for the chat
    path: --eyes-at D5,G5) and mask.png (the cut-out: everything that moves as the subject at full brightness).
    Returns the prep.json dict. Its "eyes_src_arg" (source px, survives crop changes) or "eyes_arg" (fractions
    of this crop) can be passed back as --eyes-src / --eyes after editing."""
    t0 = time.time()
    os.makedirs(outdir, exist_ok=True)
    src = _load(path)
    ana = _analysis(src, outdir, cache, log)
    ana.mask(model); ana.face(); ana.pet()   # warm every slow step, so any later export (e.g. turning blink on) is fast
    an = Animator(src, out=size, style=style, subject=subject, bg=bg, model=model, center=center, zoom=zoom,
                  eyes=eyes, blink=blink, params=params, log=log, analysis=ana, eye_kind=eye_kind)
    return _write_prep(an, path, src, outdir, model, center, zoom, blink, t0)

KIND_COLOR = {"cartoon": (0, 230, 255), "photo": (255, 230, 0)}

def _font(px):
    try:
        return ImageFont.load_default(size=px)
    except TypeError:   # Pillow < 10.1
        return ImageFont.load_default()

def draw_eyes(an, im, scale=1.0, dx=0, dy=0, labels=True):
    """Outline the eyes (ellipse of each eye's box) on a PIL image of the visible square."""
    dr = ImageDraw.Draw(im)
    out = an.out
    f = _font(max(12, int(13 * scale)))
    for i, ((cx, cy, w, h), e) in enumerate(zip(an.eye_boxes(), an.eyes), 1):
        col = KIND_COLOR.get(e["kind"], (255, 230, 0))
        x0, y0 = dx + (cx - w / 2) * out * scale, dy + (cy - h / 2) * out * scale
        x1, y1 = dx + (cx + w / 2) * out * scale, dy + (cy + h / 2) * out * scale
        dr.ellipse((x0, y0, x1, y1), outline=col, width=max(2, int(2 * scale)))
        if labels:
            dr.text((x1 + 3, y0 - 2), f"{i} {e['kind']}", fill=col, font=f, stroke_width=2, stroke_fill=(0, 0, 0))
    return im

def grid_image(an, crop_img, n=GRID, scale=1.5):
    """The visible square, upscaled, with an n x n grid labelled A.. across and 1.. down (chat eye placement)."""
    out = an.out
    S = int(out * scale); pad = int(28 * scale / 1.5)
    im = Image.new("RGB", (S + 2 * pad, S + 2 * pad), (24, 24, 28))
    im.paste(crop_img.resize((S, S), Image.LANCZOS), (pad, pad))
    dr = ImageDraw.Draw(im, "RGBA")
    f = _font(int(16 * scale / 1.5))
    for k in range(n + 1):
        p = pad + k * S / n
        dr.line([(p, pad), (p, pad + S)], fill=(255, 255, 255, 110), width=1)
        dr.line([(pad, p), (pad + S, p)], fill=(255, 255, 255, 110), width=1)
    for k in range(n):
        c = pad + (k + 0.5) * S / n
        for x, y in ((c, pad / 2), (c, pad + S + pad / 2)):
            dr.text((x, y), chr(65 + k), fill=(255, 255, 255), font=f, anchor="mm")
        for x, y in ((pad / 2, c), (pad + S + pad / 2, c)):
            dr.text((x, y), str(k + 1), fill=(255, 255, 255), font=f, anchor="mm")
    draw_eyes(an, im, scale, pad, pad)
    return im

def mask_image(an, crop_img):
    """Cut-out view: the subject (what tilts/floats/breathes) at full brightness, the rest dimmed and tinted."""
    o, out = an.off, an.out
    a = an.alpha[o:o + out, o:o + out][..., None]
    rgb = np.asarray(crop_img).astype(np.float32) / 255
    dim = rgb * 0.25 + np.array([0.35, 0.0, 0.25], np.float32) * 0.35
    f = rgb * a + dim * (1 - a)
    edge = cv2.morphologyEx((a[..., 0] > 0.5).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    f[edge] = (1.0, 0.3, 0.8)
    return Image.fromarray((np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8))

def _write_prep(an, path, src, outdir, model, center, zoom, blink, t0):
    d = os.path.join(outdir, "prep"); os.makedirs(d, exist_ok=True)
    u8 = lambda a: (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(np.dstack([u8(an.fg), u8(an.alpha)]), "RGBA").save(os.path.join(d, "subject.png"))
    Image.fromarray(u8(an.bg_fill)).save(os.path.join(d, "background.png"))
    o, out = an.off, an.out
    crop_img = Image.fromarray(u8(an.img[o:o + out, o:o + out]))
    crop_img.save(os.path.join(d, "crop.png"))
    draw_eyes(an, crop_img.copy()).save(os.path.join(d, "preview.png"))
    grid_image(an, crop_img).save(os.path.join(d, "grid.png"))
    mask_image(an, crop_img).save(os.path.join(d, "mask.png"))
    boxes = an.eye_boxes()
    H0, W0 = src.shape[:2]
    cx, cy, side = an.crop
    prep = dict(
        input=path, key=an.analysis.key, style=an.style, subject_motions=an.subj, bg_motions=an.bgm,
        size=out, model=model, blink=blink,
        crop=dict(cx=round(float(cx), 2), cy=round(float(cy), 2), side=round(float(side), 2),
                  # --center/--zoom values that reproduce this crop
                  center=[round(float(cx) / W0, 4), round(float(cy) / H0, 4)],
                  zoom=round(min(W0, H0) / float(side), 4)),
        canvas=dict(size=an.E, offset=o, out=out),   # subject/background PNGs are canvas-sized; visible square = [offset, offset+out)
        anchored=bool(an.anchored), mask_cover=round(an.mask_cover, 3),
        illustration=round(an.illustration, 3), eye_kind=an.eye_kind,
        eye_source=an.eye_source(), eyes=_eyes_json(an, boxes),
        eyes_arg=eyes_arg(boxes), eyes_src_arg=format_eye_specs(an.eye_specs()),
        grid=dict(cols="".join(chr(65 + k) for k in range(GRID)), rows=GRID,
                  usage="--eyes-at D5,G5 (cell names from grid.png; add :cartoon or :photo to force a blink style)"),
        files=dict(subject="prep/subject.png", background="prep/background.png", preview="prep/preview.png",
                   crop="prep/crop.png", grid="prep/grid.png", mask="prep/mask.png"),
        cache=dict(hits=an.analysis.hits, misses=an.analysis.misses),
        prep_s=round(time.time() - t0, 2))
    with open(os.path.join(d, "prep.json"), "w") as f:
        json.dump(prep, f, indent=2, default=str)
    return prep

def _eyes_json(an, boxes):
    """Per-eye JSON: blink style, box in output fractions, and position/size in source-image px."""
    return [dict(kind=e["kind"], box=list(b), src={k: round(float(v), 1) for k, v in sp.items() if k != "kind"},
                 grabbed=bool(e.get("grab", {}).get("found", False)))
            for e, b, sp in zip(an.eyes, boxes, an.eye_specs())]

# ----------------------------------------------------------------- live preview layers (editor step 3)
LIVE_SUBJECT = ("float", "breathe", "tilt", "blink")   # subject motions the browser preview animates itself
LIVE_BG = ("parallax",)                                 # background motions it animates itself
LIVE_PARAMS = ("float_amp", "breathe_amp", "tilt_deg", "blink_at", "blink_len", "parallax_amp")
BLINK_LEVELS = 8   # the preview closes the eyes in this many steps (the export renders every frame exactly)

def live_layers(an, d, url=""):
    """Write the layers the browser preview (preview.js) animates, into directory d, and return its config.

    background.png  the background frozen at t=0 with every motion except parallax (the preview drifts it itself)
    subject.png     the subject colour + alpha on the padded canvas (the preview tilts / breathes / floats it)
    blink.png       eye patches closing in BLINK_LEVELS steps, side by side (level k = blink amount k/BLINK_LEVELS),
                    cropped to the box around all eyes; pasted over subject.png before the subject transform
    Motions not listed in LIVE_SUBJECT / LIVE_BG (swirl, hue, twinkle, particles, sparkles, sweep) are not animated
    in the preview: they show as they look at t=0 and are reported under "frozen"."""
    os.makedirs(d, exist_ok=True)
    u8 = lambda a: (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(u8(an.background(0.0, skip=("parallax",)))).save(os.path.join(d, "background.png"))
    Image.fromarray(np.dstack([u8(an.fg), u8(an.alpha)]), "RGBA").save(os.path.join(d, "subject.png"))
    blink = None
    if an.eyes and "blink" in an.subj:
        levels = [an.apply_blink(an.fg, k / BLINK_LEVELS) for k in range(1, BLINK_LEVELS + 1)]
        base = u8(an.fg)
        changed = np.zeros(an.fg.shape[:2], bool)
        for rgb in levels:
            changed |= (u8(rgb) != base).any(2)
        if changed.any():
            ys, xs = np.nonzero(changed)
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            a = u8(an.alpha[y0:y1, x0:x1])
            strip = np.concatenate([np.dstack([u8(rgb[y0:y1, x0:x1]), a]) for rgb in levels], 1)
            Image.fromarray(strip, "RGBA").save(os.path.join(d, "blink.png"))
            blink = dict(levels=BLINK_LEVELS, box=[x0, y0, x1, y1])
    if blink is None and os.path.exists(os.path.join(d, "blink.png")):
        os.remove(os.path.join(d, "blink.png"))
    present = list(an.subj) + list(an.bgm)
    cfg = dict(
        canvas=an.E, off=an.off, out=an.out, pivot=[float(an.pivot[0]), float(an.pivot[1])],
        anchored=bool(an.anchored), has_subject=an.sbox is not None, style=an.style,
        subject_motions=list(an.subj), bg_motions=list(an.bgm),
        live=[m for m in present if m in LIVE_SUBJECT + LIVE_BG],
        frozen=[m for m in present if m not in LIVE_SUBJECT + LIVE_BG and m != "backdrop"],
        params={k: float(an.P[k]) for k in LIVE_PARAMS}, seconds=an.seconds, fps=an.fps,
        eyes=len(an.eyes), blink=blink,
        files={k: url + k + ".png" for k in ("background", "subject") + (("blink",) if blink else ())})
    with open(os.path.join(d, "live.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    return cfg

def export(path, outdir, style="portrait", subject=None, bg=None, seconds=3.0, fps=20, size=512,
           model="birefnet-general-lite", center=None, zoom=1.0, eyes=None, blink="auto", params=None,
           cache=True, log=print, eye_kind="auto"):
    """Render and encode (GIF + WebP + contact sheet + eyes.png + report.json). Reuses the cached analysis in
    <outdir>/prep/ when it matches the image, so changing eyes, motions or parameters skips segmentation.
    prep/ itself is left as prepare() wrote it (the auto-detected eyes stay there); eyes.png shows the eyes
    this export used."""
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    src = _load(path)
    ana = _analysis(src, outdir, cache, log)
    an = Animator(src, out=size, style=style, subject=subject, bg=bg, seconds=seconds, fps=fps, model=model,
                  center=center, zoom=zoom, eyes=eyes, blink=blink, params=params, log=log, analysis=ana,
                  eye_kind=eye_kind)
    t_prep = time.time() - t0
    o = an.off
    draw_eyes(an, Image.fromarray((np.clip(an.img[o:o + an.out, o:o + an.out], 0, 1) * 255 + 0.5).astype(np.uint8))
              ).save(os.path.join(outdir, "eyes.png"))
    t1 = time.time()
    frames = an.render()
    # exact periodicity check: t=1.0 must equal t=0
    end = an.frame_t(1.0) if hasattr(an, "frame_t") else None
    t_render = time.time() - t1
    base = os.path.join(outdir, "avatar")
    g = export_mod.save_gif(frames, base + ".gif", fps)
    w = export_mod.save_webp(frames, base + ".webp", fps)
    # contact sheet: evenly spaced, plus mid-blink frame if blinking
    N = len(frames)
    idx = list(np.linspace(0, N, 6, endpoint=False).astype(int))
    if an.eyes and "blink" in an.subj:
        kb = int(round((an.P["blink_at"] + an.P["blink_len"] / 2) * N)) % N
        idx[4] = kb
    idx = sorted(set(idx))
    export_mod.contact_sheet(frames, os.path.join(outdir, "contact-sheet.png"), idx,
                             [f"f{k} t={k / fps:.2f}s" + (" blink" if an.eyes and abs(k - int(round((an.P['blink_at'] + an.P['blink_len'] / 2) * N))) < 1 else "") for k in idx])
    boxes = an.eye_boxes()
    report = dict(input=path, style=style, subject_motions=an.subj, bg_motions=an.bgm,
                  model=model, eyes=[e["kind"] for e in an.eyes], eye_source=an.eye_source(), eyes_arg=eyes_arg(boxes),
                  eyes_src_arg=format_eye_specs(an.eye_specs()), eye_details=_eyes_json(an, boxes),
                  anchored=bool(an.anchored), frames=N, fps=fps,
                  params={k: v for k, v in an.P.items() if DEFAULTS.get(k) != v},
                  seconds=seconds, gif=export_mod.check_file(base + ".gif"), webp=export_mod.check_file(base + ".webp"),
                  periodic_t1_equals_t0=bool(end is not None and np.array_equal(end, frames[0])),
                  cache=dict(hits=ana.hits, misses=ana.misses),
                  prep_s=round(t_prep, 1), frames_s=round(t_render, 1), encode_s=round(time.time() - t1 - t_render, 1),
                  render_s=round(time.time() - t0, 1))
    with open(os.path.join(outdir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    return report

def run(path, outdir, style="portrait", subject=None, bg=None, seconds=3.0, fps=20, size=512,
        model="birefnet-general-lite", center=None, zoom=1.0, eyes=None, blink="auto", params=None, log=print,
        cache=True, eye_kind="auto"):
    """Full pipeline (prepare + export). Same as before; the slow analysis is cached in <outdir>/prep/."""
    return export(path, outdir, style, subject, bg, seconds, fps, size, model, center, zoom, eyes, blink, params,
                  cache=cache, log=log, eye_kind=eye_kind)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("-o", "--out", default="out")
    ap.add_argument("--style", default="portrait", choices=list(STYLES))
    ap.add_argument("--subject", help=f"override subject motions, comma list of {SUBJECT_MOTIONS} or 'none'")
    ap.add_argument("--bg", help=f"override background motions, comma list of {BG_MOTIONS} or 'none'")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--model", default="birefnet-general-lite", choices=MODELS,
                    help="cut-out model; birefnet-general is slower (~30 s, ~8 GB RAM peak) but keeps headsets, props and bodies")
    ap.add_argument("--center", help="crop center as fractions x,y of the source image (e.g. 0.43,0.47)")
    ap.add_argument("--zoom", type=float, default=1.0, help=">1 crops tighter")
    ap.add_argument("--eyes", help="manual eyes in output fractions 'cx,cy,w,h[,kind];...' (tied to this crop)")
    ap.add_argument("--eyes-src", help="manual eyes in source-image px 'x,y[,w,h][,kind];...'; without w,h the eye "
                                       "shape is found around the point. Survives crop/zoom/style changes")
    ap.add_argument("--eyes-at", help="manual eyes as cells of prep/grid.png, e.g. 'D5,G5' or 'D5:cartoon,G5:cartoon'")
    ap.add_argument("--eye-kind", default="auto", choices=list(EYE_KINDS),
                    help="blink style for eyes left on auto: cartoon (eye shape squashes shut) or photo (eyelid "
                         "slides down). auto picks cartoon for flat artwork / logo / space styles, photo otherwise")
    ap.add_argument("--blink", default="auto", choices=["auto", "face", "pet", "cartoon", "off"])
    ap.add_argument("--prepare-only", action="store_true",
                    help="run only the slow step (mask, crop, eyes); write <out>/prep/ and print prep.json")
    ap.add_argument("--no-cache", action="store_true", help="don't read or write the <out>/prep/ cache")
    ap.add_argument("--set", action="append", default=[], help=f"tune a parameter, e.g. --set tilt_deg=6. Keys: {list(DEFAULTS)}")
    a = ap.parse_args()
    params = {}
    for kv in a.set:
        k, v = kv.split("="); params[k] = float(v) if "." in v or k not in ("n_sparkles", "n_particles") else int(v)
    eyes = collect_eyes(a.eyes, a.eyes_src, a.eyes_at)
    center = tuple(map(float, a.center.split(","))) if a.center else None
    subject, bg = parse_list(a.subject, SUBJECT_MOTIONS), parse_list(a.bg, BG_MOTIONS)
    if a.prepare_only:
        r = prepare(a.image, a.out, a.style, subject, bg, a.size, a.model, center, a.zoom, eyes, a.blink, params,
                    cache=not a.no_cache, eye_kind=a.eye_kind)
    else:
        r = run(a.image, a.out, a.style, subject, bg, a.seconds, a.fps, a.size, a.model, center, a.zoom, eyes,
                a.blink, params, cache=not a.no_cache, eye_kind=a.eye_kind)
    print(json.dumps(r, indent=2, default=str))

if __name__ == "__main__":
    main()
