#!/usr/bin/env python3
"""Avatar animator CLI: any image -> seamless looping animated avatar (GIF + WebP).

Example:
  python animate.py photo.jpg --style portrait -o out/portrait
  python animate.py photo.jpg --style portrait -o out/portrait --prepare-only   # slow step only: crop + eyes
  python animate.py photo.jpg --style portrait -o out/portrait --eyes "0.40,0.42,0.08,0.05;0.58,0.42,0.08,0.05"
    (re-export with the same -o reuses out/portrait/prep/, so segmentation is skipped)
"""
import argparse, json, os, sys, time
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from avatar_animator.core import Analysis, Animator, STYLES, SUBJECT_MOTIONS, BG_MOTIONS, DEFAULTS
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

def parse_eyes(s):
    """Parse --eyes 'cx,cy,w,h;...' into a list of 4-tuples (fractions of the output square). '' or 'none' -> []."""
    if s is None: return None
    if s.strip().lower() in ("", "none"): return []
    eyes = []
    for e in s.split(";"):
        if not e.strip(): continue
        v = tuple(map(float, e.split(",")))
        if len(v) != 4: raise SystemExit(f"--eyes needs cx,cy,w,h per eye, got {e!r}")
        eyes.append(v)
    return eyes

def prepare(path, outdir, style="portrait", subject=None, bg=None, size=512, model="birefnet-general-lite",
            center=None, zoom=1.0, eyes=None, blink="auto", params=None, cache=True, log=print):
    """Slow step only: subject mask, crop, background fill and eye detection. No frames are rendered or encoded.

    Writes <outdir>/prep/: the cached analysis (analysis.json, mask-<model>.png), prep.json, subject.png
    (RGBA subject layer on the padded canvas), background.png (filled background on the same canvas) and
    preview.png (the cropped square with the eyes outlined and numbered). Returns the prep.json dict; its
    "eyes_arg" string can be passed straight back as --eyes / export(eyes=...) after editing."""
    t0 = time.time()
    os.makedirs(outdir, exist_ok=True)
    src = _load(path)
    ana = _analysis(src, outdir, cache, log)
    ana.mask(model); ana.face(); ana.pet()   # warm every slow step, so any later export (e.g. turning blink on) is fast
    an = Animator(src, out=size, style=style, subject=subject, bg=bg, model=model, center=center, zoom=zoom,
                  eyes=eyes, blink=blink, params=params, log=log, analysis=ana)
    return _write_prep(an, path, src, outdir, model, center, zoom, blink, t0)

def _write_prep(an, path, src, outdir, model, center, zoom, blink, t0):
    d = os.path.join(outdir, "prep"); os.makedirs(d, exist_ok=True)
    u8 = lambda a: (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(np.dstack([u8(an.fg), u8(an.alpha)]), "RGBA").save(os.path.join(d, "subject.png"))
    Image.fromarray(u8(an.bg_fill)).save(os.path.join(d, "background.png"))
    o, out = an.off, an.out
    prev = Image.fromarray(u8(an.img[o:o + out, o:o + out]))
    dr = ImageDraw.Draw(prev)
    boxes = an.eye_boxes()
    for i, (cx, cy, w, h) in enumerate(boxes, 1):
        x0, y0, x1, y1 = (cx - w / 2) * out, (cy - h / 2) * out, (cx + w / 2) * out, (cy + h / 2) * out
        dr.ellipse((x0, y0, x1, y1), outline=(255, 230, 0), width=2)
        dr.text((x1 + 3, y0 - 2), str(i), fill=(255, 230, 0))
    prev.save(os.path.join(d, "preview.png"))
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
        eye_source=an.eye_source(), eyes=[dict(kind=e["kind"], box=list(b)) for e, b in zip(an.eyes, boxes)],
        eyes_arg=eyes_arg(boxes),
        files=dict(subject="prep/subject.png", background="prep/background.png", preview="prep/preview.png"),
        cache=dict(hits=an.analysis.hits, misses=an.analysis.misses),
        prep_s=round(time.time() - t0, 2))
    with open(os.path.join(d, "prep.json"), "w") as f:
        json.dump(prep, f, indent=2, default=str)
    return prep

def export(path, outdir, style="portrait", subject=None, bg=None, seconds=3.0, fps=20, size=512,
           model="birefnet-general-lite", center=None, zoom=1.0, eyes=None, blink="auto", params=None,
           cache=True, log=print):
    """Render and encode (GIF + WebP + contact sheet + report.json). Reuses the cached analysis in
    <outdir>/prep/ when it matches the image, so changing eyes, motions or parameters skips segmentation."""
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    src = _load(path)
    ana = _analysis(src, outdir, cache, log)
    an = Animator(src, out=size, style=style, subject=subject, bg=bg, seconds=seconds, fps=fps, model=model,
                  center=center, zoom=zoom, eyes=eyes, blink=blink, params=params, log=log, analysis=ana)
    t_prep = time.time() - t0
    if cache:   # keep prep/ (layers, preview, eyes) in step with what was exported
        _write_prep(an, path, src, outdir, model, center, zoom, blink, t0)
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
                  eyes=[e["kind"] for e in an.eyes], eye_source=an.eye_source(), eyes_arg=eyes_arg(boxes),
                  anchored=bool(an.anchored), frames=N, fps=fps,
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
        cache=True):
    """Full pipeline (prepare + export). Same as before; the slow analysis is cached in <outdir>/prep/."""
    return export(path, outdir, style, subject, bg, seconds, fps, size, model, center, zoom, eyes, blink, params,
                  cache=cache, log=log)

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
    ap.add_argument("--model", default="birefnet-general-lite", choices=["birefnet-general-lite", "birefnet-general", "u2net", "isnet-general-use"])
    ap.add_argument("--center", help="crop center as fractions x,y of the source image (e.g. 0.43,0.47)")
    ap.add_argument("--zoom", type=float, default=1.0, help=">1 crops tighter")
    ap.add_argument("--eyes", help="manual eyes in output fractions 'cx,cy,w,h;cx,cy,w,h'")
    ap.add_argument("--blink", default="auto", choices=["auto", "face", "pet", "cartoon", "off"])
    ap.add_argument("--prepare-only", action="store_true",
                    help="run only the slow step (mask, crop, eyes); write <out>/prep/ and print prep.json")
    ap.add_argument("--no-cache", action="store_true", help="don't read or write the <out>/prep/ cache")
    ap.add_argument("--set", action="append", default=[], help=f"tune a parameter, e.g. --set tilt_deg=6. Keys: {list(DEFAULTS)}")
    a = ap.parse_args()
    params = {}
    for kv in a.set:
        k, v = kv.split("="); params[k] = float(v) if "." in v or k not in ("n_sparkles", "n_particles") else int(v)
    eyes = parse_eyes(a.eyes) or None
    center = tuple(map(float, a.center.split(","))) if a.center else None
    subject, bg = parse_list(a.subject, SUBJECT_MOTIONS), parse_list(a.bg, BG_MOTIONS)
    if a.prepare_only:
        r = prepare(a.image, a.out, a.style, subject, bg, a.size, a.model, center, a.zoom, eyes, a.blink, params,
                    cache=not a.no_cache)
    else:
        r = run(a.image, a.out, a.style, subject, bg, a.seconds, a.fps, a.size, a.model, center, a.zoom, eyes,
                a.blink, params, cache=not a.no_cache)
    print(json.dumps(r, indent=2, default=str))

if __name__ == "__main__":
    main()
