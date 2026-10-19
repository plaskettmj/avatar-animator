#!/usr/bin/env python3
"""Avatar animator CLI: any image -> seamless looping animated avatar (GIF + WebP).

Example:
  python animate.py photo.jpg --style portrait -o out/portrait
"""
import argparse, json, os, sys, time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from avatar_animator.core import Animator, STYLES, SUBJECT_MOTIONS, BG_MOTIONS, DEFAULTS
from avatar_animator import export

def parse_list(s, allowed):
    if s is None: return None
    if s.strip().lower() in ("", "none"): return []
    items = [x.strip() for x in s.split(",") if x.strip()]
    bad = [x for x in items if x not in allowed]
    if bad: raise SystemExit(f"unknown motion(s) {bad}; choose from {allowed}")
    return items

def run(path, outdir, style="portrait", subject=None, bg=None, seconds=3.0, fps=20, size=512,
        model="birefnet-general-lite", center=None, zoom=1.0, eyes=None, blink="auto", params=None, log=print):
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    src = np.asarray(Image.open(path).convert("RGB"))
    an = Animator(src, out=size, style=style, subject=subject, bg=bg, seconds=seconds, fps=fps, model=model,
                  center=center, zoom=zoom, eyes=eyes, blink=blink, params=params, log=log)
    frames = an.render()
    # exact periodicity check: t=1.0 must equal t=0
    end = an.frame_t(1.0) if hasattr(an, "frame_t") else None
    base = os.path.join(outdir, "avatar")
    g = export.save_gif(frames, base + ".gif", fps)
    w = export.save_webp(frames, base + ".webp", fps)
    # contact sheet: evenly spaced, plus mid-blink frame if blinking
    N = len(frames)
    idx = list(np.linspace(0, N, 6, endpoint=False).astype(int))
    if an.eyes and "blink" in an.subj:
        kb = int(round((an.P["blink_at"] + an.P["blink_len"] / 2) * N)) % N
        idx[4] = kb
    idx = sorted(set(idx))
    export.contact_sheet(frames, os.path.join(outdir, "contact-sheet.png"), idx,
                         [f"f{k} t={k / fps:.2f}s" + (" blink" if an.eyes and abs(k - int(round((an.P['blink_at'] + an.P['blink_len'] / 2) * N))) < 1 else "") for k in idx])
    report = dict(input=path, style=style, subject_motions=an.subj, bg_motions=an.bgm,
                  eyes=[e["kind"] for e in an.eyes], anchored=bool(an.anchored), frames=N, fps=fps,
                  seconds=seconds, gif=export.check_file(base + ".gif"), webp=export.check_file(base + ".webp"),
                  periodic_t1_equals_t0=bool(end is not None and np.array_equal(end, frames[0])),
                  render_s=round(time.time() - t0, 1))
    json.dump(report, open(os.path.join(outdir, "report.json"), "w"), indent=2, default=str)
    return report

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
    ap.add_argument("--set", action="append", default=[], help=f"tune a parameter, e.g. --set tilt_deg=6. Keys: {list(DEFAULTS)}")
    a = ap.parse_args()
    params = {}
    for kv in a.set:
        k, v = kv.split("="); params[k] = float(v) if "." in v or k not in ("n_sparkles", "n_particles") else int(v)
    eyes = [tuple(map(float, e.split(","))) for e in a.eyes.split(";")] if a.eyes else None
    center = tuple(map(float, a.center.split(","))) if a.center else None
    r = run(a.image, a.out, a.style, parse_list(a.subject, SUBJECT_MOTIONS), parse_list(a.bg, BG_MOTIONS),
            a.seconds, a.fps, a.size, a.model, center, a.zoom, eyes, a.blink, params)
    print(json.dumps(r, indent=2, default=str))

if __name__ == "__main__":
    main()
