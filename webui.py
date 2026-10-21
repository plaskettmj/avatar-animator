#!/usr/bin/env python3
"""Minimal local web UI (stdlib only). Run: python webui.py  -> http://127.0.0.1:8765
The eye editor is at http://127.0.0.1:8765/editor (click an eye to add it, drag, resize, test blink, export).

JSON endpoints (used by the eye editor; bots can call them too):
  POST /prepare  multipart form (image + the page's fields, plus "model" and "eye_kind") -> {"job", "prep",
                 "urls"}: the slow step only (mask, crop, background fill, eye detection). Or JSON {"job",
                 "model", "style", "eye_kind"} to re-prepare an existing job (e.g. switch the cut-out model;
                 the first run of a model is slow, later ones are cached). JSON may also set the crop: "center"
                 [x, y] (fractions of the source image) and "zoom" (1 = the image's short side), or "auto_crop":
                 true; the square is kept inside the image. Re-cropping skips segmentation and eye detection (only
                 the crop and background fill run) and keeps the eyes, which are stored in source px. The crop is
                 saved with the job, so /eyes, /preview and /export use it. prep.eyes[*].src is each eye in
                 source-image px (stays valid when the crop changes); prep.eyes[*].box is in output fractions.
  POST /eyes     JSON {"job", "eyes": [{"x", "y", "w"?, "h"?, "kind"?}, ...] (source px), "test_blink": bool}
                 -> {"eyes": [...resolved...], "urls": {"blink": strip PNG}}. Eyes without w/h are grabbed: the
                 eye shape around the point is found automatically, and kind "auto" picks cartoon or photo.
                 With test_blink, renders a strip of rest-pose frames closing the eyes (no loop, no encoding).
  POST /export   JSON {"job", "eyes": [{x, y, w, h, kind}, ...] (source px) or "cx,cy,w,h;..." / [[cx,cy,w,h], ...]
                 (output fractions, as --eyes), "eyes_src", "eyes_at" (strings, as the CLI flags), "no_blink",
                 "style", "subject", "bg", "model", "eye_kind", "seconds", "fps", "blink", "params"}
                 -> {"job", "report", "urls"}. Reuses the job's cached analysis, so it skips segmentation.
                 Omitted settings fall back to the ones used in /prepare.
  POST /preview  JSON {"job", "eyes" (as /eyes), "no_blink", "subject", "bg", "style"?} -> {"eyes", "live", "urls"}:
                 the layers the editor's live preview animates in the browser (preview.js): background frozen at t=0
                 without parallax, the subject layer, and the eye patches closing in steps. Sliders (float, breathe,
                 tilt, blink timing, parallax, loop length) then update instantly with no server call; only eye
                 edits and motion toggles that change the frozen background call this again. live.frozen lists the
                 motions the preview shows still (swirl, hue, twinkle, particles, sparkles, sweep): Export renders them.
"""
import html, io, json, os, re, sys, threading, time, uuid
import numpy as np
from PIL import Image
from email.parser import BytesParser
from email.policy import default as email_default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from animate import run, prepare, export, parse_eyes, collect_eyes, live_layers, MODELS
from avatar_animator.core import STYLES, SUBJECT_MOTIONS, BG_MOTIONS, EYE_KINDS, Analysis, Animator
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
JOB_RE = re.compile(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{6}")
BLINK_STEPS = (0.0, 0.35, 0.7, 1.0, 0.7, 0.35)
WORK = threading.Lock()   # one heavy job at a time: two segmentations at once can run a small machine out of RAM

ROOT = os.environ.get("AVATAR_ANIMATOR_JOBS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "webui-jobs"))
os.makedirs(ROOT, exist_ok=True)

def page(body):
    return f"""<!doctype html><meta charset=utf-8><title>Avatar animator</title>
<style>body{{font-family:system-ui;max-width:900px;margin:2em auto;background:#111;color:#eee}}
fieldset{{border:1px solid #444;margin:.6em 0}} img{{max-width:100%;border-radius:12px}}
.av{{width:96px;height:96px;border-radius:50%;object-fit:cover;margin-right:10px}} a{{color:#8cf}}</style>
<h1>Avatar animator</h1>{body}"""

FORM = page(f"""<form method=post enctype=multipart/form-data action=/animate>
<p><input type=file name=image accept=image/* required></p>
<p>Style <select name=style>{''.join(f'<option>{s}</option>' for s in STYLES)}</select>
 &nbsp; Seconds <input name=seconds value=3 size=3> &nbsp; FPS <input name=fps value=20 size=3></p>
<fieldset><legend>Override subject motions (leave all unticked = style default)</legend>
{''.join(f'<label><input type=checkbox name=subject value={m}> {m}</label> ' for m in SUBJECT_MOTIONS)}</fieldset>
<fieldset><legend>Override background motions</legend>
{''.join(f'<label><input type=checkbox name=bg value={m}> {m}</label> ' for m in BG_MOTIONS)}</fieldset>
<p><button>Animate</button> (takes ~30-90 s) &nbsp; or open the <a href=/editor>eye editor</a></p></form>""")

class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            return self._send(200, FORM)
        if path == "/editor":
            with open(os.path.join(HERE, "editor.html"), "rb") as f:
                return self._send(200, f.read())
        if path == "/preview.js":
            with open(os.path.join(HERE, "preview.js"), "rb") as f:
                return self._send(200, f.read(), "text/javascript; charset=utf-8")
        if path.startswith("/jobs/"):
            p = os.path.normpath(os.path.join(ROOT, path[len("/jobs/"):]))
            if p.startswith(ROOT) and os.path.isfile(p):
                ct = {"gif": "image/gif", "webp": "image/webp", "png": "image/png", "jpg": "image/jpeg", "json": "application/json"}.get(p.rsplit(".", 1)[-1], "application/octet-stream")
                return self._send(200, open(p, "rb").read(), ct)
        self._send(404, "not found", "text/plain")

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, indent=1, default=str), "application/json")

    def _form(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + self.rfile.read(n)
        msg = BytesParser(policy=email_default).parsebytes(raw)
        fields, img = {}, None
        for part in msg.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if name == "image":
                img = part.get_content()
            else:
                fields.setdefault(name, []).append(part.get_content().strip())
        return fields, img

    @staticmethod
    def _new_job(img):
        job = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        d = os.path.join(ROOT, job); os.makedirs(d)
        src = os.path.join(d, "input"); open(src, "wb").write(img)
        return job, d, src

    @staticmethod
    def _urls(job, names):
        return {n: f"/jobs/{job}/{n}" for n in names if os.path.isfile(os.path.join(ROOT, job, n))}

    PREP_FILES = ["prep/" + f for f in ("preview.png", "crop.png", "grid.png", "mask.png", "subject.png",
                                        "background.png", "prep.json", "source.jpg")]

    def _body_json(self):
        return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")

    @staticmethod
    def _job_dir(job):
        job = str(job or "")
        d = os.path.join(ROOT, job)
        return (job, d) if JOB_RE.fullmatch(job) and os.path.isfile(os.path.join(d, "input")) else (None, None)

    @staticmethod
    def _settings(d):
        sp = os.path.join(d, "settings.json")
        if os.path.exists(sp):
            with open(sp) as f:
                return json.load(f)
        return {}

    @staticmethod
    def _save_settings(d, settings):
        with open(os.path.join(d, "settings.json"), "w") as f:
            json.dump(settings, f)

    @staticmethod
    def _clean(settings):
        st = dict(settings)
        if st.get("style") not in STYLES: st.pop("style", None)
        if st.get("model") not in MODELS: st.pop("model", None)
        if st.get("eye_kind") not in EYE_KINDS: st.pop("eye_kind", None)
        try:   # crop: centre as source fractions in [0, 1], zoom >= 1 (the square never grows past the image)
            c = st.get("center")
            st["center"] = None if c is None else [min(max(float(c[0]), 0.0), 1.0), min(max(float(c[1]), 0.0), 1.0)]
            st["zoom"] = min(max(float(st.get("zoom", 1.0)), 1.0), 20.0)
        except (TypeError, ValueError, IndexError):
            st["center"], st["zoom"] = None, 1.0
        if st["center"] is None and st["zoom"] == 1.0:
            st.pop("center"); st.pop("zoom")   # auto crop
        return st

    @staticmethod
    def _source_preview(d):
        """prep/source.jpg: the whole input (long side <= 1280) for the editor's crop drag."""
        p = os.path.join(d, "prep", "source.jpg")
        if not os.path.exists(p):
            im = Image.open(os.path.join(d, "input")).convert("RGB"); im.thumbnail((1280, 1280))
            os.makedirs(os.path.dirname(p), exist_ok=True); im.save(p, quality=90)

    def _prepare(self):
        if self.headers.get("Content-Type", "").startswith("application/json"):
            try:
                req = self._body_json()
            except ValueError:
                return self._json(400, dict(error="body must be JSON"))
            job, d = self._job_dir(req.get("job"))
            if not job:
                return self._json(404, dict(error="unknown job"))
            settings = self._settings(d)
            for k in ("style", "model", "eye_kind", "subject", "bg", "center", "zoom"):
                if k in req: settings[k] = req[k]
            if req.get("auto_crop"):
                settings.pop("center", None); settings.pop("zoom", None)
            src = os.path.join(d, "input")
        else:
            fields, img = self._form()
            if not img:
                return self._json(400, dict(error="no image uploaded"))
            job, d, src = self._new_job(img)
            settings = dict(style=fields.get("style", ["portrait"])[0], subject=fields.get("subject") or None,
                            bg=fields.get("bg") or None, model=fields.get("model", ["birefnet-general-lite"])[0],
                            eye_kind=fields.get("eye_kind", ["auto"])[0])
        settings = self._clean(settings)
        try:
            prep = prepare(src, d, **settings)
            self._source_preview(d)
        except Exception as e:
            return self._json(500, dict(error=str(e)))
        self._save_settings(d, settings)
        self._json(200, dict(job=job, prep=prep, settings=settings, urls=self._urls(job, self.PREP_FILES)))

    @staticmethod
    def _eye_list(req):
        """Manual eyes from a request: "eyes" as a list of dicts (source px) or the --eyes formats, plus the
        "eyes_src" / "eyes_at" strings."""
        eyes = req.get("eyes")
        specs = []
        if isinstance(eyes, str):
            specs += parse_eyes(eyes) or []
        elif eyes:
            for e in eyes:
                if isinstance(e, dict):
                    kind = str(e.get("kind") or "auto").lower()
                    if kind not in EYE_KINDS: raise ValueError(f"eye kind must be one of {EYE_KINDS}")
                    w, h = e.get("w"), e.get("h")
                    specs.append(dict(x=float(e["x"]), y=float(e["y"]), w=float(w) if w else None,
                                      h=float(h) if h else None, kind=kind, space="src"))
                else:
                    specs.append(tuple(map(float, e[:4])) + ((str(e[4]),) if len(e) > 4 else ()))
        return collect_eyes(specs or None, req.get("eyes_src"), req.get("eyes_at"))

    def _animator(self, d, req, force_blink=True):
        """Animator for a job from its saved settings plus the request's eyes / motion overrides (cached analysis)."""
        st = self._clean(self._settings(d))
        eyes = self._eye_list(req)
        src = np.asarray(Image.open(os.path.join(d, "input")).convert("RGB"))
        ana = Analysis(src, cache_dir=os.path.join(d, "prep"), log=lambda *a, **k: None)
        style = req.get("style") if req.get("style") in STYLES else st.get("style", "portrait")
        subj = req.get("subject") if isinstance(req.get("subject"), list) else st.get("subject")
        subj = [m for m in (subj if subj is not None else STYLES[style]["subject"]) if m in SUBJECT_MOTIONS]
        bg = req.get("bg") if isinstance(req.get("bg"), list) else st.get("bg")
        if bg is not None: bg = [m for m in bg if m in BG_MOTIONS]
        if req.get("no_blink"):
            subj, eyes = [m for m in subj if m != "blink"], None
        elif force_blink and "blink" not in subj:
            subj.append("blink")
        return Animator(src, style=style, subject=subj, bg=bg, model=st.get("model", "birefnet-general-lite"),
                        center=st.get("center"), zoom=st.get("zoom", 1.0),
                        eyes=eyes, eye_kind=st.get("eye_kind", "auto"), analysis=ana, log=lambda *a, **k: None)

    @staticmethod
    def _eye_json(an):
        res = []
        for e, sp in zip(an.eyes, an.eye_specs()):
            r = {k: (round(float(v), 1) if k != "kind" else v) for k, v in sp.items()}
            r["grabbed"] = bool(e.get("grab", {}).get("found", False))
            if e["kind"] == "cartoon":   # outline of the grabbed shape, for drawing
                cs, _ = cv2.findContours(e["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                c = cv2.approxPolyDP(max(cs, key=cv2.contourArea), 1.0, True)[:, 0, :].astype(float)
                r["outline"] = [[round(v, 1) for v in an.canvas_to_src(x, y)] for x, y in c]
            res.append(r)
        # eyes the crop leaves out: kept (source px) so the editor still has them when the crop moves back
        res += [dict(e, grabbed=False, off_crop=True) for e in getattr(an, "eyes_off", [])]
        return sorted(res, key=lambda r: r["x"])

    def _eyes(self):
        """Resolve eyes (grab shapes, pick blink styles) and optionally render a blink test strip."""
        try:
            req = self._body_json()
        except ValueError:
            return self._json(400, dict(error="body must be JSON"))
        job, d = self._job_dir(req.get("job"))
        if not job:
            return self._json(404, dict(error="unknown job"))
        try:
            an = self._animator(d, {"eyes": req.get("eyes"), "eyes_src": req.get("eyes_src"), "eyes_at": req.get("eyes_at")})
        except (Exception, SystemExit) as e:
            return self._json(500, dict(error=str(e)))
        res = self._eye_json(an)
        urls = {}
        if req.get("test_blink"):
            strip = np.concatenate([an.still(b) for b in BLINK_STEPS], 1)
            Image.fromarray(strip).save(os.path.join(d, "blink-test.png"))
            urls["blink"] = f"/jobs/{job}/blink-test.png?t={time.time():.3f}"
        self._json(200, dict(job=job, eyes=res, frames=len(BLINK_STEPS), size=an.out, urls=urls))

    def _preview(self):
        """Write the live-preview layers for the current eyes and motions (see POST /preview above)."""
        try:
            req = self._body_json()
        except ValueError:
            return self._json(400, dict(error="body must be JSON"))
        job, d = self._job_dir(req.get("job"))
        if not job:
            return self._json(404, dict(error="unknown job"))
        t0 = time.time()
        try:
            an = self._animator(d, req, force_blink=False)
            stamp = f"?t={time.time():.3f}"
            live = live_layers(an, os.path.join(d, "live"))
        except (Exception, SystemExit) as e:
            return self._json(500, dict(error=str(e)))
        urls = {k: f"/jobs/{job}/live/{os.path.basename(v)}{stamp}" for k, v in live["files"].items()}
        live["prep_s"] = round(time.time() - t0, 2)
        self._json(200, dict(job=job, eyes=self._eye_json(an), live=live, urls=urls))

    def _export(self):
        try:
            req = self._body_json()
        except ValueError:
            return self._json(400, dict(error="body must be JSON"))
        job, d = self._job_dir(req.get("job"))
        if not job:
            return self._json(404, dict(error="unknown job"))
        settings = self._settings(d)
        for k in ("style", "subject", "bg", "blink", "model", "eye_kind", "center", "zoom"):
            if k in req: settings[k] = req[k]
        settings = self._clean(settings)
        if req.get("no_blink"):
            subj = settings.get("subject") or STYLES[settings.get("style", "portrait")]["subject"]
            settings["subject"] = [m for m in subj if m != "blink"]
        try:
            eyes = self._eye_list(req)
            r = export(os.path.join(d, "input"), d, seconds=float(req.get("seconds", 3)), fps=int(req.get("fps", 20)),
                       eyes=eyes, params=req.get("params") or None, **settings)
        except (Exception, SystemExit) as e:
            return self._json(500, dict(error=str(e)))
        self._json(200, dict(job=job, report=r, urls=self._urls(job, ["avatar.gif", "avatar.webp", "contact-sheet.png",
                                                                     "report.json", "eyes.png"])))

    def do_POST(self):
        with WORK:
            return self._post()

    def _post(self):
        if self.path == "/prepare":
            return self._prepare()
        if self.path == "/export":
            return self._export()
        if self.path == "/eyes":
            return self._eyes()
        if self.path == "/preview":
            return self._preview()
        fields, img = self._form()
        if not img:
            return self._send(400, page("<p>No image uploaded.</p>"))
        job, d, src = self._new_job(img)
        style = fields.get("style", ["portrait"])[0]
        try:
            r = run(src, d, style=style, subject=fields.get("subject") or None, bg=fields.get("bg") or None,
                    seconds=float(fields.get("seconds", ["3"])[0]), fps=int(fields.get("fps", ["20"])[0]))
        except Exception as e:
            return self._send(500, page(f"<p>Failed: {html.escape(str(e))}</p><p><a href=/>back</a></p>"))
        body = f"""<p><img class=av src=/jobs/{job}/avatar.gif> <img class=av src=/jobs/{job}/avatar.webp style=width:48px;height:48px>
 <img class=av src=/jobs/{job}/avatar.gif style=width:32px;height:32px> (sidebar-size previews)</p>
<p><a href=/jobs/{job}/avatar.gif download>avatar.gif</a> ({r['gif']['size_mb']} MB) ·
<a href=/jobs/{job}/avatar.webp download>avatar.webp</a> ({r['webp']['size_mb']} MB)</p>
<img src=/jobs/{job}/contact-sheet.png><pre>{html.escape(json.dumps(r, indent=1, default=str))}</pre><p><a href=/>animate another</a></p>"""
        self._send(200, page(body))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8765))
    print(f"Avatar animator UI on http://127.0.0.1:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
