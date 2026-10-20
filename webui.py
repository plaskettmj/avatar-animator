#!/usr/bin/env python3
"""Minimal local web UI (stdlib only). Run: python webui.py  -> http://127.0.0.1:8765

JSON endpoints (for the eye editor and for bots):
  POST /prepare  multipart form (same fields as the page) -> {"job", "prep", "urls"}: the slow step only
                 (mask, crop, background fill, eye detection). prep.eyes_arg / prep.eyes[*].box are fractions
                 of the output square, the same format /export and --eyes take.
  POST /export   JSON {"job", "eyes": "cx,cy,w,h;..." or [[cx,cy,w,h], ...], "style", "subject", "bg",
                 "seconds", "fps", "blink", "params"} -> {"job", "report", "urls"}. Reuses the job's cached
                 analysis, so it skips segmentation. Omitted settings fall back to the ones used in /prepare.
"""
import html, io, json, os, re, sys, time, uuid
from email.parser import BytesParser
from email.policy import default as email_default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from animate import run, prepare, export, parse_eyes
from avatar_animator.core import STYLES, SUBJECT_MOTIONS, BG_MOTIONS

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
<p><button>Animate</button> (takes ~30-90 s)</p></form>""")

class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path == "/":
            return self._send(200, FORM)
        if self.path.startswith("/jobs/"):
            p = os.path.normpath(os.path.join(ROOT, self.path[len("/jobs/"):]))
            if p.startswith(ROOT) and os.path.isfile(p):
                ct = {"gif": "image/gif", "webp": "image/webp", "png": "image/png", "json": "application/json"}.get(p.rsplit(".", 1)[-1], "application/octet-stream")
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

    def _prepare(self):
        fields, img = self._form()
        if not img:
            return self._json(400, dict(error="no image uploaded"))
        job, d, src = self._new_job(img)
        settings = dict(style=fields.get("style", ["portrait"])[0], subject=fields.get("subject") or None,
                        bg=fields.get("bg") or None)
        try:
            prep = prepare(src, d, **settings)
        except Exception as e:
            return self._json(500, dict(error=str(e)))
        json.dump(settings, open(os.path.join(d, "settings.json"), "w"))
        self._json(200, dict(job=job, prep=prep, urls=self._urls(job, ["prep/" + f for f in
                   ("preview.png", "subject.png", "background.png", "prep.json")])))

    def _export(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except ValueError:
            return self._json(400, dict(error="body must be JSON"))
        job = str(req.get("job", ""))
        d = os.path.join(ROOT, job)
        if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{6}", job) or not os.path.isfile(os.path.join(d, "input")):
            return self._json(404, dict(error="unknown job"))
        sp = os.path.join(d, "settings.json")
        settings = json.load(open(sp)) if os.path.exists(sp) else {}
        for k in ("style", "subject", "bg", "blink"):
            if k in req: settings[k] = req[k]
        eyes = req.get("eyes")
        try:
            eyes = parse_eyes(eyes) if isinstance(eyes, str) else [tuple(map(float, e)) for e in eyes] if eyes else None
            r = export(os.path.join(d, "input"), d, seconds=float(req.get("seconds", 3)), fps=int(req.get("fps", 20)),
                       eyes=eyes or None, params=req.get("params") or None, **settings)
        except (Exception, SystemExit) as e:
            return self._json(500, dict(error=str(e)))
        self._json(200, dict(job=job, report=r, urls=self._urls(job, ["avatar.gif", "avatar.webp", "contact-sheet.png",
                                                                     "report.json", "prep/preview.png"])))

    def do_POST(self):
        if self.path == "/prepare":
            return self._prepare()
        if self.path == "/export":
            return self._export()
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
