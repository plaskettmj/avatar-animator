#!/usr/bin/env python3
"""Minimal local web UI (stdlib only). Run: python webui.py  -> http://127.0.0.1:8765"""
import html, io, json, os, sys, time, uuid
from email.parser import BytesParser
from email.policy import default as email_default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from animate import run
from avatar_animator.core import STYLES, SUBJECT_MOTIONS, BG_MOTIONS

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webui-jobs")
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

    def do_POST(self):
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
        if not img:
            return self._send(400, page("<p>No image uploaded.</p>"))
        job = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        d = os.path.join(ROOT, job); os.makedirs(d)
        src = os.path.join(d, "input"); open(src, "wb").write(img)
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
