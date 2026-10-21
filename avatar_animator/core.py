"""Avatar animator core: segmentation, layering, motion presets, rendering."""
import hashlib, json, math, os, re
import numpy as np, cv2
from PIL import Image

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Model weights are downloaded on first use (never bundled). Override the folder with AVATAR_ANIMATOR_MODELS.
MODEL_DIR = os.environ.get("AVATAR_ANIMATOR_MODELS", os.path.join(HERE, "models"))
os.environ.setdefault("U2NET_HOME", MODEL_DIR)  # rembg stores weights in $U2NET_HOME/models/<name>/

# Extra model files fetched on first use: (url, sha256)
MODEL_FILES = {
    "face_landmarker.task": (  # MediaPipe Face Landmarker bundle, Apache-2.0
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"),
    "haarcascade_frontalcatface_extended.xml": (  # OpenCV cat-face cascade by Joseph Howse, BSD-3-Clause
        "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalcatface_extended.xml",
        "ffd0d1d28f07d0376c89db4c9845009c2855cc76c8cff763d5331dc9361da854"),
}

def model_path(name):
    """Return the local path of a model file, downloading and checksum-verifying it on first use.
    Returns None if the download fails (callers then skip that feature)."""
    path = os.path.join(MODEL_DIR, name)
    if os.path.exists(path):
        return path
    url, sha = MODEL_FILES[name]
    import hashlib, urllib.request
    os.makedirs(MODEL_DIR, exist_ok=True)
    tmp = path + ".part"
    try:
        print(f"  downloading {name} ...")
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            f.write(r.read())
        if hashlib.sha256(open(tmp, "rb").read()).hexdigest() != sha:
            raise ValueError("checksum mismatch")
        os.replace(tmp, path)
        return path
    except Exception as e:
        print(f"  could not download {name}: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return None

# ----------------------------------------------------------------- presets
STYLES = {
    # subject motions                       background motions
    "portrait": dict(subject=["breathe", "tilt", "blink"], bg=["parallax", "particles"]),
    "logo":     dict(subject=["float", "breathe", "sweep"], bg=["backdrop", "hue", "sparkles"]),
    "space":    dict(subject=["float", "blink"], bg=["swirl", "twinkle"]),
    "pet":      dict(subject=["tilt", "breathe", "blink"], bg=["parallax", "sparkles"]),
}
SUBJECT_MOTIONS = ["float", "breathe", "tilt", "blink", "sweep"]
BG_MOTIONS = ["parallax", "swirl", "twinkle", "sparkles", "particles", "sweep", "hue", "backdrop"]

DEFAULTS = dict(
    float_amp=0.035,      # fraction of output size
    breathe_amp=0.035,    # scale pulse amplitude
    tilt_deg=4.0,
    blink_at=0.62, blink_len=0.11,   # fraction of loop
    parallax_amp=0.03,
    swirl_deg=80.0, swirl_ratio=1.0, swirl_tilt=0.0, swirl_r0=0.55, swirl_r1=0.75,
    hue_deg=40.0,
    sweep_strength=0.6,
    n_sparkles=12, n_particles=36,
)

# ----------------------------------------------------------------- segmentation
_sessions = {}
def _session_options():
    """ONNX Runtime options for the cut-out models. The CPU memory arena is off: with it on, the full BiRefNet's
    second run in the same process grew the arena from about 9 to 14 GB (the web UI was OOM-killed on a 16 GB box
    on the second full-model Prepare); without it memory goes back to the OS after each run."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = False
    return so

def segment(img_rgb_u8, model="birefnet-general-lite"):
    """Return soft subject mask in [0,1] (float32) using rembg (MIT) + chosen weights."""
    from rembg import new_session, remove
    if model not in _sessions:
        # keep one model in memory: the full BiRefNet peaks around 8 GB, so holding the lite one as well
        # (e.g. the web editor switching cut-out models) can exhaust a 16 GB machine
        _sessions.clear()
        import gc; gc.collect()
        _sessions[model] = new_session(model, sess_opts=_session_options())
    m = remove(Image.fromarray(img_rgb_u8), session=_sessions[model], only_mask=True)
    return np.asarray(m).astype(np.float32) / 255.0

# ----------------------------------------------------------------- eyes
LEFT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

def detect_face_eyes(img_rgb_u8):
    """MediaPipe Face Landmarker (Apache-2.0). Returns (eyes, face_box) or ([], None)."""
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks import python as mpt
        path = model_path("face_landmarker.task")
        if not path:
            return [], None
        opts = vision.FaceLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=path, delegate=mpt.BaseOptions.Delegate.CPU),
            num_faces=1)
        lm = vision.FaceLandmarker.create_from_options(opts)
        try:
            r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img_rgb_u8)))
        finally:
            lm.close()
    except Exception as e:  # graceful fallback
        print("  face landmarker unavailable:", e)
        return [], None
    if not r.face_landmarks:
        return [], None
    H, W = img_rgb_u8.shape[:2]
    pts = np.array([[p.x * W, p.y * H] for p in r.face_landmarks[0]], np.float32)
    eyes = []
    for idx in (LEFT_EYE, RIGHT_EYE):
        poly = pts[idx]
        eyes.append(dict(kind="photo", poly=poly))
    x0, y0 = pts.min(0); x1, y1 = pts.max(0)
    return eyes, (x0, y0, x1 - x0, y1 - y0)

def detect_cartoon_eyes(img, mask):
    """Heuristic for mascots/characters: two similar high-contrast blobs inside the subject."""
    m = (mask > 0.5).astype(np.uint8)
    if m.sum() < 500:
        return []
    m = cv2.erode(m, np.ones((5, 5), np.uint8))
    lum = img.mean(2)
    med = np.median(lum[m > 0])
    # brightness contrast, or a strong colour shift (cyan eyes on a white helmet have similar brightness)
    labc = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    chroma = np.linalg.norm(labc[..., 1:] - np.median(labc[m > 0][:, 1:], 0), axis=2)
    cand = (((np.abs(lum - med) > 0.45) | (chroma > 30)) & (m > 0)).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(cand)
    area_s = m.sum()
    comps = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if not (0.003 * area_s < a < 0.12 * area_s):
            continue
        if a / float(w * h) < 0.35:   # wedge-shaped eyes fill about 40% of their box
            continue
        comps.append((i, cen[i], a, w, h))
    best = None
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            a, b = comps[i], comps[j]
            if max(a[2], b[2]) / min(a[2], b[2]) > 3.0:
                continue
            dx, dy = abs(a[1][0] - b[1][0]), abs(a[1][1] - b[1][1])
            sz = math.sqrt((a[2] + b[2]) / 2)
            if not (0.6 * sz < dx < 5 * sz) or dy > 1.2 * sz:
                continue
            score = dy / sz + abs(math.log(a[2] / b[2]))
            if best is None or score < best[0]:
                best = (score, a, b)
    if best is None:
        return []
    eyes = []
    for c in best[1:]:
        em = (lab == c[0]).astype(np.uint8)
        eyes.append(dict(kind="cartoon", mask=em))
    return eyes

def detect_pet_eyes(img_rgb_u8):
    """Cat face: OpenCV Haar cascade by Joseph Howse (BSD-3-Clause) gives the face box; eyes are then
    located as the most distinct-coloured blob near the expected eye positions (Lab distance + Otsu)
    and fitted with an ellipse. Returns a list of eye polygons in image px, or []."""
    path = model_path("haarcascade_frontalcatface_extended.xml")
    if not path:
        return []
    g = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2GRAY)
    faces = cv2.CascadeClassifier(path).detectMultiScale(g, 1.05, 3, minSize=(60, 60))
    if len(faces) == 0:
        return []
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    lab = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2Lab).astype(np.float32)
    polys = []
    for fx in (0.28, 0.72):
        gx, gy = x + fx * w, y + 0.43 * h
        win = int(0.17 * w)
        X0, Y0 = int(max(gx - win, 0)), int(max(gy - win, 0))
        sub = lab[Y0:Y0 + 2 * win, X0:X0 + 2 * win]
        if sub.size == 0:
            return []
        d = np.linalg.norm(sub - np.median(sub.reshape(-1, 3), 0), axis=2)
        d8 = np.clip(d * 255 / max(d.max(), 1e-3), 0, 255).astype(np.uint8)
        _, bm = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bm = cv2.morphologyEx(bm, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        bm = cv2.morphologyEx(bm, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        n, lbl, st, cen = cv2.connectedComponentsWithStats(bm)
        best = None
        for i in range(1, n):
            if not (np.pi * (0.035 * w) ** 2 < st[i, 4] < np.pi * (0.13 * w) ** 2):
                continue
            dd = math.hypot(cen[i][0] + X0 - gx, cen[i][1] + Y0 - gy)
            if dd < 0.8 * win and (best is None or dd < best[0]):
                best = (dd, i)
        if best is None:
            return []   # be conservative: no blink rather than a wrong one
        cnt, _ = cv2.findContours((lbl == best[1]).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        (ex, ey), (ea, eb), ang = cv2.fitEllipse(cnt[0])
        poly = cv2.ellipse2Poly((int(ex + X0), int(ey + Y0)), (int(ea / 2 * 1.05), int(eb / 2 * 1.05)), int(ang), 0, 360, 10)
        polys.append(poly.astype(np.float32))
    return polys

def eyes_from_boxes(boxes, E):
    """Manual eyes: list of (cx, cy, w, h) in canvas px -> photo-style ellipse eyes."""
    eyes = []
    for (cx, cy, w, h) in boxes:
        t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        poly = np.stack([cx + w / 2 * np.cos(t), cy + h / 2 * np.sin(t)], 1).astype(np.float32)
        eyes.append(dict(kind="photo", poly=poly))
    return eyes

EYE_KINDS = ("auto", "cartoon", "photo")
ILLUSTRATION_THRESHOLD = 0.85   # illustration_score at or above this counts as flat artwork (cartoon blink)

def illustration_score(img_rgb_u8):
    """Share of pixels covered by the 12 most common coarse colours (RGB in steps of 32). Flat artwork
    (cartoons, logos, app icons) scores about 0.9-1.0, photos about 0.6-0.7."""
    small = cv2.resize(np.ascontiguousarray(img_rgb_u8), (512, 512), interpolation=cv2.INTER_NEAREST)
    q = (small // 32).reshape(-1, 3).astype(np.int32)
    c = np.bincount(q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2], minlength=512)
    return float(np.sort(c)[::-1][:12].sum() / c.sum())

def _grab_once(img, cx, cy, hw, hh, out, reach):
    """One pass of grab_eye in a window of half-size (hw, hh) around (cx, cy). Returns (blob, d, thr, x0, y0,
    touches_edge) or None. blob is the chosen component (uint8, window-sized)."""
    x0, x1 = int(max(cx - hw, 0)), int(min(cx + hw + 1, img.shape[1]))
    y0, y1 = int(max(cy - hh, 0)), int(min(cy + hh + 1, img.shape[0]))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    win = (np.clip(img[y0:y1, x0:x1], 0, 1) * 255).astype(np.uint8)
    lab = cv2.cvtColor(win, cv2.COLOR_RGB2LAB).astype(np.float32)
    ring = np.concatenate([lab[:2].reshape(-1, 3), lab[-2:].reshape(-1, 3),
                           lab[:, :2].reshape(-1, 3), lab[:, -2:].reshape(-1, 3)])
    d = np.linalg.norm(lab - np.median(ring, 0), axis=2)
    otsu, _ = cv2.threshold(np.clip(d * 2, 0, 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(otsu / 2, 22.0)
    cand = cv2.morphologyEx((d > thr).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab_, st, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    px, py = cx - x0, cy - y0
    H, W = cand.shape
    best = None
    for i in range(1, n):
        bx, by, bw, bh, a = st[i]
        if not (0.0012 * out * out <= a <= 0.06 * out * out):
            continue
        if (bx == 0) + (by == 0) + (bx + bw == W) + (by + bh == H) >= 3:
            continue   # a region that fills the window is background, not an eye
        ci = (lab_[by:by + bh, bx:bx + bw] == i).astype(np.uint8)
        cs, _ = cv2.findContours(ci, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(np.concatenate(cs) + np.array([bx, by], np.int32))
        # signed distance from the point to the blob's hull: > 0 inside. A blob that surrounds the point (the
        # eye outline around the iris) wins over a small one next to it (a pupil or highlight), then the biggest.
        sd = cv2.pointPolygonTest(hull, (float(px), float(py)), True)
        key = (0, 0.0, -a) if sd >= 0 else (1, -sd, -a)
        if -sd <= reach and (best is None or key < best[0]):
            best = (key, i)
    if best is None:
        return None
    bx, by, bw, bh, _ = st[best[1]]
    touches = ((bx == 0 and x0 > 0) or (by == 0 and y0 > 0) or
               (bx + bw == W and x1 < img.shape[1]) or (by + bh == H and y1 < img.shape[0]))
    return (lab_ == best[1]).astype(np.uint8), d, thr, x0, y0, touches

def grab_eye(img, cx, cy, out, box=None):
    """Find the shape of the eye at a clicked point (canvas px) in an RGB float image.

    The colour around the edge of a search window is taken as the face colour; pixels far from it (Lab
    distance, Otsu) form candidate blobs, and the blob around (or nearest) the point, holes filled and convex
    hull taken, is the eye. If the blob runs into the window edge (an off-centre click), the window is moved
    onto it and enlarged. box=(w, h) limits the window to about 1.3x that size. Returns (mask uint8 same size
    as img, info) or (None, info) when nothing eye-like is there."""
    hw, hh = (max(box[0] * 0.65, 4), max(box[1] * 0.65, 4)) if box else (0.12 * out, 0.12 * out)
    reach = (0.5 * max(box) if box else 0.04 * out) + 2
    r = _grab_once(img, cx, cy, hw, hh, out, reach)
    for _ in range(2):
        if r is None or not r[5] or box:
            break
        m, _, _, x0, y0, _ = r
        ys, xs = np.nonzero(m)
        ncx, ncy = x0 + (xs.min() + xs.max()) / 2, y0 + (ys.min() + ys.max()) / 2
        hw, hh = hw * 1.5, hh * 1.5
        r2 = _grab_once(img, ncx, ncy, hw, hh, out, reach)
        if r2 is None:
            break
        r = r2
    if r is None:
        return None, dict(found=False)
    m, d, thr, x0, y0, _ = r
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(cs, key=cv2.contourArea))
    filled = np.zeros_like(m)
    cv2.drawContours(filled, [hull], -1, 1, -1)
    info = dict(found=True, solidity=round(float(m.sum()) / max(float(filled.sum()), 1.0), 3),
                contrast=round(float(d[m > 0].mean() / thr), 2))
    filled = cv2.dilate(filled, np.ones((3, 3), np.uint8))   # take in the anti-aliased outline
    full = np.zeros(img.shape[:2], np.uint8)
    full[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = filled
    return full, info

def parse_eye_spec(s, space="src"):
    """'x,y[,w,h][,kind]' -> eye spec dict. space='src': source-image px; space='out': fractions of the output square."""
    toks = [t.strip() for t in s.replace(":", ",").split(",") if t.strip()]
    kind = "auto"
    if toks and not re.fullmatch(r"[-+0-9.eE]+", toks[-1]):
        kind = toks.pop().lower()
    if kind not in EYE_KINDS:
        raise ValueError(f"eye kind must be one of {EYE_KINDS}, got {kind!r}")
    v = [float(t) for t in toks]
    if len(v) not in (2, 4):
        raise ValueError(f"an eye needs x,y or x,y,w,h (plus an optional kind), got {s!r}")
    return dict(x=v[0], y=v[1], w=v[2] if len(v) == 4 else None, h=v[3] if len(v) == 4 else None,
                kind=kind, space=space)

def format_eye_specs(specs):
    """Eye specs (source px) -> the --eyes-src string 'x,y,w,h,kind;...'."""
    return ";".join(f"{e['x']:.1f},{e['y']:.1f},{e['w']:.1f},{e['h']:.1f},{e['kind']}" for e in specs)

# ----------------------------------------------------------------- slow analysis (cacheable)
ANALYSIS_VERSION = 1

class Analysis:
    """The slow, settings-independent part of preparing an image: the subject mask (BiRefNet/rembg) and the
    face and cat-eye detectors, all run on a working copy with long side <= 1024 px. Results are computed on
    first use and, when cache_dir is given, saved there and reused by later runs on the same image bytes.
    Everything that depends on style, crop, size or eyes is cheap and is recomputed by Animator each time."""

    def __init__(self, src_rgb, cache_dir=None, key=None, log=print):
        self.log = log
        H0, W0 = src_rgb.shape[:2]
        self.sc = min(1.0, 1024 / max(H0, W0))
        self.work = (cv2.resize(src_rgb, (int(W0 * self.sc), int(H0 * self.sc)), interpolation=cv2.INTER_AREA)
                     if self.sc < 1 else src_rgb)
        self.key = key or hashlib.sha256(np.ascontiguousarray(src_rgb).tobytes()).hexdigest()
        self.cache_dir = cache_dir
        self.hits, self.misses = [], []   # which slow steps came from the cache / were computed (unique, in order)
        self._meta = dict(version=ANALYSIS_VERSION, key=self.key, source_size=[W0, H0], scale=self.sc)
        self._masks = {}
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            mp = os.path.join(cache_dir, "analysis.json")
            old = None
            if os.path.exists(mp):
                try:
                    with open(mp) as f:
                        old = json.load(f)
                except ValueError:
                    old = None
            if old and old.get("version") == ANALYSIS_VERSION and old.get("key") == self.key:
                self._meta = old
            else:   # different image (or old format): drop stale cache files
                for f in os.listdir(cache_dir):
                    if f == "analysis.json" or (f.startswith("mask-") and f.endswith(".png")):
                        os.remove(os.path.join(cache_dir, f))
                self._save()

    def _note(self, lst, what):
        # a step computed in this run is a miss; later in-memory reuse of it is not also a cache hit
        if what not in lst and not (lst is self.hits and what in self.misses):
            lst.append(what)

    def _save(self):
        if self.cache_dir:
            tmp = os.path.join(self.cache_dir, "analysis.json.part")
            with open(tmp, "w") as f:
                json.dump(self._meta, f, indent=1)
            os.replace(tmp, os.path.join(self.cache_dir, "analysis.json"))

    def mask(self, model="birefnet-general-lite"):
        """Soft subject mask at working resolution, float32 in [0,1]."""
        if model in self._masks:
            return self._masks[model]
        path = os.path.join(self.cache_dir, f"mask-{model}.png") if self.cache_dir else None
        if path and os.path.exists(path):
            m = np.asarray(Image.open(path)).astype(np.float32) / 255.0   # lossless: rembg masks are 8-bit
            self._note(self.hits, f"mask:{model}")
        else:
            m = segment(self.work, model)
            self._note(self.misses, f"mask:{model}")
            if path:
                Image.fromarray(np.round(m * 255).astype(np.uint8)).save(path)
        self._masks[model] = m
        return m

    def face(self):
        """(eyes, face_box) from MediaPipe on the working copy, cached."""
        if "face" in self._meta:
            self._note(self.hits, "face")
            f = self._meta["face"]
            return ([dict(kind="photo", poly=np.array(p, np.float32)) for p in f["eyes"]],
                    tuple(np.float32(v) for v in f["box"]) if f["box"] is not None else None)
        eyes, box = detect_face_eyes(self.work)
        self._note(self.misses, "face")
        self._meta["face"] = dict(eyes=[e["poly"].tolist() for e in eyes],
                                  box=[float(v) for v in box] if box is not None else None)
        self._save()
        return eyes, box

    def pet(self):
        """Cat-eye polygons on the working copy, cached."""
        if "pet" in self._meta:
            self._note(self.hits, "pet")
            return [np.array(p, np.float32) for p in self._meta["pet"]]
        polys = detect_pet_eyes(self.work)
        self._note(self.misses, "pet")
        self._meta["pet"] = [p.tolist() for p in polys]
        self._save()
        return polys

# ----------------------------------------------------------------- helpers
def soft(mask, r):
    return cv2.GaussianBlur(mask.astype(np.float32), (0, 0), max(r, 0.5))

def fill_background(img, mask, dil):
    """Fill subject hole: inpaint at low res (large holes) + blend into full res."""
    hole = (cv2.dilate((mask > 0.03).astype(np.uint8), np.ones((dil, dil), np.uint8)) > 0)
    H, W = mask.shape
    f = 4
    small = cv2.resize(img, (W // f, H // f), interpolation=cv2.INTER_AREA)
    hs = cv2.resize(hole.astype(np.uint8), (W // f, H // f), interpolation=cv2.INTER_NEAREST)
    hs = cv2.dilate(hs, np.ones((7, 7), np.uint8))   # keep fringe pixels out of the inpaint boundary
    inp = cv2.inpaint((small * 255).astype(np.uint8), hs, 9, cv2.INPAINT_TELEA).astype(np.float32) / 255
    inp = cv2.GaussianBlur(inp, (0, 0), 2)
    inp = cv2.resize(inp, (W, H), interpolation=cv2.INTER_CUBIC)
    a = soft(cv2.dilate(hole.astype(np.uint8), np.ones((9, 9), np.uint8)), dil / 4)[..., None]
    return img * (1 - a) + inp * a

def affine_about(px, py, angle_deg, scale, tx, ty):
    M = cv2.getRotationMatrix2D((float(px), float(py)), angle_deg, scale)
    M[0, 2] += tx; M[1, 2] += ty
    return M

def tri(p):
    return np.sin(np.pi * p) ** 2

def clamp_crop(cx, cy, side, W, H):
    """Keep the visible crop square inside the source image: side at most the image's short side, centre moved so
    no edge passes the border (past the border there are only replicated edge pixels, which look smeared).
    preview.js clampCrop() mirrors this for the editor's crop drag."""
    side = float(min(max(side, 8.0), W, H))
    return (float(min(max(cx, side / 2), W - side / 2)), float(min(max(cy, side / 2), H - side / 2)), side)

# ----------------------------------------------------------------- the animator
class Animator:
    def __init__(self, src_rgb, out=512, style="portrait", subject=None, bg=None, seconds=3.0, fps=20,
                 model="birefnet-general-lite", center=None, zoom=1.0, eyes=None, blink="auto",
                 seed=7, params=None, log=print, analysis=None, eye_kind="auto"):
        """eyes: manual eyes, as a list of eye specs (dicts from parse_eye_spec: x, y, optional w, h, kind and
        space 'src' = source-image px or 'out' = fractions of the output square) or, as before, (cx, cy, w, h)
        tuples in output fractions (an optional 5th item is the kind). Specs without w/h are grabbed: the eye
        shape around the point is found automatically. eye_kind: blink style for eyes whose kind is 'auto'."""
        self.log = log
        self.P = dict(DEFAULTS); self.P.update(params or {})
        st = STYLES.get(style, STYLES["portrait"])
        self.subj = list(subject if subject is not None else st["subject"])
        self.bgm = list(bg if bg is not None else st["bg"])
        if eyes and "blink" not in self.subj:
            self.subj.append("blink")   # placing eyes by hand means "make these blink", whatever the style
        self.style = style
        self.out, self.fps, self.seconds = out, fps, seconds
        self.N = int(round(fps * seconds))
        self.rng = np.random.default_rng(seed)
        self.margin = 0.14
        E = self.E = int(round(out * (1 + 2 * self.margin)))
        self.off = (E - out) // 2

        # --- slow analysis (segmentation, eye detectors) on a working copy (long side <= 1024).
        # An Analysis can be passed in (e.g. loaded from a prep cache) so re-exports skip it.
        src = src_rgb
        H0, W0 = src.shape[:2]
        if analysis is None:
            analysis = Analysis(src, log=log)
        self.analysis = analysis
        sc, work = analysis.sc, analysis.work
        mask_w = analysis.mask(model)
        mask_full = cv2.resize(mask_w, (W0, H0), interpolation=cv2.INTER_LINEAR)
        self.mask_cover = float((mask_w > 0.5).mean())

        # --- face (for crop) on the working copy
        face_eyes, face_box = ([], None)
        if "blink" in self.subj and blink in ("auto", "face"):
            face_eyes, face_box = analysis.face()

        # --- choose crop square (source px)
        ys, xs = np.nonzero(mask_full > 0.5)
        if center is not None:
            cx, cy = center[0] * W0, center[1] * H0
            side = min(W0, H0) / zoom
        elif face_box is not None:
            fx, fy, fw, fh = [v / sc for v in face_box]
            side = min(max(fw, fh) * 2.6, min(W0, H0))
            cx, cy = fx + fw / 2, fy + fh / 2 + 0.04 * side
            side /= zoom
        elif len(xs) > 50 and self.mask_cover < 0.92:
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            bw, bh = x1 - x0, y1 - y0
            side = max(bw, bh) * {'space': 2.6, 'logo': 1.55}.get(style, 1.3)
            anchored = y1 > H0 - 3
            if anchored:   # subject runs off the bottom (bust/pet photo): keep top of subject in view
                side = min(side, min(W0, H0))
                cx, cy = (x0 + x1) / 2, y0 + 0.43 * side
            else:
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            side = max(side, 0.25 * min(W0, H0)) / zoom
        else:
            cx, cy, side = W0 / 2, H0 / 2, min(W0, H0) / zoom
        self.crop_wanted = (float(cx), float(cy), float(side))
        cx, cy, side = clamp_crop(cx, cy, side, W0, H0)
        self.crop = (cx, cy, side)
        self.crop_clamped = bool(np.abs(np.subtract(self.crop, self.crop_wanted)).max() > 0.5)

        # --- extract extended canvas E x E (margin around the visible square)
        s = out / side
        M = np.float32([[s, 0, E / 2 - s * cx], [0, s, E / 2 - s * cy]])
        self.M, self.scale, self.src_size = M, s, (W0, H0)
        srcf = src.astype(np.float32) / 255
        self.img = cv2.warpAffine(srcf, M, (E, E), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        # mask outside the source image is empty, except below the bottom edge when the subject runs off it
        # (busts, pets): there the last row is extended so the body continues when the subject moves.
        self.mask = np.clip(cv2.warpAffine(mask_full, M, (E, E), flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=0), 0, 1)
        # any image edge the subject touches: extend (replicate) subject + pixels into the margin
        xl, xr = int(math.ceil(M[0, 2])) + 1, int(math.floor(s * W0 + M[0, 2])) - 1
        yt, yb = int(math.ceil(M[1, 2])) + 1, int(math.floor(s * H0 + M[1, 2])) - 1
        if 0 < yb < E and (mask_full[-1] > 0.5).mean() > 0.02:
            self.mask[yb:] = self.mask[yb - 1]; self.img[yb:] = self.img[yb - 1]
        if 0 < yt < E and (mask_full[0] > 0.5).mean() > 0.02:
            self.mask[:yt] = self.mask[yt]; self.img[:yt] = self.img[yt]
        if 0 < xl < E and (mask_full[:, 0] > 0.5).mean() > 0.02:
            self.mask[:, :xl] = self.mask[:, xl:xl + 1]; self.img[:, :xl] = self.img[:, xl:xl + 1]
        if 0 < xr < E and (mask_full[:, -1] > 0.5).mean() > 0.02:
            self.mask[:, xr:] = self.mask[:, xr - 1:xr]; self.img[:, xr:] = self.img[:, xr - 1:xr]
        if self.mask_cover >= 0.92 and center is None:
            self.mask[:] = 0  # whole image is "subject": animate as background only
        self.alpha = np.clip((self.mask - 0.05) / 0.9, 0, 1) ** 1.2
        m5 = self.alpha > 0.5
        if m5.any():
            yy, xx = np.nonzero(m5)
            self.sbox = (xx.min(), yy.min(), xx.max(), yy.max())
            self.anchored = yy.max() >= E - 3
            self.pivot = ((xx.min() + xx.max()) / 2, E if self.anchored else yy.mean())
        else:
            self.sbox, self.anchored, self.pivot = None, False, (E / 2, E / 2)

        # --- eyes in canvas coords
        self.eyes = []
        self.illustration = illustration_score(analysis.work)
        self.eye_kind = eye_kind if eye_kind in EYE_KINDS else "auto"
        if "blink" in self.subj:
            if eyes:
                for spec in eyes:
                    if not isinstance(spec, dict):
                        spec = dict(x=spec[0], y=spec[1], w=spec[2], h=spec[3],
                                    kind=spec[4] if len(spec) > 4 else "auto", space="out")
                    self.eyes.append(self._resolve_eye(spec, "manual"))
            if face_eyes and not self.eyes:
                for e in face_eyes:
                    p = e["poly"] / sc
                    p = np.stack([s * p[:, 0] + M[0, 2], s * p[:, 1] + M[1, 2]], 1)
                    self.eyes.append(dict(kind="photo", poly=p.astype(np.float32), src="face"))
            if not self.eyes and blink in ("auto", "pet"):
                for p in analysis.pet():
                    p = p / sc
                    p = np.stack([s * p[:, 0] + M[0, 2], s * p[:, 1] + M[1, 2]], 1)
                    self.eyes.append(dict(kind="photo", poly=p.astype(np.float32), src="pet"))
            if not self.eyes and blink in ("auto", "cartoon"):
                self.eyes = detect_cartoon_eyes(self.img, self.mask)
                for e in self.eyes:
                    e["src"] = "cartoon"
            # Detected photo-style eyes on flat artwork (MediaPipe often finds cartoon faces, but its eye
            # outlines are slivers there), or a forced eye_kind: re-fit them as clicked points.
            if self.eyes and self.eyes[0]["src"] != "manual":
                want, guessed = self.eye_kind, False
                if want == "auto" and self.illustration >= ILLUSTRATION_THRESHOLD and self.eyes[0]["kind"] == "photo":
                    want, guessed = "cartoon", True
                if want == "auto" and self.eyes[0]["src"] == "cartoon":
                    want, guessed = "cartoon", True   # re-grab: the detector's blobs are eroded / partial
                if want != "auto" and (guessed or any(e["kind"] != want for e in self.eyes)):
                    refit = []
                    for e in self.eyes:
                        x0, y0, x1, y1 = self._canvas_bbox(e)
                        r = self._resolve_eye(dict(x=(x0 + x1) / 2, y=(y0 + y1) / 2, kind=want, space="canvas"),
                                              e["src"], fallback_box=(x1 - x0, y1 - y0))
                        refit.append(r)
                    if not guessed or all(r["grab"]["found"] for r in refit):
                        self.eyes = refit
        # Manual eyes the crop leaves out entirely (zoomed in on one part of the image) can't blink; they are set
        # aside in eyes_off (source px) so an editor can keep them for when the crop moves back.
        self.eyes_off = []
        off = [e for e in self.eyes if not self._on_canvas(e)]
        self.eyes = [e for e in self.eyes if self._on_canvas(e)]
        for e in off:
            if "at" not in e:
                continue   # a detected eye outside the crop: nothing to keep
            ex, ey, ew, eh = e["at"]
            sx, sy = self.canvas_to_src(ex, ey)
            self.eyes_off.append(dict(x=round(float(sx), 1), y=round(float(sy), 1), w=round(ew / self.scale, 1),
                                      h=round(eh / self.scale, 1), kind=e["kind"]))
        for e in self.eyes:
            e.setdefault("spec", self._spec_from_canvas(e))
        self.eyes.sort(key=lambda e: e["spec"]["x"])   # eye 1 = leftmost in the image
        self.log(f"  crop center=({cx:.0f},{cy:.0f}) side={side:.0f}px  subject cover={self.mask_cover:.2f} "
                 f"anchored={self.anchored}  eyes={len(self.eyes)} ({self.eyes[0]['kind'] if self.eyes else 'none'})")
        if "blink" in self.subj and not self.eyes:
            self.log("  blink requested but no eyes found -> skipped")

        # --- layers
        dil = max(9, int(E * 0.06)) | 1
        self.bg = fill_background(self.img, self.mask, dil) if self.sbox else self.img.copy()
        # decontaminated foreground colour: at rest bg*(1-a) + fg*a reproduces the input exactly,
        # so soft edges carry no halo of the filled background (and no old-background fringe on new backdrops)
        a3 = self.alpha[..., None]
        self.fg = np.where(a3 > 0.02, np.clip((self.img - (1 - a3) * self.bg) / np.maximum(a3, 0.02), 0, 1), self.img).astype(np.float32)
        self.bg_fill = self.bg   # filled background before any backdrop replacement (for previews)
        self._prep_backdrop()
        self._prep_blink()
        self._prep_swirl()
        self._prep_sparkles()

    # ------------------------------------------------------------ eyes
    def src_to_canvas(self, x, y):
        return self.scale * x + self.M[0, 2], self.scale * y + self.M[1, 2]

    def canvas_to_src(self, x, y):
        return (x - self.M[0, 2]) / self.scale, (y - self.M[1, 2]) / self.scale

    def _on_canvas(self, e):
        if e["kind"] == "cartoon":
            return bool(e["mask"].any())
        (x0, y0), (x1, y1) = e["poly"].min(0), e["poly"].max(0)
        return x1 > 0 and y1 > 0 and x0 < self.E and y0 < self.E

    @staticmethod
    def _canvas_bbox(e):
        if e["kind"] == "cartoon":
            ys, xs = np.nonzero(e["mask"])
            return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)
        (x0, y0), (x1, y1) = e["poly"].min(0), e["poly"].max(0)
        return float(x0), float(y0), float(x1), float(y1)

    def _spec_from_canvas(self, e):
        x0, y0, x1, y1 = self._canvas_bbox(e)
        cx, cy = self.canvas_to_src((x0 + x1) / 2, (y0 + y1) / 2)
        return dict(x=round(float(cx), 1), y=round(float(cy), 1), w=round(float(x1 - x0) / self.scale, 1),
                    h=round(float(y1 - y0) / self.scale, 1), kind=e["kind"])

    def _resolve_eye(self, spec, src, fallback_box=None):
        """Eye spec -> internal eye (canvas coords). Cartoon eyes get a shape mask (grabbed from the image, or
        the box's ellipse when nothing is found); photo eyes get an ellipse outline for the eyelid slide."""
        out, E = self.out, self.E
        space = spec.get("space", "src")
        w, h = spec.get("w"), spec.get("h")
        if space == "out":
            cx, cy = self.off + spec["x"] * out, self.off + spec["y"] * out
            w, h = (w * out if w else None), (h * out if h else None)
        elif space == "canvas":
            cx, cy = spec["x"], spec["y"]
        else:
            cx, cy = self.src_to_canvas(spec["x"], spec["y"])
            w, h = (w * self.scale if w else None), (h * self.scale if h else None)
        box = (w, h) if w and h else None
        given = box is not None   # a sized eye keeps its position and size; only its shape is grabbed inside
        kind = spec.get("kind") or "auto"
        if kind == "auto":
            kind = self.eye_kind
        grabbed, info = (None, dict(found=False))
        if kind != "photo" or box is None:
            grabbed, info = grab_eye(self.img, cx, cy, out, box)
        if kind == "auto":
            flat = self.style in ("logo", "space") or self.illustration >= ILLUSTRATION_THRESHOLD
            kind = "cartoon" if grabbed is not None and flat else "photo"
        if box is None:
            if grabbed is not None:
                ys, xs = np.nonzero(grabbed)
                x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
                cx, cy, box = (x0 + x1) / 2, (y0 + y1) / 2, (float(x1 - x0), float(y1 - y0))
            else:
                box = fallback_box or (0.10 * out, 0.065 * out)
        if kind == "cartoon":
            m = grabbed
            if m is None:
                m = np.zeros((E, E), np.uint8)
                cv2.ellipse(m, (int(round(cx)), int(round(cy))), (max(int(box[0] / 2), 2), max(int(box[1] / 2), 2)),
                            0, 0, 360, 1, -1)
            e = dict(kind="cartoon", mask=m)
        else:
            e = eyes_from_boxes([(cx, cy, box[0], box[1])], E)[0]
        e["src"], e["grab"] = src, info
        e["at"] = (float(cx), float(cy), float(box[0]), float(box[1]))   # canvas centre and size, as resolved
        if given:
            sx, sy = self.canvas_to_src(cx, cy)
            e["spec"] = dict(x=round(float(sx), 1), y=round(float(sy), 1), w=round(float(box[0] / self.scale), 1),
                             h=round(float(box[1] / self.scale), 1), kind=kind)
        return e

    def eye_boxes(self):
        """Eyes as (cx, cy, w, h) fractions of the output square: the same format as --eyes / run(eyes=...)."""
        boxes = []
        for e in self.eyes:
            x0, y0, x1, y1 = self._canvas_bbox(e)
            boxes.append(tuple(round(float(v), 4) for v in (
                ((x0 + x1) / 2 - self.off) / self.out, ((y0 + y1) / 2 - self.off) / self.out,
                (x1 - x0) / self.out, (y1 - y0) / self.out)))
        return boxes

    def eye_specs(self):
        """Eyes in source-image px with their blink style: dicts x, y, w, h, kind (the --eyes-src format).
        These stay valid when the crop, zoom, size or style changes."""
        return [dict(e["spec"]) for e in self.eyes]

    def eye_source(self):
        """Which detector produced the eyes: manual, face, pet, cartoon or none."""
        if not self.eyes:
            return "none"
        return self.eyes[0].get("src", "face")

    def still(self, b=0.0):
        """Rest pose (t=0 background, untransformed subject) with the eyes closed by b in [0, 1], as uint8.
        Used for the editor's 'Test blink' without rendering or encoding a loop."""
        rgb = self.apply_blink(self.fg, b) if (self.eyes and b > 0) else self.fg
        bg = self.background(0.0)
        a3 = self.alpha[..., None]
        f = bg * (1 - a3) + rgb * a3 if self.sbox is not None else bg
        o = self.off
        return (np.clip(f[o:o + self.out, o:o + self.out], 0, 1) * 255 + 0.5).astype(np.uint8)

    # ------------------------------------------------------------ preparation
    def _prep_backdrop(self):
        """Logo-style: replace flat background with a soft gradient from the subject's colors."""
        if "backdrop" not in self.bgm:
            return
        outside = self.alpha < 0.05
        if outside.sum() and self.bg[outside].std() > 0.08:
            return  # background has real content; keep it
        E = self.E
        if self.sbox is not None:
            cols = self.img[self.alpha > 0.5]
            hsv = cv2.cvtColor(cols[None].astype(np.float32), cv2.COLOR_RGB2HSV)[0]
            sat = hsv[:, 1] > 0.25
            base = cols[sat].mean(0) if sat.sum() > 50 else cols.mean(0)
        else:
            base = np.array([0.3, 0.4, 0.9], np.float32)
        yy, xx = np.mgrid[:E, :E].astype(np.float32) / E
        r = np.sqrt((xx - 0.5) ** 2 + (yy - 0.45) ** 2)
        dark = base * 0.25
        light = np.clip(base * 0.8 + 0.25, 0, 1)
        g = np.clip(1 - r / 0.75, 0, 1)[..., None]
        self.bg = dark * (1 - g) + light * g

    def _prep_blink(self):
        """Photo eyes: an 'eyelid slide' (skin/fur above the eye is stretched down over it).
        Cartoon eyes: eye shapes are lifted off an inpainted face and squashed flat."""
        self.blink_layers = []
        if not self.eyes:
            return
        E = self.E
        cart = [e for e in self.eyes if e["kind"] == "cartoon"]
        if cart:
            holes = np.zeros((E, E), np.uint8)
            for e in cart:
                em = cv2.dilate(e["mask"], np.ones((5, 5), np.uint8))
                e["pivot_y"] = np.nonzero(em)[0].mean()
                e["alpha"] = np.clip(soft(em.astype(np.float32), 1.5) * 1.2, 0, 1)
                holes |= em
            u8 = (self.fg * 255).astype(np.uint8)
            self.closed = cv2.inpaint(u8, cv2.dilate(holes, np.ones((3, 3), np.uint8)), 7,
                                      cv2.INPAINT_TELEA).astype(np.float32) / 255
            self.eyeless_alpha = soft(holes.astype(np.float32), 2)
        for e in self.eyes:
            if e["kind"] != "photo":
                continue
            em = np.zeros((E, E), np.uint8)
            cv2.fillPoly(em, [np.round(e["poly"]).astype(np.int32)], 1)
            cols = np.nonzero(em.any(0))[0]
            if len(cols) < 3:
                continue
            xa, xb = cols.min(), cols.max()
            yu = np.array([np.nonzero(em[:, x])[0].min() if em[:, x].any() else np.nan for x in range(xa, xb + 1)])
            yl = np.array([np.nonzero(em[:, x])[0].max() if em[:, x].any() else np.nan for x in range(xa, xb + 1)])
            ok = ~np.isnan(yu)
            xs = np.arange(xa, xb + 1)
            yu = np.interp(xs, xs[ok], yu[ok]); yl = np.interp(xs, xs[ok], yl[ok])
            h = float((yl - yu).max())
            pad = int(0.35 * (xb - xa)) + 2
            X = np.arange(xa - pad, xb + pad + 1)
            YU = np.interp(X, xs, yu); YL = np.interp(X, xs, yl)
            open_h = np.interp(X, xs, yl - yu, left=0, right=0)
            wcol = cv2.GaussianBlur((open_h / max(h, 1)).astype(np.float32)[None], (0, 0), max(pad / 3, 1))[0]
            wcol = np.clip(wcol * 1.3, 0, 1)
            YU = cv2.GaussianBlur(YU.astype(np.float32)[None], (0, 0), 2)[0]
            YL = np.maximum(cv2.GaussianBlur(YL.astype(np.float32)[None], (0, 0), 2)[0], YU + 1)
            L = 0.9 * h
            y0, y1 = int(YU.min() - L - 2), int(YL.max() + 0.25 * h + 2)
            self.blink_layers.append(dict(kind="photo", X=X, YU=YU, YL=YL, w=wcol, L=L, h=h,
                                          box=(int(X[0]), max(y0, 0), int(X[-1]) + 1, min(y1, E))))

    def apply_blink(self, rgb, b):
        rgb = rgb.copy()
        if any(e["kind"] == "cartoon" for e in self.eyes):
            base = self.closed.copy()
            sq = max(1 - b, 0.04)
            for e in self.eyes:
                if e["kind"] != "cartoon":
                    continue
                py = e["pivot_y"]
                M = np.float32([[1, 0, 0], [0, sq, py * (1 - sq)]])
                ea = cv2.warpAffine(e["alpha"], M, (self.E, self.E))[..., None]
                ec = cv2.warpAffine(self.fg, M, (self.E, self.E))
                base = base * (1 - ea) + ec * ea
            ex = self.eyeless_alpha[..., None]
            rgb = rgb * (1 - ex) + base * ex
        for e in self.blink_layers:
            if e["kind"] != "photo":
                continue
            x0, y0, x1, y1 = e["box"]
            if x1 <= x0 or y1 <= y0:
                continue
            X = e["X"][: x1 - x0]
            yy = np.arange(y0, y1, dtype=np.float32)[:, None]
            YU, YL, w, L = e["YU"][None, : x1 - x0], e["YL"][None, : x1 - x0], e["w"][None, : x1 - x0], e["L"]
            ye = YU + b * 1.04 * (YL - YU)           # lid edge position
            top = YU - L
            inside = (yy >= top) & (yy <= ye)
            ys = np.where(inside, top + (yy - top) * (L / np.maximum(ye - top, 1)), yy)
            ys = yy + (ys - yy) * w                  # taper at the eye corners
            mx = np.repeat(X[None].astype(np.float32), y1 - y0, 0)
            patch = cv2.remap(rgb.astype(np.float32), mx, (ys + 0 * mx).astype(np.float32), cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)
            lash = np.exp(-((yy - ye) / max(0.07 * e["h"], 0.8)) ** 2) * 0.45 * b * w
            patch = patch * (1 - lash[..., None])
            rgb[y0:y1, x0:x1] = patch
        return rgb

    def _prep_swirl(self):
        if "swirl" not in self.bgm:
            return
        P = self.P; E = self.E
        self.sw_c = self.pivot if (self.sbox and not self.anchored) else (E / 2, E / 2)
        if self.sbox and not self.anchored:
            x0, y0, x1, y1 = self.sbox
            self.sw_c = ((x0 + x1) / 2, y1 - 0.1 * (y1 - y0))   # disk plane sits near subject base
        T = math.radians(P["swirl_tilt"]); RAT = P["swirl_ratio"]
        cx, cy = self.sw_c
        yy, xx = np.mgrid[:E, :E].astype(np.float32)
        dx, dy = xx - cx, yy - cy
        uu = dx * math.cos(T) + dy * math.sin(T)
        vv = (-dx * math.sin(T) + dy * math.cos(T)) / RAT
        R = np.sqrt(uu ** 2 + vv ** 2)
        NT = 1440
        RMAX = int(min(R.max() + 2, 2400))
        r = np.arange(RMAX, dtype=np.float32)[:, None]
        th = (np.arange(NT, dtype=np.float32) * 2 * np.pi / NT)[None, :]
        u = r * np.cos(th); v = RAT * r * np.sin(th)
        mx = (cx + u * math.cos(T) - v * math.sin(T)).astype(np.float32)
        my = (cy + u * math.sin(T) + v * math.cos(T)).astype(np.float32)
        Pp = cv2.remap(self.bg.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        pad = 240
        b = np.concatenate([Pp[:, -pad:], Pp, Pp[:, :pad]], 1)
        L = cv2.GaussianBlur(b, (0, 0), sigmaX=55, sigmaY=2)[:, pad:-pad]
        self.sw_L, self.sw_D = L, Pp - L
        self.sw_TH = ((np.arctan2(vv, uu) % (2 * np.pi)) * NT / (2 * np.pi)).astype(np.float32)
        self.sw_R = np.clip(R, 0, RMAX - 1).astype(np.float32)
        self.sw_NT = NT
        r0 = 0.3 * E
        sub_r = 0.0
        self.sw_ph = (math.radians(P["swirl_deg"]) * NT / (2 * np.pi)
                      * np.clip((r0 / np.maximum(R, 1)) ** 0.5, 0, 1.4)).astype(np.float32)
        w = np.clip((P["swirl_r1"] * self.out - R) / max((P["swirl_r1"] - P["swirl_r0"]) * self.out, 1), 0, 1)
        self.sw_w = (w * w * (3 - 2 * w))[..., None].astype(np.float32)
        self.sw_Lc = cv2.remap(L, self.sw_TH, self.sw_R, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    def _prep_sparkles(self):
        E, out, off = self.E, self.out, self.off
        rng = self.rng
        free = (self.alpha < 0.05)
        def pick(n, prefer_bright=False):
            pts = []
            g = self.bg.mean(2)
            tries = 0
            while len(pts) < n and tries < 5000:
                tries += 1
                x, y = rng.uniform(off + 0.04 * out, off + 0.96 * out, 2)
                if not free[int(y), int(x)]:
                    continue
                if prefer_bright and g[int(y), int(x)] > 0.6:
                    continue
                if any((x - a) ** 2 + (y - b) ** 2 < (0.09 * out) ** 2 for a, b in pts):
                    continue
                pts.append((x, y))
            return pts
        self.sparkles = [(x, y, rng.uniform(0.6, 1.0), int(rng.integers(1, 3)), rng.random())
                         for x, y in pick(self.P["n_sparkles"])]
        self.twinkles = [(x, y, rng.uniform(0.5, 1.0), int(rng.integers(1, 4)), rng.random())
                         for x, y in pick(28, True)]
        self.particles = [(rng.uniform(0, E), rng.uniform(0, E), rng.uniform(1.2, 3.2), int(rng.integers(1, 3)),
                           rng.random(), rng.uniform(0.35, 0.8)) for _ in range(self.P["n_particles"])]
        # sparkle tint from subject accent
        self.tint = np.array([1.0, 0.95, 0.8], np.float32) if self.style == "logo" else np.array([1, 1, 1], np.float32)
        # particle colour: white on dark backgrounds, a saturated accent on light ones
        lum = float(self.bg.mean())
        # white on dark backgrounds; warm gold on light ones (white would vanish)
        self.pcol = np.array([1.0, 1.0, 1.0] if lum < 0.45 else [1.0, 0.72, 0.25], np.float32)
        # pre-rendered sparkle sprite
        S = 41; c = S // 2
        yy, xx = np.mgrid[:S, :S].astype(np.float32) - c
        rr = np.sqrt(xx ** 2 + yy ** 2)
        cross = np.exp(-(yy ** 2) / 1.2) * np.exp(-np.abs(xx) / 6) + np.exp(-(xx ** 2) / 1.2) * np.exp(-np.abs(yy) / 6)
        self.sprite = np.clip(cross * 0.9 + np.exp(-rr ** 2 / 6) * 1.0, 0, 1.5).astype(np.float32)

    # ------------------------------------------------------------ per-frame
    def _add_sprite(self, f, x, y, size, val, color):
        sp = cv2.resize(self.sprite, None, fx=size, fy=size, interpolation=cv2.INTER_LINEAR) * val
        h, w = sp.shape
        x0, y0 = int(round(x - w / 2)), int(round(y - h / 2))
        X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x0 + w, f.shape[1]), min(y0 + h, f.shape[0])
        if X1 <= X0 or Y1 <= Y0:
            return
        sub = sp[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0, None]
        f[Y0:Y1, X0:X1] = 1 - (1 - f[Y0:Y1, X0:X1]) * (1 - np.clip(sub * color, 0, 1))  # screen blend

    def background(self, t, skip=()):
        """Background layer at time t. skip: motions to leave out (the live preview freezes the background
        without parallax and applies parallax itself)."""
        P, E, out = self.P, self.E, self.out
        f = self.bg.copy()
        tau = 2 * np.pi * t
        bgm = [m for m in self.bgm if m not in skip]
        if "swirl" in bgm:
            NT = self.sw_NT
            pa, pb = t % 1, (t + 0.5) % 1
            wa, wb = tri(pa), tri(pb)
            da = cv2.remap(self.sw_D, ((self.sw_TH + (pa - 0.5) * self.sw_ph) % NT), self.sw_R,
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            db = cv2.remap(self.sw_D, ((self.sw_TH + (pb - 0.5) * self.sw_ph) % NT), self.sw_R,
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            nrm = 1 / math.sqrt(wa * wa + wb * wb)
            f = self.bg * (1 - self.sw_w) + (self.sw_Lc + (wa * da + wb * db) * nrm) * self.sw_w
        if "hue" in bgm:
            hsv = cv2.cvtColor(np.clip(f, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)
            hsv[..., 0] = (hsv[..., 0] + P["hue_deg"] * math.sin(tau)) % 360
            f = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        if "twinkle" in bgm:
            for (x, y, s, k, ph) in self.twinkles:
                v = max(0.0, math.sin(2 * np.pi * (k * t + ph))) ** 3
                if v > 0.02:
                    self._add_sprite(f, x, y, 0.35 * s * out / 512, v, np.array([0.9, 0.95, 1.0], np.float32))
        if "parallax" in bgm:
            M = self.parallax_matrix(t)
            f = cv2.warpAffine(f.astype(np.float32), M, (E, E), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        if "particles" in bgm:
            ov = np.zeros((E, E), np.float32)
            for (x0, y0, r, k, ph, a) in self.particles:
                y = (y0 - k * t * E) % E
                x = x0 + 0.02 * E * math.sin(2 * np.pi * (t + ph))
                cv2.circle(ov, (int(x * 4), int(y * 4)), int(r * 4 * out / 512), a, -1, cv2.LINE_AA, 2)
            ov = cv2.GaussianBlur(ov, (0, 0), 1.2 * out / 512)[..., None]
            ov = np.clip(ov, 0, 1) * 0.9
            f = f * (1 - ov) + self.pcol * ov
        if "sparkles" in bgm:
            for (x, y, s, k, ph) in self.sparkles:
                v = max(0.0, math.sin(2 * np.pi * (k * t + ph))) ** 2
                if v > 0.02:
                    self._add_sprite(f, x, y, (0.6 + 0.6 * v) * s * out / 512, v, self.tint)
        return f

    # Motion math. preview.js mirrors motion(), subject_matrix(), parallax_matrix() and blink_amount()
    # line for line (tests/test_live.py checks they agree), so keep the two in step when changing either.
    def motion(self, t):
        """Subject pose at loop time t in [0, 1]: (angle_deg, scale, tx, ty) about self.pivot."""
        P, out = self.P, self.out
        tau = 2 * np.pi * t
        ang, sc, tx, ty = 0.0, 1.0, 0.0, 0.0
        if "float" in self.subj:
            ty = -P["float_amp"] * out * (1 - math.cos(tau)) / 2   # rises from the rest pose and settles back (never sinks, so the area above the subject is never uncovered)
        if "breathe" in self.subj:
            sc = 1 + P["breathe_amp"] * (0.5 - 0.5 * math.cos(2 * tau)) if self.style == "logo" else 1 + P["breathe_amp"] * math.sin(tau) * 0.6
        if "tilt" in self.subj:
            ang = P["tilt_deg"] * math.sin(tau + 0.6)
        return ang, sc, tx, ty

    def subject_matrix(self, t):
        """2x3 affine (cv2 convention, canvas px) applied to the subject layer at time t, or None at rest."""
        ang, sc, tx, ty = self.motion(t)
        if (ang, sc, tx, ty) == (0.0, 1.0, 0.0, 0.0):
            return None
        px, py = self.pivot
        return affine_about(px, py, ang, sc, tx, ty)

    def parallax_matrix(self, t):
        """2x3 affine for the parallax background drift at time t (zoomed a touch so shifted edges never show)."""
        P, E = self.P, self.E
        A = P["parallax_amp"] * self.out
        tau = 2 * np.pi * t
        M = cv2.getRotationMatrix2D((E / 2, E / 2), 0, 1.0 + 2 * P["parallax_amp"])
        M[0, 2] += A * math.cos(tau); M[1, 2] += 0.6 * A * math.sin(tau)
        return M

    def blink_amount(self, t):
        u = (t - self.P["blink_at"]) / self.P["blink_len"]
        return float(np.sin(np.pi * u) ** 0.6) if 0 < u < 1 else 0.0

    def subject(self, t):
        """Return (rgb, alpha) of the subject layer at time t, already transformed."""
        P = self.P
        rgb, a = self.fg, self.alpha
        if self.eyes and "blink" in self.subj:
            b = self.blink_amount(t)
            if b > 0:
                rgb = self.apply_blink(rgb, b)
        M = self.subject_matrix(t)
        if M is not None:
            rgb = cv2.warpAffine(rgb.astype(np.float32), M, (self.E, self.E), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101 if self.anchored else cv2.BORDER_CONSTANT)
            a = cv2.warpAffine(a, M, (self.E, self.E), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE if self.anchored else cv2.BORDER_CONSTANT)
        if "sweep" in self.subj:
            u = (t - 0.15) / 0.45   # sweep happens in the first half of the loop
            if 0 < u < 1:
                E = self.E
                yy, xx = np.mgrid[:E, :E].astype(np.float32) / E
                d = (xx + 0.45 * yy) / 1.45
                pos = -0.15 + 1.3 * u
                band = np.exp(-((d - pos) / 0.06) ** 2) * P["sweep_strength"]
                rgb = 1 - (1 - rgb) * (1 - band[..., None])
        return rgb, a

    def frame(self, k):
        return self.frame_t((k % self.N) / self.N)

    def frame_t(self, t):
        bg = self.background(t)
        if self.sbox is not None:
            rgb, a = self.subject(t)
            a3 = a[..., None]
            f = bg * (1 - a3) + rgb * a3
        else:
            f = bg
        o = self.off
        f = np.clip(f[o:o + self.out, o:o + self.out], 0, 1)
        return (f * 255 + 0.5).astype(np.uint8)

    def render(self):
        return [self.frame(k) for k in range(self.N)]
