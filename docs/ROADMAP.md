# Known limitations and roadmap

## Known limitations
* **Speed:** about 30–90 s per avatar on an 8-core CPU. Most of that is GIF encoding (palette, plus gifsicle,
  sometimes run several times to fit the budget) and BiRefNet on CPU (5–30 s). Rendering the frames takes 5–10 s.
* **GIF size vs. quality:** photos where every pixel moves (parallax) don't fit in 4.5 MB at 512 px and 20 fps,
  so the encoder shrinks them to 448 or 384 px. WebP stays at 512 px and is 2–3× smaller; prefer it where it's
  supported.
* **Blink:**
  * Faces: MediaPipe landmarks plus an eyelid slide. It looks natural on frontal faces. Profiles, sunglasses and
    tiny faces get no blink, and the tool skips it without failing.
  * Pets: a cat-face Haar cascade plus a colour-blob eye fit. It works on frontal cats with distinct eyes. There
    is no dog detector, so use `--eyes` for dogs.
  * Cartoons: a heuristic that looks for two similar high-contrast or strongly coloured blobs inside the subject.
    It can miss unusual designs or misfire on two dots, buttons or laces; place the eyes by hand or use
    `--blink off`. Manual and detected eyes on flat artwork use the cartoon squash blink.
  * Clicked eyes: the shape is found from colour contrast with the surrounding face. Low-contrast eyes (closed,
    painted, photographic) fall back to an ellipse you size by hand. Eyes outside the cut-out don't blink
    visibly, because blinking only changes the subject layer.
* **Swirl:** a flow crossfade rather than a rigid rotation, so the texture pulses slightly in sharpness. Tilted
  disks need `swirl_ratio`/`swirl_tilt` set by hand.
* **Background fill:** OpenCV Telea inpainting at quarter resolution. It's fine for the small reveals the motions
  make, but large moves show smeary fill. The fill band around the subject is visible even at rest, so parts the
  cut-out misses (a headset next to a head) look smeared; use `--model birefnet-general` or, later, mask painting.
* **Framing:** a heuristic crop (face box ×2.6, otherwise the subject bbox); `--center` and `--zoom`, or dragging
  the crop in the editor, override it. The crop square always stays inside the image (at most the short side),
  so a style that wants more room around the subject than the image has (the logo style on app icons) gets the
  whole image rather than a smeared border; a subject wider than the image's short side is cut at the sides.
  Subjects that touch the image edge get their edge pixels replicated into the motion margin, and long
  replications can look streaky.
* **Segmentation edges:** there's no colour decontamination, so a faint fringe of the old background can move with
  the subject on high-contrast edges.
* **Live preview:** particles, sparkles, twinkle, swirl, hue and sweep are shown still in the editor's preview
  (Export animates them); blink closes in 8 steps rather than per frame; edge pixels can differ by a few levels
  from the export (browser vs OpenCV resampling).
* **Motion is rigid:** float, tilt and breathe, plus blink. There is no mesh warping or gaze movement yet.

## Roadmap
1. Live preview in the browser (done). Upload a photo, see the cropped avatar animating within seconds, fix the
   eyes, crop and motion, and encode only on Export.
   * Step 1 (done): the slow prepare step (mask, crop, fill, eyes) is split from export and cached per output
     folder / web job (`--prepare-only`, `/prepare`, `/export`).
   * Step 2 (done): eye editor. Chat path (`prep/grid.png` + `--eyes-at`) and browser editor (`/editor`: click
     to add, drag, resize, per-eye cartoon/photo blink, test blink, cut-out view and model switch). Eyes are
     stored in source-image pixels; clicked points grab the eye shape.
   * Step 3 (done): live preview. `preview.js` animates float, breathe, tilt, blink and parallax on a Canvas
     from the `/preview` layers with the renderer's own motion math (checked against it under node), so the
     sliders update instantly and encoding happens only on Export. Swirl, hue, twinkle, particles, sparkles and
     sweep are shown still (as at t=0) and labelled. Blink is previewed in 8 steps.
   * Step 4 (done): crop drag and zoom on the live preview, clamped to the image so the border is never
     smeared (the auto crop is clamped too); re-cropping reuses the cached cut-out. The full BiRefNet session
     runs without ONNX Runtime's CPU memory arena, so repeated full-model Prepares in one web UI process stay
     at about 8 GB peak instead of growing to 14 GB.
   * Step 5 (done): docs and the bot flow. The README describes how a bot offers eye placement when
     `report.json` shows no eyes: the chat grid by default, the browser editor on the bot's own desktop as an
     option, plus the live-preview sliders and crop drag. A golden test (`tests/golden_manual_eyes.py`, run by
     `run_tests.sh`) renders a sample where no eyes are detected, places them with `--eyes` / `eyes=`, and
     checks that the encoded blink frame changes inside the eyes and nowhere else.
2. Manual corrections: paint the mask, draw the swirl ellipse. (Eye placement and crop drag shipped with the live
   preview.)
3. Permissive model upgrades: BiRefNet full/HR (MIT) for hair, LaMa (Apache-2.0) for background fill, MediaPipe
   iris/blendshapes for gaze, and a permissively licensed pet keypoint model for dogs.
4. Export options: APNG, MP4/WebM loops, a small sidebar-optimised variant with stronger motion, and a size-budget
   search (fps × size × lossy) instead of the fixed ladder.
5. Tests: grow the golden suite (started with the manual-eye blink test) to faces, dogs, profiles and logo edge
   cases, checking seam, size budget and per-motion visibility at small sizes. The sample set has no dog or
   side-profile photo yet; a CC0 one would let the golden test use a real detector miss instead of limiting the
   cat to the face detector.
