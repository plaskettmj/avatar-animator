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
  * Cartoons: a heuristic that looks for two similar high-contrast blobs inside the subject. It can miss unusual
    designs or misfire on two dots or buttons; `--blink off` disables it.
* **Swirl:** a flow crossfade rather than a rigid rotation, so the texture pulses slightly in sharpness. Tilted
  disks need `swirl_ratio`/`swirl_tilt` set by hand.
* **Background fill:** OpenCV Telea inpainting at quarter resolution. It's fine for the small reveals the motions
  make, but large moves show smeary fill.
* **Framing:** a heuristic crop (face box ×2.6, otherwise the subject bbox); `--center` and `--zoom` override it.
  Subjects that touch the image edge get their edge pixels replicated, and long replications can look streaky.
* **Segmentation edges:** there's no colour decontamination, so a faint fringe of the old background can move with
  the subject on high-contrast edges.
* **Motion is rigid:** float, tilt and breathe, plus blink. There is no mesh warping or gaze movement yet.

## Roadmap
1. Live preview in the browser: send the layers once (subject PNG, filled background, eye data) and animate them
   client-side with Canvas/WebGL using the same periodic motion functions. Sliders then update instantly, and
   encoding happens only on export.
2. Manual corrections: drag the crop square, paint the mask, click to place eyes, draw the swirl ellipse.
3. Permissive model upgrades: BiRefNet full/HR (MIT) for hair, LaMa (Apache-2.0) for background fill, MediaPipe
   iris/blendshapes for gaze, and a permissively licensed pet keypoint model for dogs.
4. Export options: APNG, MP4/WebM loops, a small sidebar-optimised variant with stronger motion, and a size-budget
   search (fps × size × lossy) instead of the fixed ladder.
5. Tests: a golden-image suite (faces, pets and logo edge cases) checking seam, size budget and per-motion
   visibility at small sizes.
