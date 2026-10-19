# Credits and third-party licenses

The code in this repository is MIT-licensed (see `LICENSE`). It does not bundle any third-party code or model
weights. Python packages are installed by pip from PyPI, and model weights are downloaded on first run into
`./models/` (git-ignored). Each component keeps its own license:

## Models (downloaded on first use)

| Model | Used for | License | Source |
|---|---|---|---|
| BiRefNet general, lite (Swin-tiny, epoch 232), ONNX export by rembg (default) | subject mask | MIT, © 2024 ZhengPeng | code: <https://github.com/ZhengPeng7/BiRefNet> (LICENSE: MIT); weights: <https://huggingface.co/ZhengPeng7/BiRefNet_lite> (license: mit) |
| U²-Net (`--model u2net`, optional) | alternate subject mask | Apache-2.0 | <https://github.com/xuebinqin/U-2-Net> |
| MediaPipe Face Landmarker bundle (`face_landmarker.task`: BlazeFace short-range, FaceMesh-V2, Blendshape V2) | face and eye landmarks for blinking | Apache-2.0 (stated on each model card) | <https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker> |
| `haarcascade_frontalcatface_extended.xml` by Joseph Howse | cat-face detection | BSD-3-Clause, © 2014-2016 Joseph Howse (Nummist Media) (license text in the file header) | <https://github.com/opencv/opencv/tree/4.x/data/haarcascades> |

The extra `--model` choices `birefnet-general` (BiRefNet, MIT) and `isnet-general-use` (DIS / IS-Net,
Apache-2.0) are also permissive. They are not used unless you select them.

## Libraries (installed from PyPI)

| Library | Used for | License |
|---|---|---|
| [rembg](https://github.com/danielgatis/rembg) | background-removal wrapper, model download | MIT, © 2020 Daniel Gatis |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | model inference | MIT |
| [OpenCV](https://github.com/opencv/opencv) (`opencv-python-headless`, `opencv-contrib-python`) | inpainting, warps, cascade detection | Apache-2.0 (OpenCV 4.5+) |
| [MediaPipe](https://github.com/google-ai-edge/mediapipe) | Face Landmarker runtime | Apache-2.0 |
| [NumPy](https://numpy.org) | arrays | BSD-3-Clause |
| [Pillow](https://python-pillow.org) | image I/O, GIF/WebP encoding | MIT-CMU (HPND) |

Packages that rembg and mediapipe pull in, such as scikit-image, SciPy, PyMatting, pooch and matplotlib, are
under their own permissive licenses (BSD, MIT, PSF-style).

## Optional external tool (not bundled)

| Tool | Used for | License |
|---|---|---|
| [gifsicle](https://www.lcdf.org/gifsicle/) | shrinking GIFs (`-O3 --lossy`) | GPL-2.0. It is run as a separate program if it is installed and is never distributed or linked here. Without it, GIFs are simply larger. |

## Deliberately excluded

`bria-rmbg` (RMBG weights are non-commercial) and any other rembg model whose weights are not under a permissive
license.

## Sample images

See `test-images/SOURCES.md`. Every sample is CC0 or an original graphic.
