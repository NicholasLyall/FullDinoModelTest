# Open-Vocabulary Video Detector — SAM 3.1 (state of the art, localhost)

A tiny local web app: drop in a video, type what you want to find (e.g. `car`),
and it draws labeled bounding boxes on the video. Comma-separated prompts find
multiple things (e.g. `car, person, truck`).

Uses **SAM 3.1** — the current state-of-the-art open-vocabulary detection model.
It produces segmentation masks internally; **we only render the bounding boxes
and ignore the masks.** That's fine and expected.

This file is the build spec for Claude Code. Build it as described.

---

## Goal / UX

1. App runs on `http://localhost:8000`.
2. A video file sits in `./input/` (the user drops it there). The page can also
   accept an upload.
3. The page shows the video / processed frames with bounding boxes drawn on top.
4. At the bottom is a **text box**. The user types a prompt like:
   - `car` -> box every car
   - `car, person, truck` -> box all three
   Each comma-separated term is a separate concept to detect.
5. Each box is labeled with its class name + confidence, and the page shows a
   running **count** of boxes per class.
6. Because SAM 3 tracks objects across frames, boxes should keep a stable ID/color
   per object through the video (don't flicker as independent per-frame detections).

That's it. Keep it simple. **Boxes only — do not draw masks.**

---

## Model: SAM 3.1 (`facebook/sam3`)

SAM 3.1 is the current SOTA for open-vocabulary detection. On open-vocabulary
box detection it roughly doubles the older grounded detectors (Grounding DINO,
DINO-X, OWLv2) and beats Gemini. It accepts short noun-phrase text prompts and
detects + tracks all matching instances across a video.

**Run it via the Ultralytics package** — SAM 3 is integrated there and it handles
video tracking for you, which is the simplest path. Pseudocode:

```python
from ultralytics import SAM

model = SAM("sam3.pt")           # SAM 3.1 checkpoint
# prompt is comma-separated from the UI -> list of concepts
results = model(
    "input/clip.mp4",
    prompts=["car", "person", "truck"],   # see prompt handling below
    # we only use results[*].boxes ; ignore masks
)
```

If the installed Ultralytics API differs, follow its current SAM 3 docs — the
contract we need is: **text/concept prompt in, per-frame boxes + class + score +
track id out.** Render only the boxes.

### Prompt handling

UI takes **comma-separated** text. Split it into a list of concept strings:

```python
def parse_prompt(user_text: str) -> list[str]:
    return [p.strip().lower() for p in user_text.split(",") if p.strip()]
```

Pass that list to SAM 3 as its concept prompts. Each concept is detected/tracked
separately so per-class counts are easy.

### Threshold

Expose a confidence/score threshold (default ~0.4) as a slider so the user can
lower it to catch more boxes (more detections, more false positives).

---

## Architecture

- **Backend:** Python 3.12 + FastAPI (or Flask). PyTorch + Ultralytics + OpenCV.
- **Frontend:** one static HTML page, vanilla JS. No framework.
- **Inference:** run on GPU. For each (kept) frame, draw boxes with OpenCV
  (`cv2.rectangle` + `cv2.putText`), then either:
  - **Option A (do this first):** process the whole video for the given prompt,
    write `output/annotated.mp4`, play it in the page.
  - **Option B (optional):** stream processed frames live (MJPEG/websocket).
- **Render only boxes.** Do not overlay segmentation masks.

### Endpoints (suggested)

- `GET /` -> the page.
- `POST /detect` -> `{ "prompt": "car, person", "video": "input/clip.mp4",
  "score_threshold": 0.4, "frame_stride": 1 }`
  -> runs detection, returns path to annotated video + per-class counts.
- `GET /video/annotated.mp4` -> serves the result.

---

## Hardware notes (read these — the target machine is specific)

Target: **Ubuntu 26, RTX 5070 Ti (Blackwell, 16 GB VRAM).**

> Originally built/verified on an RTX 5050 Laptop (Blackwell, 8 GB). The 5070 Ti is
> the **same Blackwell architecture (`sm_120`)**, just a faster desktop chip with more
> VRAM, so nothing about the model/CUDA setup changes — it's the same cu128 build.
> The app's settings are tuned for 8 GB (FP16, `imgsz=1024`) and run unchanged on the
> 5070 Ti with plenty of headroom to spare.

1. **PyTorch must be a CUDA 12.8+ build.** Blackwell is compute capability
   `sm_120`; older wheels throw *"no kernel image available for execution on the
   device."* The `setup.sh` script installs the cu128 build for you. Manual:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   ```
   Verify: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`

2. **Python version on Ubuntu 26.** Ubuntu 26 may default to a Python newer than the
   pinned `torch 2.11.0+cu128` wheels support (cp39–cp313). `setup.sh` prefers
   `python3.12`/`python3.13` automatically. If the default is too new and install
   fails, `sudo apt install python3.12 python3.12-venv` and re-run.

3. **VRAM: 16 GB is plenty.** SAM 3.1 runs ~4 GB in FP16; YOLOE-26 is tiny. No OOM
   concern on this card. If you want to catch smaller/distant objects you *can* raise
   `imgsz` (e.g. 1280) for the extra headroom — but the defaults are fine. If you ever
   do hit OOM (e.g. very high-res source), lower input resolution or raise `frame_stride`.

4. **Long-clip Xid 120 GSP-panic** was a sustained-load risk noted for the laptop GPU;
   a desktop 5070 Ti has more power/cooling headroom, so it's less of a concern. No
   checkpoint/resume is built in by design — `frame_stride` and the score threshold
   are ordinary UX controls, not crash mitigations.

---

## Setup

One command — creates the venv, installs the cu128 PyTorch + all pinned deps, and
sanity-checks the GPU:

```bash
bash setup.sh
```

Manual equivalent, if you prefer:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
# SAM 3 weights: request access at https://huggingface.co/facebook/sam3,
# download sam3.pt, and place it in the project root.
```

## Run

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
# open http://localhost:8000, drop a video in ./input/, type a prompt, go
```

---

## Acceptance criteria

- [ ] Page loads on localhost with a video area and a prompt text box at the bottom.
- [ ] Typing `car` draws boxes around cars in the video.
- [ ] Typing `car, person, truck` detects all three, each labeled.
- [ ] Each box shows class + confidence; page shows per-class counts.
- [ ] Boxes are stable per object across frames (uses SAM 3 tracking), not flickery.
- [ ] Runs on the GPU in FP16 (confirm CUDA is used, not CPU).
- [ ] `frame_stride` and score threshold are adjustable.
- [ ] Detection state is checkpointed so a mid-run crash can resume.
- [ ] Only bounding boxes are rendered — no masks drawn.

## Folder layout

```
.
├── README.md          # this file
├── app.py             # FastAPI backend + inference
├── static/index.html  # the page
├── input/             # user drops the video here
└── output/            # annotated.mp4 + checkpoint files written here
```

---

### Note
If SAM 3.1 is too slow on the 5050 for a given clip, the fast fallback is
Grounding DINO (`IDEA-Research/grounding-dino-base`, boxes-native, sub-1 GB) with
the same comma-prompt UX. Only add this if needed; SAM 3.1 is the SOTA target.
