"""
Open-Vocabulary Video Detector — SAM 3 (localhost).

Drop a video in ./input/, type comma-separated concepts (e.g. "car, person, truck"),
and the app draws labeled bounding boxes on the video using SAM 3's open-vocabulary
detector + native video tracking. Boxes only — masks are ignored.

Engine: ultralytics SAM3VideoSemanticPredictor (stream=True). SAM 3 tracks each
object across frames, so every box carries a stable track id (and therefore a stable
colour) for its lifetime in the clip.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import (
    FileResponse, HTMLResponse, JSONResponse, StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

# --------------------------------------------------------------------------------------
# Paths / config
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
STATIC_DIR = ROOT / "static"
MODEL_PATH = ROOT / "sam3.pt"  # gated weights from huggingface.co/facebook/sam3
# SOTA open-weights fallback: YOLOE-26 (2026 gen). Ungated, auto-downloads, runs on 8 GB.
# Used automatically when sam3.pt is absent; SAM 3 takes over the moment it appears.
YOLOE_MODEL = "yoloe-26x-seg.pt"  # biggest/most accurate; drop to -26l/-26m for more speed

for d in (INPUT_DIR, OUTPUT_DIR, STATIC_DIR):
    d.mkdir(exist_ok=True)

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def parse_prompt(user_text: str) -> list[str]:
    """Comma-separated UI text -> list of lowercase concept strings."""
    return [p.strip().lower() for p in user_text.split(",") if p.strip()]


# --------------------------------------------------------------------------------------
# Stable per-track colours
# --------------------------------------------------------------------------------------
def color_for_id(track_id: int) -> tuple[int, int, int]:
    """Deterministic, well-spread BGR colour for a track id (golden-ratio hue)."""
    h = (track_id * 0.61803398875) % 1.0
    hsv = np.uint8([[[int(h * 179), 200, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def draw_box(frame, xyxy, label: str, color: tuple[int, int, int]) -> None:
    """Draw one bounding box + label. Boxes only — never any mask."""
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    ytop = max(0, y1 - th - base - 2)
    cv2.rectangle(frame, (x1, ytop), (x1 + tw + 4, ytop + th + base + 2), color, -1)
    cv2.putText(
        frame, label, (x1 + 2, ytop + th + 1),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
    )


def _class_name(names, c) -> str:
    """Resolve a class index to its name. YOLOE gives a dict {idx: name}; SAM 3 gives a
    list [name, ...] — handle both (the dict-only path silently mislabels SAM as '0/1/2')."""
    i = int(c)
    if isinstance(names, dict):
        return names.get(i, str(i))
    if isinstance(names, (list, tuple)):
        return names[i] if 0 <= i < len(names) else str(i)
    return str(i)


def annotate_result(result, concepts: list[str], seen_ids: dict[str, set]):
    """Draw all boxes for one engine Result onto a fresh frame copy.

    Updates `seen_ids` (cumulative unique track ids per class) in place and returns
    (annotated_frame, current_counts). Shared by the processed pass and the live stream.
    """
    frame = result.orig_img.copy()
    names = result.names
    current: dict[str, int] = {c: 0 for c in concepts}

    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        ids = (boxes.id.cpu().numpy().astype(int)
               if boxes.id is not None else np.arange(len(xyxy)))
        for box, c, s, tid in zip(xyxy, cls, confs, ids):
            cname = _class_name(names, c)
            current[cname] = current.get(cname, 0) + 1
            seen_ids.setdefault(cname, set()).add(int(tid))
            draw_box(frame, box, f"{cname} #{int(tid)} {s:.2f}", color_for_id(int(tid)))
    return frame, current


def draw_hud(frame, concepts: list[str], current: dict[str, int],
             seen_ids: dict[str, set], fps: float | None = None) -> None:
    """Overlay a translucent live counter (per-class now/total + fps) on the frame."""
    lines = []
    if fps is not None:
        lines.append(f"LIVE  {fps:4.1f} fps")
    for c in concepts:
        lines.append(f"{c}: {current.get(c, 0)} now / {len(seen_ids.get(c, set()))} total")
    if not lines:
        return
    pad, lh = 8, 24
    tw = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0] for t in lines)
    bw, bh = tw + pad * 2, lh * len(lines) + pad
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (bw, bh), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (pad, pad + lh * (i + 1) - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


# --------------------------------------------------------------------------------------
# SAM 3 model (loaded lazily, once, reused across jobs)
# --------------------------------------------------------------------------------------
class Sam3Engine:
    """Thin wrapper around the SAM 3 video-semantic predictor."""

    name = "sam3"
    label = "SAM 3 (native video tracking)"
    # VRAM scales with the number of CONCURRENTLY tracked objects: every live object
    # carries its own SAM2-style tracker state (memory features + object pointers),
    # and memory-attention runs over all of them each frame. Upstream hardcodes
    # max_num_objects=10000 ("no limit") in __init__, silently discarding its own
    # constructor param, so a busy street scene balloons unbounded — this cap is the
    # actual OOM fix. Lowest-score new detections get dropped once the cap is hit
    # (upstream logs "hitting max_num_objects" when that happens).
    MAX_TRACKED_OBJECTS = 64

    def __init__(self) -> None:
        self._predictor = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return MODEL_PATH.exists()

    def _load(self):
        if self._predictor is not None:
            return self._predictor
        if not self.available:
            raise FileNotFoundError(
                f"SAM 3 weights not found at {MODEL_PATH}. Request access at "
                "https://huggingface.co/facebook/sam3, download sam3.pt, and place it "
                "in the project root."
            )
        # Imported here so the web server boots even without torch/weights present.
        from ultralytics.models.sam import SAM3VideoSemanticPredictor

        overrides = dict(
            conf=0.4,
            task="segment",
            mode="predict",
            imgsz=1008,  # SAM 3's native input (72 × 14-px ViT patches)
            model=str(MODEL_PATH),
            half=True,        # FP16 — fits the 8 GB Blackwell card
            device=0,         # GPU
            save=False,
            verbose=False,
        )
        self._predictor = SAM3VideoSemanticPredictor(overrides=overrides)
        # Cap concurrent tracked objects (attribute is live; the ctor param is dead
        # code upstream — see MAX_TRACKED_OBJECTS comment above).
        self._predictor.max_num_objects = self.MAX_TRACKED_OBJECTS
        return self._predictor

    def stream(self, video_path: str, concepts: list[str], conf: float, stride: int):
        """Yield (result, predictor) per processed frame. Serialized: one job at a time."""
        with self._lock:
            predictor = self._load()
            predictor.args.conf = float(conf)
            predictor.args.vid_stride = int(max(1, stride))
            # SAM 3 names map class index -> concept string for this run.
            for result in predictor(source=video_path, text=concepts, stream=True):
                yield result


class YoloeEngine:
    """SOTA open-weights fallback: YOLOE-26 with text prompts + built-in tracker (stable IDs)."""

    name = "yoloe"
    label = "YOLOE-26 (open-weights SOTA fallback)"
    available = True  # weights auto-download (ungated)

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            from ultralytics import YOLOE

            self._model = YOLOE(YOLOE_MODEL)
        return self._model

    def stream(self, video_path: str, concepts: list[str], conf: float, stride: int):
        """Yield per-frame Results. .track keeps persistent IDs across the clip."""
        with self._lock:
            model = self._load()
            # Set the open-vocabulary concepts for this run.
            model.set_classes(concepts, model.get_text_pe(concepts))
            # FP32: YOLOE-x is tiny (fits easily) and its FP32 text embeddings would
            # otherwise clash with a half-precision model (Half != float dtype error).
            for result in model.track(
                source=video_path, stream=True, persist=True,
                conf=float(conf), vid_stride=int(max(1, stride)),
                half=False, device=0, verbose=False,
            ):
                yield result


SAM3 = Sam3Engine()
YOLOE_ENGINE = YoloeEngine()


def select_engine(force: str | None = None):
    """Pick the engine: SAM 3 when its weights exist, else the YOLOE-26 fallback.

    `force` ('sam3' | 'yoloe') overrides auto-selection. Returns None only if SAM 3
    is forced but its weights are missing.
    """
    if force == "yoloe":
        return YOLOE_ENGINE
    if force == "sam3":
        return SAM3 if SAM3.available else None
    return SAM3 if SAM3.available else YOLOE_ENGINE


# --------------------------------------------------------------------------------------
# Job tracking (in-memory; one job at a time via the engine lock)
# --------------------------------------------------------------------------------------
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def _set(job_id: str, **kw) -> None:
    with JOBS_LOCK:
        JOBS[job_id].update(kw)


def run_detection(job_id: str, engine, video_path: Path, concepts: list[str],
                  conf: float, stride: int) -> None:
    """Background worker: stream the chosen engine over the video, draw boxes, write
    annotated.mp4. If the engine crashes mid-stream, the frames processed so far are still
    encoded and saved, so a partial result stays watchable."""
    tmp_path = OUTPUT_DIR / f"_tmp_{job_id}.mp4"
    final_path = OUTPUT_DIR / "annotated.mp4"
    state = {"writer": None, "processed": 0}

    def finalize() -> bool:
        """Release the writer and re-encode mp4v -> browser-friendly H.264.
        Returns True if a playable file was produced (>=1 frame written)."""
        w = state["writer"]
        if w is not None:
            w.release()
            state["writer"] = None
        if not tmp_path.exists() or state["processed"] == 0:
            return False
        if shutil.which("ffmpeg"):
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(tmp_path),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                 str(final_path)],
                check=True, capture_output=True,
            )
            tmp_path.unlink(missing_ok=True)
        else:
            tmp_path.replace(final_path)
        return True

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        out_fps = max(1.0, src_fps / max(1, stride))

        seen_ids: dict[str, set] = {c: set() for c in concepts}
        _set(job_id, status="running", total=max(1, total // max(1, stride)), backend=engine.name)

        for result in engine.stream(str(video_path), concepts, conf, stride):
            frame, current = annotate_result(result, concepts, seen_ids)
            if state["writer"] is None:
                h, w = frame.shape[:2]
                state["writer"] = cv2.VideoWriter(
                    str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))
            state["writer"].write(frame)
            state["processed"] += 1
            counts = {c: {"current": current.get(c, 0), "total": len(seen_ids.get(c, set()))}
                      for c in concepts}
            _set(job_id, processed=state["processed"], counts=counts)

        if state["processed"] == 0:
            raise RuntimeError("No frames were processed.")
        _set(job_id, status="encoding")
        finalize()
        _set(job_id, status="done", output="/video/annotated.mp4", finished=time.time())
    except Exception as e:  # noqa: BLE001 — report any failure to the UI
        import traceback
        tb = traceback.format_exc()
        print(tb, flush=True)  # full traceback to the server console
        loc = next((l.strip() for l in reversed(tb.strip().splitlines())
                    if l.strip().startswith("File ")), "")
        # Save whatever was processed so it's still watchable.
        saved = False
        try:
            saved = finalize()
        except Exception as enc_err:  # noqa: BLE001
            print("partial-encode failed:", enc_err, flush=True)
        msg = (f"{type(e).__name__}: {e}  [{loc}]  "
               f"(processed {state['processed']} frames)")
        upd = {"status": "error", "error": msg + (" — partial video saved" if saved else "")}
        if saved:
            upd["output"] = "/video/annotated.mp4"
        _set(job_id, **upd)


def live_mjpeg(engine, video_path: Path, concepts: list[str], conf: float, stride: int):
    """Generator: stream annotated frames as MJPEG (multipart/x-mixed-replace).

    Boxes + a live HUD are drawn straight onto each JPEG, so the browser just points an
    <img> at this endpoint and watches detections appear as fast as the GPU produces them.
    No final file is written — this is the live-preview path (YOLOE).
    """
    seen_ids: dict[str, set] = {c: set() for c in concepts}
    last = time.time()
    fps = 0.0
    try:
        for result in engine.stream(str(video_path), concepts, conf, stride):
            frame, current = annotate_result(result, concepts, seen_ids)
            now = time.time()
            dt = now - last
            last = now
            if dt > 0:
                inst = 1.0 / dt
                fps = inst if fps == 0.0 else 0.7 * fps + 0.3 * inst
            draw_hud(frame, concepts, current, seen_ids, fps)
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                   + str(len(buf)).encode() + b"\r\n\r\n" + buf.tobytes() + b"\r\n")
    except GeneratorExit:
        # Browser closed the stream (Stop / navigated away) — let the engine lock release.
        return


# --------------------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------------------
app = FastAPI(title="SAM 3 Open-Vocabulary Video Detector")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    page = STATIC_DIR / "index.html"
    if not page.exists():
        raise HTTPException(500, "static/index.html missing")
    return HTMLResponse(page.read_text())


@app.get("/status")
def status() -> JSONResponse:
    active = select_engine()
    return JSONResponse({
        "sam3_available": SAM3.available,
        "active_backend": active.name,
        "active_label": active.label,
        "yoloe_model": YOLOE_MODEL,
        "model_path": str(MODEL_PATH),
        "videos": [p.name for p in sorted(INPUT_DIR.iterdir())
                   if p.suffix.lower() in VIDEO_EXTS],
    })


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> JSONResponse:
    if Path(file.filename).suffix.lower() not in VIDEO_EXTS:
        raise HTTPException(400, "Unsupported file type")
    dest = INPUT_DIR / Path(file.filename).name
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return JSONResponse({"saved": dest.name})


@app.post("/detect")
async def detect(payload: dict) -> JSONResponse:
    engine = select_engine(payload.get("backend"))  # auto, or force 'sam3'/'yoloe'
    if engine is None:
        raise HTTPException(
            503,
            "SAM 3 weights (sam3.pt) not found. Request access at "
            "https://huggingface.co/facebook/sam3, or use the YOLOE-26 fallback.",
        )
    concepts = parse_prompt(payload.get("prompt", ""))
    if not concepts:
        raise HTTPException(400, "Empty prompt — type at least one concept.")
    video_name = payload.get("video")
    video_path = INPUT_DIR / Path(video_name or "").name
    if not video_path.exists():
        raise HTTPException(404, f"Video not found in input/: {video_name}")

    conf = float(payload.get("score_threshold", 0.4))
    stride = int(payload.get("frame_stride", 1))

    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "queued", "processed": 0, "total": 0, "counts": {},
                        "concepts": concepts, "backend": engine.name, "error": None, "output": None}
    threading.Thread(
        target=run_detection,
        args=(job_id, engine, video_path, concepts, conf, stride),
        daemon=True,
    ).start()
    return JSONResponse({"job_id": job_id, "backend": engine.name})


@app.get("/live")
def live(video: str, prompt: str, backend: str = "yoloe",
         score_threshold: float = 0.4, frame_stride: int = 1) -> StreamingResponse:
    """Live MJPEG preview — boxes stream in as the GPU finishes each frame.

    Defaults to YOLOE (fast enough to feel live, ~7-8 fps on this card). SAM 3 works too
    but at ~0.7 fps it's better suited to the processed /detect path.
    """
    engine = select_engine(backend)
    if engine is None:
        raise HTTPException(503, "SAM 3 weights (sam3.pt) not found.")
    concepts = parse_prompt(prompt)
    if not concepts:
        raise HTTPException(400, "Empty prompt — type at least one concept.")
    video_path = INPUT_DIR / Path(video).name
    if not video_path.exists():
        raise HTTPException(404, f"Video not found in input/: {video}")
    return StreamingResponse(
        live_mjpeg(engine, video_path, concepts,
                   float(score_threshold), int(max(1, frame_stride))),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/progress/{job_id}")
def progress(job_id: str) -> JSONResponse:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job")
        return JSONResponse(dict(job))


@app.get("/video/annotated.mp4")
def annotated() -> FileResponse:
    path = OUTPUT_DIR / "annotated.mp4"
    if not path.exists():
        raise HTTPException(404, "No annotated video yet")
    # FileResponse handles HTTP Range requests, so the browser can seek/scrub.
    return FileResponse(path, media_type="video/mp4")


@app.get("/video/source/{name}")
def source_video(name: str) -> FileResponse:
    path = INPUT_DIR / Path(name).name
    if not path.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(path, media_type="video/mp4")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
