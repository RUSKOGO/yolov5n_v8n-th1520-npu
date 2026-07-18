#!/usr/bin/env python3
"""
Webcam → NPU YOLOv5n → WebSocket JPEG preview + class names.

Run next to libyolov5n.so and model.params:

  python3 check.py --source auto
  open http://<board-ip>:8000/
"""
from __future__ import annotations

import argparse
import asyncio
import ctypes
import glob
import os
import threading
import time
from typing import List, Optional, Set, Tuple, Union
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

cv2.setNumThreads(2)

INPUT_H = 640
INPUT_W = 640
MAX_BOXES = 100
CAM_W = 640
CAM_H = 480
CAM_FPS = 30
PREVIEW_FPS = 20
PREVIEW_MAX_W = 480
JPEG_QUALITY = 45

_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATE_DIRS = [
    os.getcwd(),
    _HERE,
    os.path.join(_HERE, ".."),
]


def _find_file(name: str) -> str:
    for d in _CANDIDATE_DIRS:
        p = os.path.abspath(os.path.join(d, name))
        if os.path.isfile(p):
            return p
    return os.path.abspath(os.path.join(os.getcwd(), name))


LIB_PATH = _find_file("libyolov5n.so")
PARAMS_PATH = _find_file("model.params").encode("utf-8")

# COCO 80 classes (YOLOv5)
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

if not os.path.exists(LIB_PATH):
    raise SystemExit(f"Missing {LIB_PATH}")

lib = ctypes.CDLL(LIB_PATH)
lib.init_model.argtypes = [ctypes.c_char_p]
lib.init_model.restype = ctypes.c_void_p
lib.run_inference.argtypes = [
    ctypes.c_void_p,
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=3, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,
]
lib.run_inference.restype = ctypes.c_int
lib.release_model.argtypes = [ctypes.c_void_p]
lib.yolo_set_thresholds.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float]
lib.yolo_set_thresholds.restype = ctypes.c_int
lib.yolo_warmup.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.yolo_warmup.restype = ctypes.c_int
lib.yolo_get_timings_us.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]
lib.yolo_get_timings_us.restype = ctypes.c_int


def get_timings_ms(sess) -> Tuple[float, float, float]:
    pre = ctypes.c_float()
    npu = ctypes.c_float()
    post = ctypes.c_float()
    lib.yolo_get_timings_us(sess, ctypes.byref(pre), ctypes.byref(npu), ctypes.byref(post))
    return pre.value / 1000.0, npu.value / 1000.0, post.value / 1000.0


def class_name(cid: int) -> str:
    if 0 <= cid < len(COCO_NAMES):
        return COCO_NAMES[cid]
    return str(cid)


def letterbox_bgr(frame):
    h, w = frame.shape[:2]
    scale = min(INPUT_W / w, INPUT_H / h)
    nw, nh = int(scale * w), int(scale * h)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (INPUT_H - nh) // 2
    bottom = INPUT_H - nh - top
    left = (INPUT_W - nw) // 2
    right = INPUT_W - nw - left
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128)
    )
    return np.ascontiguousarray(padded, dtype=np.uint8), scale, left, top


def draw_boxes(frame, boxes, scale, dw, dh):
    for b in boxes:
        x1 = int((b[0] - dw) / scale)
        y1 = int((b[1] - dh) / scale)
        x2 = int((b[2] - dw) / scale)
        y2 = int((b[3] - dh) / scale)
        cid = int(b[5])
        label = f"{class_name(cid)} {b[4]:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        tw = max(80, 9 * len(label))
        cv2.rectangle(frame, (x1, max(0, y1 - 18)), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(
            frame, label, (x1 + 2, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )
    return frame


def _parse_source(source: str) -> Union[int, str]:
    s = source.strip()
    if s.lower() in ("auto", "any", ""):
        return "auto"
    if s.isdigit():
        return int(s)
    if s.startswith("/dev/video"):
        return s
    return s


def list_v4l_nodes() -> List[str]:
    return sorted(glob.glob("/dev/video*"), key=lambda p: (len(p), p))


def _mute_cv2_logs() -> None:
    """Silence noisy V4L2/GStreamer probe warnings."""
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        try:
            cv2.setLogLevel(3)  # LOG_LEVEL_ERROR in some builds
        except Exception:
            pass


def _configure_cam(cap: cv2.VideoCapture) -> None:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass


def _frame_ok(cap: cv2.VideoCapture) -> bool:
    for _ in range(4):
        ok, frame = cap.read()
        if ok and frame is not None and getattr(frame, "size", 0) > 0:
            return True
        time.sleep(0.03)
    return False


def _as_dev_path(dev: Union[int, str]) -> str:
    if isinstance(dev, int):
        return f"/dev/video{dev}"
    return str(dev)


def _try_open_one(dev: Union[int, str]) -> Optional[cv2.VideoCapture]:
    """
    Open once per backend, then tweak FOURCC in-place (no reopen spam).
    Prefer /dev/videoN paths — OpenCV index open is flaky on some boards.
    """
    path = _as_dev_path(dev)
    if path.startswith("/dev/") and not os.path.exists(path):
        return None

    # V4L2 first; CAP_ANY only as fallback (avoids GStreamer spam when V4L works)
    for backend, bname in ((cv2.CAP_V4L2, "V4L2"), (cv2.CAP_ANY, "ANY")):
        cap = cv2.VideoCapture(path, backend)
        if not cap.isOpened():
            continue

        _configure_cam(cap)
        for fourcc in (None, "MJPG", "YUYV"):
            if fourcc is not None:
                try:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
                    _configure_cam(cap)
                except Exception:
                    continue
            if _frame_ok(cap):
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"Webcam OK: {path} {w}x{h} ({bname}, {fourcc or 'default'})")
                return cap

        cap.release()
    return None


def open_webcam(source: Union[int, str]) -> cv2.VideoCapture:
    _mute_cv2_logs()
    nodes = list_v4l_nodes()
    print(f"V4L devices: {nodes or '(none)'}")

    if source == "auto":
        ordered: List[str] = list(nodes)
    elif isinstance(source, int):
        ordered = [_as_dev_path(source)]
        for alt in (source + 1, source + 2, 1, 2, 0):
            p = _as_dev_path(alt)
            if p not in ordered and (not p.startswith("/dev/") or os.path.exists(p)):
                ordered.append(p)
    else:
        ordered = [source]
        # if user passed index-like path failure neighbors
        if source.startswith("/dev/video"):
            suf = source.replace("/dev/video", "")
            if suf.isdigit():
                for alt in (int(suf) + 1, int(suf) + 2, 1, 0):
                    p = _as_dev_path(alt)
                    if p not in ordered and os.path.exists(p):
                        ordered.append(p)

    tried = []
    for dev in ordered:
        tried.append(str(dev))
        print(f"Trying {dev} ...")
        cap = _try_open_one(dev)
        if cap is not None:
            return cap

    raise RuntimeError(
        "Cannot open webcam.\n"
        f"  tried: {tried}\n"
        f"  nodes: {nodes or 'none'}\n"
        "  Try: python3 check.py --source /dev/video1\n"
        "  Debug: ls -l /dev/video*; v4l2-ctl --list-devices"
    )


class Pipeline:
    def __init__(self, source: str):
        self.source = _parse_source(source)
        self.is_cam = self.source == "auto" or isinstance(self.source, int) or (
            isinstance(self.source, str) and self.source.startswith("/dev/video")
        )
        # Camera first — fail fast before loading NPU weights
        self._cap = open_webcam(self.source) if self.is_cam else None

        self.sess = lib.init_model(PARAMS_PATH)
        if not self.sess:
            if self._cap is not None:
                self._cap.release()
            raise RuntimeError("init_model failed")
        lib.yolo_set_thresholds(self.sess, 0.25, 0.45)
        lib.yolo_warmup(self.sess, 2)

        self.out_boxes = np.zeros(MAX_BOXES * 6, dtype=np.float32)
        self.lock = threading.Lock()
        self.jpeg: Optional[bytes] = None
        self.frame_id = 0
        self.labels: List[str] = []
        self.stats = {
            "frames": 0,
            "compute_fps": 0.0,
            "npu_fps": 0.0,
            "pre_ms": 0.0,
            "npu_ms": 0.0,
            "post_ms": 0.0,
            "compute_ms": 0.0,
            "boxes": 0,
            "dropped": 0,
            "labels": "",
            "source": str(source),
        }
        self._stop = threading.Event()
        self._latest = None
        self._latest_lock = threading.Lock()
        self._new_jpeg = threading.Event()
        self._cap_thread = threading.Thread(target=self._capture_loop, name="cap", daemon=True)
        self._inf_thread = threading.Thread(target=self._infer_loop, name="inf", daemon=True)

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._clients: Set[WebSocket] = set()
        self._clients_lock = threading.Lock()
        self._send_busy: Set[WebSocket] = set()
        self._broadcast_task: Optional[asyncio.Task] = None

    def start(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._cap_thread.start()
        self._inf_thread.start()
        self._broadcast_task = loop.create_task(self._broadcast_loop())

    def stop(self):
        self._stop.set()
        self._new_jpeg.set()
        if self._broadcast_task and self._loop:
            self._broadcast_task.cancel()
        self._cap_thread.join(timeout=2)
        self._inf_thread.join(timeout=2)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        lib.release_model(self.sess)

    def _open_cap(self):
        if self.is_cam or self.source == "auto":
            return open_webcam(self.source)
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {self.source}")
        return cap

    def _capture_loop(self):
        fail = 0
        cap = self._cap if self._cap is not None else self._open_cap()
        self._cap = None
        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                fail += 1
                if fail > 60:
                    try:
                        cap.release()
                    except Exception:
                        pass
                    time.sleep(0.2)
                    try:
                        cap = self._open_cap()
                        fail = 0
                    except Exception as e:
                        print(f"cam reopen failed: {e}")
                        time.sleep(1.0)
                else:
                    time.sleep(0.005)
                continue
            fail = 0
            with self._latest_lock:
                if self._latest is not None:
                    self.stats["dropped"] += 1
                self._latest = frame
        try:
            cap.release()
        except Exception:
            pass

    def _pop_latest(self):
        with self._latest_lock:
            frame = self._latest
            self._latest = None
            return frame

    def _broadcast_jpeg(self, jpeg: bytes):
        # Infer thread must NOT block on network — only signal.
        self._new_jpeg.set()

    async def _broadcast_loop(self):
        """Async sender: always latest frame, skip clients still sending."""
        last_id = -1
        while not self._stop.is_set():
            await asyncio.sleep(0.01)
            with self.lock:
                jpeg = self.jpeg
                fid = self.frame_id
            if jpeg is None or fid == last_id:
                continue
            last_id = fid
            with self._clients_lock:
                clients = list(self._clients)
            for ws in clients:
                if ws in self._send_busy:
                    continue
                self._send_busy.add(ws)
                asyncio.create_task(self._send_one(ws, jpeg))

    async def _send_one(self, ws: WebSocket, jpeg: bytes):
        try:
            await ws.send_bytes(jpeg)
        except Exception:
            self.remove_client(ws)
            try:
                await ws.close()
            except Exception:
                pass
        finally:
            self._send_busy.discard(ws)

    def add_client(self, ws: WebSocket):
        with self._clients_lock:
            self._clients.add(ws)

    def remove_client(self, ws: WebSocket):
        with self._clients_lock:
            self._clients.discard(ws)
        self._send_busy.discard(ws)

    def _infer_loop(self):
        ema_compute_fps = 0.0
        ema_npu_fps = 0.0
        last_jpeg_t = 0.0
        jpeg_interval = 1.0 / PREVIEW_FPS

        while not self._stop.is_set():
            frame = self._pop_latest()
            if frame is None:
                time.sleep(0.0005)
                continue

            t0 = time.time()
            padded, scale, dw, dh = letterbox_bgr(frame)
            n = lib.run_inference(self.sess, padded, self.out_boxes, MAX_BOXES)
            if n < 0:
                continue
            compute_ms = (time.time() - t0) * 1000.0
            pre_ms, npu_ms, post_ms = get_timings_ms(self.sess)

            compute_fps = 1000.0 / compute_ms if compute_ms > 0 else 0.0
            npu_fps = 1000.0 / npu_ms if npu_ms > 0 else 0.0
            ema_compute_fps = (
                compute_fps if ema_compute_fps == 0 else 0.85 * ema_compute_fps + 0.15 * compute_fps
            )
            ema_npu_fps = npu_fps if ema_npu_fps == 0 else 0.85 * ema_npu_fps + 0.15 * npu_fps

            labels = []
            if n > 0:
                boxes = self.out_boxes[: n * 6].reshape(-1, 6)
                for b in boxes:
                    labels.append(f"{class_name(int(b[5]))}:{b[4]:.2f}")
            else:
                boxes = []

            with self.lock:
                self.labels = labels
                self.stats.update(
                    {
                        "frames": self.stats["frames"] + 1,
                        "compute_fps": round(ema_compute_fps, 2),
                        "npu_fps": round(ema_npu_fps, 2),
                        "pre_ms": round(pre_ms, 2),
                        "npu_ms": round(npu_ms, 2),
                        "post_ms": round(post_ms, 2),
                        "compute_ms": round(compute_ms, 2),
                        "boxes": int(n),
                        "labels": ", ".join(labels[:12]),
                    }
                )

            now = time.time()
            if now - last_jpeg_t < jpeg_interval:
                continue
            last_jpeg_t = now

            vis = frame.copy()
            draw_boxes(vis, boxes, scale, dw, dh)
            lines = [
                f"COMPUTE {ema_compute_fps:.1f} FPS ({compute_ms:.1f}ms)",
                f"NPU {ema_npu_fps:.1f} FPS ({npu_ms:.1f}ms)",
            ]
            if labels:
                lines.append(" | ".join(labels[:6]))
            y = 24
            for line in lines:
                cv2.putText(vis, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(vis, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
                y += 22

            h, w = vis.shape[:2]
            if w > PREVIEW_MAX_W:
                nh = int(h * PREVIEW_MAX_W / w)
                vis = cv2.resize(vis, (PREVIEW_MAX_W, nh), interpolation=cv2.INTER_AREA)

            ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok:
                continue
            jpeg = buf.tobytes()
            with self.lock:
                self.jpeg = jpeg
                self.frame_id += 1
            self._broadcast_jpeg(jpeg)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if pipe is not None and pipe._loop is None:
        pipe.start(asyncio.get_running_loop())
    yield
    if pipe is not None:
        pipe.stop()


app = FastAPI(lifespan=lifespan)
pipe: Optional[Pipeline] = None

PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>NPU webcam WS</title>
<style>
body{margin:0;background:#111;color:#eee;font-family:monospace}
#wrap{display:flex;flex-direction:column;align-items:center;padding:8px;gap:8px}
canvas{max-width:960px;width:100%;background:#000;image-rendering:auto}
pre{font-size:13px;white-space:pre-wrap;max-width:960px}
#st{color:#0f0}
</style></head>
<body><div id="wrap">
<h3>TH1520 NPU — webcam WebSocket</h3>
<div id="st">connecting...</div>
<canvas id="c"></canvas>
<pre id="s">...</pre>
</div>
<script>
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const st = document.getElementById('st');
let ws = null;
let drawing = false;
let pending = null; // only keep latest blob

function paintBlob(blob){
  if (drawing) { pending = blob; return; }
  drawing = true;
  createImageBitmap(blob).then(bmp => {
    if (canvas.width !== bmp.width || canvas.height !== bmp.height) {
      canvas.width = bmp.width;
      canvas.height = bmp.height;
    }
    ctx.drawImage(bmp, 0, 0);
    bmp.close();
    drawing = false;
    if (pending) {
      const p = pending; pending = null;
      paintBlob(p);
    }
  }).catch(() => { drawing = false; });
}

function connect(){
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(proto + '://' + location.host + '/ws');
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => { st.textContent = 'WS connected'; };
  ws.onclose = () => {
    st.textContent = 'WS disconnected — reconnect in 1s';
    setTimeout(connect, 1000);
  };
  ws.onerror = () => { try { ws.close(); } catch(e){} };
  ws.onmessage = (ev) => {
    // always replace — never queue frames (kills lag)
    paintBlob(new Blob([ev.data], {type:'image/jpeg'}));
  };
}

async function tickStats(){
  try{
    const r = await fetch('/stats', {cache:'no-store'});
    const j = await r.json();
    document.getElementById('s').textContent =
      'compute_fps: '+j.compute_fps+'\\n'+
      'npu_fps:     '+j.npu_fps+'\\n'+
      'compute_ms:  '+j.compute_ms+'\\n'+
      'npu_ms:      '+j.npu_ms+'\\n'+
      'boxes:       '+j.boxes+'\\n'+
      'objects:     '+j.labels+'\\n'+
      'frames:      '+j.frames+'\\n'+
      'dropped:     '+j.dropped;
  }catch(e){}
}
connect();
setInterval(tickStats, 500);
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


@app.get("/stats")
def stats():
    if pipe is None:
        return JSONResponse({"error": "not ready"})
    with pipe.lock:
        return JSONResponse(dict(pipe.stats))


@app.websocket("/ws")
async def ws_video(ws: WebSocket):
    await ws.accept()
    if pipe is None:
        await ws.close()
        return
    pipe.add_client(ws)
    with pipe.lock:
        jpeg = pipe.jpeg
    if jpeg:
        try:
            await ws.send_bytes(jpeg)
        except Exception:
            pipe.remove_client(ws)
            return
    try:
        # Block until client disconnects (browser need not send anything)
        while True:
            await ws.receive()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        pipe.remove_client(ws)


def main():
    global pipe
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        default="auto",
        help="auto | 0 | 1 | /dev/video1  (UVC often needs video1, not video0)",
    )
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    print(f"Loading NPU + webcam source={args.source}")
    pipe = Pipeline(args.source)
    print(f"Open http://<board-ip>:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    # pipe.stop() already in lifespan shutdown


if __name__ == "__main__":
    main()
