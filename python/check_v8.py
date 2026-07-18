#!/usr/bin/env python3
"""
Webcam → NPU YOLOv8n → WebSocket JPEG preview + class names.

Needs libyolov8n.so + model.params (or model_v8.params):

  python3 check_v8.py --source auto
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

# Before importing cv2 — kill GStreamer probes on /dev/video*
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_GSTREAMER", "0")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

cv2.setNumThreads(2)

INPUT_H = 640
INPUT_W = 640
MAX_BOXES = 100
# Lower USB bandwidth → fewer errno=19 disconnects on LPi4A
CAM_W = 320
CAM_H = 240
CAM_FPS = 15
PREVIEW_FPS = 12
PREVIEW_MAX_W = 480
JPEG_QUALITY = 40

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


LIB_PATH = _find_file("libyolov8n.so")
_params = _find_file("model_v8.params")
if not os.path.isfile(_params):
    _params = _find_file("model.params")
PARAMS_PATH = _params.encode("utf-8")

# Custom model class names (PPE etc.) — NOT COCO.
# Priority: --names / classes.txt next to .so / cwd / package.
def _load_class_names() -> List[str]:
    candidates = [
        _find_file("classes.txt"),
        os.path.join(os.getcwd(), "classes.txt"),
        os.path.join(_HERE, "classes.txt"),
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                names = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
            if names:
                print(f"Loaded {len(names)} class names from {p}")
                return names
    # Fallback: id0..id9 (custom models — never pretend COCO)
    return [f"id{i}" for i in range(10)]


CLASS_NAMES: List[str] = _load_class_names()


def set_class_names(names: List[str]) -> None:
    global CLASS_NAMES
    CLASS_NAMES = list(names)

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
    if 0 <= cid < len(CLASS_NAMES):
        return CLASS_NAMES[cid]
    return f"id{cid}"


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
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    try:
        cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
    except Exception:
        pass
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass


def _frame_ok(cap: cv2.VideoCapture) -> bool:
    for _ in range(6):
        ok, frame = cap.read()
        if ok and frame is not None and getattr(frame, "size", 0) > 0:
            return True
        time.sleep(0.05)
    return False


def _as_dev_path(dev: Union[int, str]) -> str:
    if isinstance(dev, int):
        return f"/dev/video{dev}"
    return str(dev)


def _try_open_one(dev: Union[int, str]) -> Optional[cv2.VideoCapture]:
    """V4L2 only — CAP_ANY pulls in GStreamer and wastes seconds on reopen."""
    path = _as_dev_path(dev)
    if path.startswith("/dev/") and not os.path.exists(path):
        return None

    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None

    for fourcc in ("MJPG", "YUYV", None):
        if fourcc is not None:
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            except Exception:
                continue
        _configure_cam(cap)
        if _frame_ok(cap):
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Webcam OK: {path} {w}x{h} (V4L2, {fourcc or 'default'})")
            return cap

    cap.release()
    return None


def _wait_device(path: str, timeout_s: float = 4.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if os.path.exists(path):
            return True
        time.sleep(0.15)
    return os.path.exists(path)


def reopen_webcam(path: str) -> Optional[cv2.VideoCapture]:
    """Fast soft reopen of one node (no GStreamer probe storm)."""
    _mute_cv2_logs()
    if not _wait_device(path, 3.0):
        return None
    time.sleep(0.3)
    return _try_open_one(path)


def open_webcam(source: Union[int, str], prefer: Optional[str] = None):
    """Return (VideoCapture, device_path)."""
    _mute_cv2_logs()
    time.sleep(0.4)
    nodes = list_v4l_nodes()
    print(f"V4L devices: {nodes or '(none)'}")

    ordered: List[str] = []
    if prefer:
        ordered.append(prefer)
    if source == "auto":
        # video0 is usually capture; video1 often metadata-only on UVC
        ordered.extend(nodes)
    elif isinstance(source, int):
        ordered.append(_as_dev_path(source))
        for alt in (0, 1, 2):
            ordered.append(_as_dev_path(alt))
    else:
        ordered.append(str(source))
        ordered.extend(nodes)

    seen: Set[str] = set()
    uniq: List[str] = []
    for d in ordered:
        if d in seen:
            continue
        if d.startswith("/dev/") and not os.path.exists(d):
            continue
        seen.add(d)
        uniq.append(d)

    tried = []
    for dev in uniq:
        tried.append(dev)
        print(f"Trying {dev} ...")
        cap = _try_open_one(dev)
        if cap is not None:
            return cap, dev

    raise RuntimeError(
        "Cannot open webcam.\n"
        f"  tried: {tried}\n"
        f"  nodes: {nodes or 'none'}\n"
        "  Try: python3 check_v8.py --source /dev/video0\n"
        "  or:  python3 check_v8.py --source image.jpg   # без камеры, замер FPS"
    )


class Pipeline:
    def __init__(self, source: str):
        self.source = _parse_source(source)
        self.is_cam = self.source == "auto" or isinstance(self.source, int) or (
            isinstance(self.source, str) and self.source.startswith("/dev/video")
        )
        self._cam_pref: Optional[str] = (
            self.source if isinstance(self.source, str) and self.source.startswith("/dev/") else None
        )

        # NPU first — USB cams often drop if opened before VIP9000 init
        self._cap = None
        self.sess = lib.init_model(PARAMS_PATH)
        if not self.sess:
            raise RuntimeError("init_model failed")
        lib.yolo_set_thresholds(self.sess, 0.25, 0.45)
        lib.yolo_warmup(self.sess, 1)

        if self.is_cam:
            self._cap, self._cam_pref = open_webcam(self.source, prefer=self._cam_pref)
        elif isinstance(self.source, str) and os.path.isfile(self.source):
            img = cv2.imread(self.source)
            if img is None:
                raise RuntimeError(f"Cannot read image: {self.source}")
            self._still = img
            self._cap = None
            print(f"Still image mode: {self.source} {img.shape[1]}x{img.shape[0]}")
        else:
            self._still = None
            self._cap = None

        if not hasattr(self, "_still"):
            self._still = None

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
            cap, path = open_webcam(self.source, prefer=self._cam_pref)
            self._cam_pref = path
            return cap
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {self.source}")
        return cap

    def _capture_loop(self):
        # Still image: push same frame forever — clean FPS without USB cam
        if self._still is not None:
            while not self._stop.is_set():
                with self._latest_lock:
                    if self._latest is not None:
                        self.stats["dropped"] += 1
                    self._latest = self._still
                time.sleep(0.001)
            return

        fail = 0
        reopen_n = 0
        cap = self._cap if self._cap is not None else self._open_cap()
        self._cap = None
        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                fail += 1
                if fail < 15:
                    time.sleep(0.01)
                    continue
                try:
                    cap.release()
                except Exception:
                    pass
                path = self._cam_pref or "/dev/video0"
                # Soft reopen same node first (avoid full probe + GStreamer)
                time.sleep(1.0)
                new_cap = reopen_webcam(path) if self.is_cam else None
                if new_cap is None and self.is_cam:
                    try:
                        new_cap = self._open_cap()
                    except Exception as e:
                        reopen_n += 1
                        if reopen_n <= 3 or reopen_n % 10 == 0:
                            print(f"cam reopen failed ({reopen_n}): {e}")
                        time.sleep(2.0)
                        continue
                if new_cap is None:
                    time.sleep(2.0)
                    continue
                cap = new_cap
                fail = 0
                reopen_n = 0
                print(f"cam reopened: {self._cam_pref}")
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
<h3>TH1520 NPU YOLOv8n — webcam WebSocket</h3>
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
        help="auto | 0 | /dev/video0 | path/to.jpg (still = замер FPS без камеры)",
    )
    ap.add_argument(
        "--names",
        default="",
        help="classes.txt (one name per line, id order) — custom PPE model",
    )
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    if args.names:
        with open(args.names, "r", encoding="utf-8") as f:
            names = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
        set_class_names(names)
        print(f"Using {len(names)} names from {args.names}: {names}")
    else:
        print(f"Class names ({len(CLASS_NAMES)}): {CLASS_NAMES}")

    print(f"Loading YOLOv8 NPU + webcam source={args.source}")
    print(f"  lib={LIB_PATH}")
    print(f"  params={PARAMS_PATH.decode()}")
    pipe = Pipeline(args.source)
    print(f"Open http://<board-ip>:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    # pipe.stop() already in lifespan shutdown


if __name__ == "__main__":
    main()
