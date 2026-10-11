"""USB camera (read in the background) and frame description by a vision model in Ollama."""

import base64
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import httpx

from models import VISIONS, capabilities, load_config

SNAPSHOTS = Path(__file__).parent / "snapshots"
KEEP_SNAPSHOTS = 200

DESCRIBE_PROMPT = (
    "Opisz po polsku, zwięźle (2-4 zdania), co widać na zdjęciu z kamery: najważniejsze obiekty, "
    "ludzi i ich położenie w kadrze (lewo, środek, prawo). Nie zgaduj rzeczy, których nie widać."
)
QUESTION_PROMPT = (
    "Na podstawie zdjęcia z kamery odpowiedz krótko po polsku na pytanie: {question}\n"
    "Jeśli ze zdjęcia nie da się tego stwierdzić, napisz to wprost. Dodaj jedno zdanie ogólnego opisu kadru."
)


class Camera:
    """Reads frames in a separate thread and keeps the latest one: for the live preview and photos for the VLM."""

    def __init__(self, device=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera /dev/video{device}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cond = threading.Condition()
        self._frame, self._t, self._count = None, 0.0, 0
        self._preview, self._preview_t = None, 0.0
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        with self._cond:  # auto-exposure needs a dozen or so frames after power-on
            if not self._cond.wait_for(lambda: self._count >= 15, timeout=10):
                raise RuntimeError("camera returns no frames")

    def _loop(self):
        while self._running:
            ok, img = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self._cond:
                self._frame, self._t = img, time.monotonic()
                self._count += 1
                self._cond.notify_all()

    def frame(self, after=None, timeout=3.0):
        """Frame read after the moment `after` (default: now), e.g. only once the head has stopped moving."""
        after = time.monotonic() if after is None else after
        with self._cond:
            if not self._cond.wait_for(lambda: self._t > after, timeout):
                raise RuntimeError("kamera nie zwraca klatek")
            return self._frame

    def preview_jpeg(self, quality=70):
        """Latest frame as JPEG for the live preview, encoded once per frame."""
        with self._cond:
            img, t = self._frame, self._t
        if t != self._preview_t:
            self._preview = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()
            self._preview_t = t
        return self._preview

    def snapshot(self, after=None, quality=85):
        """Photo for the VLM, also saved in snapshots/; returns (jpeg, path)."""
        jpg = cv2.imencode(".jpg", self.frame(after), [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()
        SNAPSHOTS.mkdir(exist_ok=True)
        path = SNAPSHOTS / f"{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}.jpg"
        path.write_bytes(jpg)
        for old in sorted(SNAPSHOTS.glob("*.jpg"))[:-KEEP_SNAPSHOTS]:
            old.unlink()
        return jpg, path

    def close(self):
        self._running = False
        self._thread.join(timeout=2)
        self.cap.release()


def describe(jpeg, vision, question=None, on_token=None, timeout=180):
    """Frame description from Ollama by a model from models.VISIONS; on_token gets the text pieces live."""
    prompt = QUESTION_PROMPT.format(question=question) if question else DESCRIBE_PROMPT
    body = {
        "model": vision["model"],
        "stream": True,
        # A loaded Gemma keeps the GPU at ~100% and ~127 W even with no requests (Ollama 0.17 on ROCm),
        # so we free it after 2 min of silence; loading it again costs ~5 s on the first description.
        "keep_alive": "2m",
        "messages": [{"role": "user", "content": prompt, "images": [base64.b64encode(jpeg).decode()]}],
        # num_ctx overrides the long Modelfile context (jaro-jm/vlm: 131k) that would take half the VRAM as KV cache
        "options": {"num_ctx": 4096, "temperature": 0.2, "num_predict": 300},
    }
    if "thinking" in capabilities(vision["base"], vision["model"]):
        body["think"] = False  # for describing a frame, thinking only adds time
    parts = []
    with httpx.stream("POST", f"{vision['base']}/api/chat", json=body, timeout=timeout) as r:
        if r.status_code >= 400:
            r.read()
            raise RuntimeError(f"{vision['label']}: {r.text}")
        for line in r.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if "error" in chunk:
                raise RuntimeError(chunk["error"])
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                parts.append(piece)
                if on_token:
                    on_token(piece)
    return "".join(parts).strip()


if __name__ == "__main__":
    cam = Camera()
    try:
        t0 = time.monotonic()
        jpg, path = cam.snapshot()
        t1 = time.monotonic()
        vision = VISIONS[load_config()["vision"]]
        describe(jpg, vision, " ".join(sys.argv[1:]) or None, on_token=lambda s: print(s, end="", flush=True))
        print(f"\n\n[photo {t1 - t0:.1f} s, description {time.monotonic() - t1:.1f} s, {path}]")
    finally:
        cam.close()
