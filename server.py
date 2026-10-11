"""Jarvis web page at http://127.0.0.1:8090: live camera, model stream, head control and model selection."""

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import Agent, Head, call_label
from vision import SNAPSHOTS

HOST, PORT = "127.0.0.1", 8090
WEB = Path(__file__).with_name("web")
LOG_LIMIT = 20000  # events kept so that a refreshed page can replay the whole conversation
PREVIEW_FPS = 15


class Bus:
    """Events from agent threads to all open pages (SSE), plus a log to replay after a refresh."""

    def __init__(self, loop):
        self.loop, self.clients, self.log = loop, set(), []

    def emit(self, kind, /, **data):
        self.loop.call_soon_threadsafe(self._publish, {"type": kind, "t": time.time(), **data})

    def _publish(self, event):
        self.log.append(event)
        del self.log[:-LOG_LIMIT]
        for queue in self.clients:
            queue.put_nowait(event)


def find_gpu():
    """sysfs directory of the card with the most VRAM (7900 XTX, not the iGPU)."""
    cards = [d for d in Path("/sys/class/drm").glob("card*/device")
             if d.parent.name[4:].isdigit() and (d / "mem_info_vram_total").exists()]
    return max(cards, key=lambda d: int((d / "mem_info_vram_total").read_text()), default=None)


def read_first(*paths):
    for p in paths:
        try:
            return int(p.read_text())
        except (OSError, ValueError):
            continue
    return None


def gpu_stats(dev):
    if dev is None:
        return None
    hwmon = next(iter(sorted((dev / "hwmon").glob("hwmon*"))), None)
    temp = read_first(hwmon / "temp2_input", hwmon / "temp1_input") if hwmon else None  # junction, otherwise edge
    power = read_first(hwmon / "power1_average", hwmon / "power1_input") if hwmon else None
    return {
        "busy": read_first(dev / "gpu_busy_percent"),
        "vram_used": read_first(dev / "mem_info_vram_used"),
        "vram_total": read_first(dev / "mem_info_vram_total"),
        "temp_c": temp // 1000 if temp else None,
        "power_w": power // 1_000_000 if power else None,
    }


state = {}


@asynccontextmanager
async def lifespan(app):
    SNAPSHOTS.mkdir(exist_ok=True)
    bus = Bus(asyncio.get_running_loop())
    head = await asyncio.to_thread(Head, lambda s: bus.emit("head", **s))
    state.update(bus=bus, head=head, agent=Agent(head, bus.emit), gpu=find_gpu(),
                 http=httpx.AsyncClient(timeout=0.8))
    print(f"Jarvis: http://{HOST}:{PORT}", flush=True)
    yield
    await state["http"].aclose()
    await asyncio.to_thread(head.close)


SNAPSHOTS.mkdir(exist_ok=True)
app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB), name="static")
app.mount("/snapshots", StaticFiles(directory=SNAPSHOTS), name="snapshots")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/events")
async def events(request: Request):
    bus = state["bus"]
    queue, backlog = asyncio.Queue(), list(bus.log)
    bus.clients.add(queue)

    async def stream():
        try:
            # current state first, because the log sometimes gets cleared (new conversation, brain change)
            for event in ({"type": "hello"}, {"type": "config", **state["agent"].config_info()},
                          {"type": "head", **state["head"].state()}):
                yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
            for event in backlog:
                yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
            while not await request.is_disconnected():
                try:
                    event = await asyncio.wait_for(queue.get(), 15)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
                    continue
                yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
        finally:
            bus.clients.discard(queue)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/stream.mjpg")
async def preview():
    cam = state["head"].cam

    async def frames():
        last = None
        while True:
            jpg = await asyncio.to_thread(cam.preview_jpeg)
            if jpg is not last:
                last = jpg
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            await asyncio.sleep(1 / PREVIEW_FPS)

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


class ChatIn(BaseModel):
    text: str
    think: bool = True


class ConfigIn(BaseModel):
    brain: str | None = None
    vision: str | None = None


class LookIn(BaseModel):
    direction: str
    degrees: int = 30


def busy_error():
    return HTTPException(409, "Jarvis is still working")


@app.post("/api/chat")
def chat(body: ChatIn):
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "empty command")
    agent = state["agent"]
    if not agent.start(agent.ask, text, body.think):
        raise busy_error()
    return {"ok": True}


@app.post("/api/stop")
def stop():
    state["agent"].cancelled = True
    return {"ok": True}


@app.post("/api/reset")
async def reset():
    agent, bus = state["agent"], state["bus"]
    if agent.busy:
        raise busy_error()
    bus.log.clear()
    agent.reset()
    await asyncio.to_thread(state["head"].look, "center")
    return {"ok": True}


@app.post("/api/look")
async def look(body: LookIn):
    agent = state["agent"]
    if agent.busy:
        raise busy_error()
    state["bus"].emit("manual", label=call_label("look", body.model_dump()))
    return {"result": await asyncio.to_thread(state["head"].look, body.direction, body.degrees)}


@app.post("/api/describe")
def describe_now():
    agent = state["agent"]
    if agent.busy:
        raise busy_error()
    call_id = f"manual_{int(time.time() * 1000)}"
    state["bus"].emit("manual", label=f"Manual: view description by {agent.vision['label']}")
    if not agent.start(agent.describe_now, call_id):
        raise busy_error()
    return {"ok": True}


@app.post("/api/config")
def set_config(body: ConfigIn):
    agent, bus = state["agent"], state["bus"]
    if agent.busy:
        raise busy_error()
    if body.brain and body.brain != agent.config["brain"]:
        # a new brain starts a new conversation; the clear goes to the loop before the events from set_config
        bus.loop.call_soon_threadsafe(bus.log.clear)
    try:
        agent.set_config(body.brain, body.vision)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return agent.config_info()


@app.get("/api/sys")
async def sys_stats():
    http, agent = state["http"], state["agent"]

    async def model_state(entry):
        """ready (vLLM is running), loaded (model in VRAM), idle (not in VRAM) or offline."""
        try:
            if entry.get("backend") == "vllm":
                ok = (await http.get(f"{entry['base']}/v1/models")).status_code == 200
                return "ready" if ok else "offline"
            loaded = (await http.get(f"{entry['base']}/api/ps")).json().get("models", [])
        except (httpx.HTTPError, ValueError):
            return "offline"
        return "loaded" if any(m.get("name", "").startswith(entry["model"]) for m in loaded) else "idle"

    brain, vision = await asyncio.gather(model_state(agent.brain), model_state(agent.vision))
    return {
        "gpu": await asyncio.to_thread(gpu_stats, state["gpu"]),
        "load": round(os.getloadavg()[0], 1),
        "brain": brain,
        "vision": vision,
        "busy": agent.busy,
    }


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning", timeout_graceful_shutdown=2)
