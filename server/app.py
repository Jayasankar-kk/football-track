"""Football telestrator server: markings that follow the players you click.

Run:  python server/app.py   (then open http://localhost:8000)

WebSocket protocol (/ws)
  client -> server (JSON text):
    {"type":"open", "kind":"file|webcam|url", "value":..., "simulate_live":bool}
    {"type":"pause"} {"type":"resume"} {"type":"step"} {"type":"seek","frame":n} {"type":"speed","value":0.5}
    {"type":"attach","anchor_id":"a1","frame":n,"point":[x,y]}   follow the player clicked at (x, y)
    {"type":"attach","anchor_id":"a1","frame":n,"box":[x1,y1,x2,y2]}   ... or the player inside a dragged box
    {"type":"detach","anchor_id":"a1"}
  server -> client:
    text   {"type":"info"|"error"|"status"|"attached", ...}
    binary [uint32 LE header length][header JSON][JPEG bytes (may be empty)]
           header = {frame, w, h, anchors:{id:{box,state}}, paused, live, fps, speed}
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import struct
import sys
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent))
from follow import FollowPlayer  # noqa: E402
from sources import open_source  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
VIDEOS = ROOT / "videos"
VIDEOS.mkdir(exist_ok=True)
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

app = FastAPI()


@app.middleware("http")
async def no_cache(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/videos")
def list_videos():
    return sorted(p.name for p in VIDEOS.iterdir() if p.suffix.lower() in VIDEO_EXT)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    name = Path(file.filename or "video.mp4").name
    if Path(name).suffix.lower() not in VIDEO_EXT:
        return {"error": "Unsupported file type"}
    with open(VIDEOS / name, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"name": name}


app.mount("/web", StaticFiles(directory=WEB), name="web")


class Session:
    """One browser connection: plays one source and follows the players marked on it."""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.player: FollowPlayer | None = None
        self.paused = False
        self.step_once = False
        self.refresh = False  # resend markings for the held frame
        self.speed = 1.0  # playback speed for files (0.1 .. 1)

    async def send_json(self, obj):
        await self.ws.send_text(json.dumps(obj))

    async def send_frame(self, idx, jpeg, w, h, anchors):
        p = self.player
        header = json.dumps({
            "frame": idx, "w": w, "h": h, "anchors": anchors,
            "paused": self.paused, "live": p.live, "fps": p.fps, "speed": p.speed,
        }).encode()
        await self.ws.send_bytes(struct.pack("<I", len(header)) + header + (jpeg or b""))

    async def handle(self, msg: dict):
        kind = msg.get("type")
        p = self.player
        if kind == "open":
            await self.open(msg)
        elif kind == "pause":
            self.paused = True
            self.refresh = True
        elif kind == "resume":
            self.paused = False
        elif kind == "step":
            self.step_once = True
        elif kind == "seek" and p:
            await asyncio.to_thread(p.seek, int(msg.get("frame", 0)))
            self.step_once = True
        elif kind == "speed":
            self.speed = min(1.0, max(0.1, float(msg.get("value", 1.0))))
            if p and not p.live:
                p.speed = self.speed
        elif kind == "attach" and p:
            # Setting up a tracker takes a moment: do it in the background so pause/play/record stay instant.
            asyncio.create_task(self.attach(p, msg))
        elif kind == "detach" and p:
            await asyncio.to_thread(p.detach, str(msg["anchor_id"]))
            self.refresh = True

    async def attach(self, p, msg):
        try:
            aid = str(msg["anchor_id"])
            frame = int(msg.get("frame", p.shown_idx or 0))
            box = await asyncio.to_thread(p.attach_at, aid, frame, msg.get("point"), msg.get("box"))
            if box is None:
                await self.send_json({"type": "error", "message": "No player found there: try dragging a box around the player"})
            await self.send_json({"type": "attached", "anchor_id": aid, "frame": frame, "box": box})
            self.refresh = True
        except Exception as e:  # socket closed / source switched meanwhile
            print("attach failed:", repr(e))

    async def open(self, msg):
        if self.player:
            old, self.player = self.player, None
            await asyncio.to_thread(old.close)
        try:
            src = await asyncio.to_thread(open_source, msg.get("kind"), str(msg.get("value", "")),
                                          bool(msg.get("simulate_live")), VIDEOS)
            self.player = await asyncio.to_thread(FollowPlayer, src)
        except Exception as e:  # report any open failure to the UI
            await self.send_json({"type": "error", "message": str(e)})
            return
        if not src.live:
            self.player.speed = self.speed
        self.paused = False
        await self.send_json({"type": "info", "live": src.live, "fps": src.fps, "total": src.total})

    async def loop(self):
        while True:
            try:
                await self._loop()
            except (WebSocketDisconnect, asyncio.CancelledError):
                return
            except Exception as e:  # keep the session alive on a bad frame / source error
                if self.ws.client_state.name != "CONNECTED":
                    return
                print("playback error:", repr(e))
                try:
                    await self.send_json({"type": "error", "message": f"Playback error: {e}"})
                except Exception:
                    return
                await asyncio.sleep(0.2)

    async def _loop(self):
        next_t = time.perf_counter()
        while True:
            p = self.player
            if p is None:
                await asyncio.sleep(0.05)
                continue

            if self.paused and not p.live and not self.step_once:
                if self.refresh and p.shown_idx is not None:
                    self.refresh = False
                    _, anchors = await asyncio.to_thread(p.current)
                    await self.send_frame(p.shown_idx, b"", 0, 0, anchors)
                await asyncio.sleep(0.03)
                next_t = time.perf_counter()
                continue

            item = await asyncio.to_thread(p.next_frame, 0.5)
            if p is not self.player:
                continue  # source switched while waiting
            if item is None:
                if p.finished() and not self.paused:
                    self.paused = True
                    await self.send_json({"type": "status", "message": "End of video"})
                    self.refresh = True
                next_t = time.perf_counter()
                continue
            self.step_once = False
            idx, jpeg, w, h, _, anchors = item
            # A paused live feed keeps tracking but the browser keeps the frozen picture.
            await self.send_frame(idx, b"" if (self.paused and p.live) else jpeg, w, h, anchors)

            # Pace playback at the source frame rate x speed; if we fell behind, don't burst to catch up.
            next_t += 1.0 / (p.fps * (1.0 if p.live else p.speed))
            delay = next_t - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            elif delay < -0.25:
                next_t = time.perf_counter()

    async def close(self):
        if self.player:
            await asyncio.to_thread(self.player.close)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    session = Session(ws)
    loop_task = asyncio.create_task(session.loop())
    try:
        while True:
            await session.handle(json.loads(await ws.receive_text()))
    except WebSocketDisconnect:
        pass
    finally:
        loop_task.cancel()
        await session.close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PORT", "8000"))
    print(f"Football telestrator on http://localhost:{port}")
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=port, log_level="warning")
