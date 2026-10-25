"""Video sources: seekable files, and live sources (webcam / RTSP / HTTP / simulated)."""
from __future__ import annotations

import collections
import threading
import time

import cv2


class FileSource:
    """Plays a file frame-by-frame; the server controls pacing, pause and seek."""

    live = False

    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.pos = 0

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None, None
        idx = self.pos
        self.pos += 1
        return frame, idx

    def seek(self, frame_idx: int):
        frame_idx = max(0, min(int(frame_idx), max(self.total - 1, 0)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self.pos = frame_idx

    def close(self):
        self.cap.release()


class LiveSource:
    """Grabs frames on a background thread into a short queue.

    Every frame is handed out in order (so playback is smooth); if the consumer
    falls more than `max_buffer` frames behind, the oldest frames are dropped so
    the feed stays close to real time. `simulate=True` replays a file at its native
    frame rate (looping) so live behaviour can be tested without a camera.
    """

    live = True

    def __init__(self, spec, simulate: bool = False):
        self.spec, self.simulate = spec, simulate
        self.cap = self._open()
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        if self.fps > 120 or self.fps < 1:
            self.fps = 25.0
        self.total = 0
        self.max_buffer = int(self.fps)
        self._buf = collections.deque()
        self.dropped = 0
        self._cond = threading.Condition()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _open(self):
        if isinstance(self.spec, int):
            cap = cv2.VideoCapture(self.spec, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(self.spec)
            if not cap.isOpened():
                cap = cv2.VideoCapture(self.spec)
        else:
            cap = cv2.VideoCapture(self.spec)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open live source: {self.spec}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _run(self):
        idx, failures, next_t = 0, 0, time.perf_counter()
        while not self._stop:
            ok, frame = self.cap.read()
            if not ok:
                if self.simulate:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                failures += 1
                if failures > 50:  # try to reconnect a dropped stream
                    self.cap.release()
                    time.sleep(1.0)
                    try:
                        self.cap = self._open()
                    except RuntimeError:
                        pass
                    failures = 0
                time.sleep(0.02)
                continue
            failures = 0
            with self._cond:
                self._buf.append((frame, idx))
                while len(self._buf) > self.max_buffer:
                    self._buf.popleft()
                    self.dropped += 1
                self._cond.notify_all()
            idx += 1
            if self.simulate:
                next_t += 1.0 / self.fps
                delay = next_t - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.perf_counter()

    def read(self, timeout: float = 2.0):
        with self._cond:
            if not self._cond.wait_for(lambda: self._buf or self._stop, timeout) or not self._buf:
                return None, None
            return self._buf.popleft()

    def seek(self, frame_idx: int):
        pass

    def close(self):
        self._stop = True
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=2)
        self.cap.release()


def open_source(kind: str, value: str, simulate_live: bool, video_dir):
    if kind == "file":
        path = str((video_dir / value).resolve())
        if not path.startswith(str(video_dir.resolve())):
            raise RuntimeError("Invalid file")
        return LiveSource(path, simulate=True) if simulate_live else FileSource(path)
    if kind == "webcam":
        return LiveSource(int(value or 0))
    if kind == "url":
        return LiveSource(value)
    raise RuntimeError(f"Unknown source kind: {kind}")
