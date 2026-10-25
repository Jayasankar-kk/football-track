"""Follow the players the analyst clicks: no player detector, no PyTorch.

The analyst clicks a player (or drags a box around them); a tiny single-object tracker
(OpenCV ViTTrack, 0.7 MB ONNX) then follows that player frame by frame on the CPU
(~15-20 ms per marked player per 960 px frame on a low-end laptop).

    reader thread: decode -> resize -> update every marked player's tracker -> JPEG (thread pool)
    display (Session): paced at the video frame rate, never waits for anything heavy

Swap protection: the tracker's own confidence does not drop when it latches onto another
player at a crossing, but the shirt colour does. If the colour inside the tracked box stops
matching the clicked player for two checks in a row, the marking is hidden ("lost") instead
of being drawn on the wrong person; the analyst re-clicks to re-attach.
"""
from __future__ import annotations

import collections
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SOT_DIR = ROOT / "models"
DISPLAY_MAX_W = int(os.environ.get("FT_DISPLAY_W", "960"))
AHEAD_S = 2.5  # files: decode/track this far ahead of the display, so CPU hiccups don't freeze playback
RING_S = 12.0  # keep this much shown video (as JPEG) so a click on a paused/frozen frame can be used
CATCHUP_MAX_UPDATES = 12  # when attaching on an old frame, catch the tracker up with at most this many updates
COLOUR_CHECK_EVERY = 5  # frames
# Colour drop that means "this is no longer the clicked player": below COLOUR_DROP x the player's own
# usual similarity AND below COLOUR_MAX_SIM, twice in a row. Relative, because a far-away player's tiny
# torso gives a noisy signature (usual ~0.2) while a near one is stable (~0.7).
COLOUR_DROP = 0.4
COLOUR_MAX_SIM = 0.3
COLOUR_MIN_PX = 30  # boxes shorter than this are too small for a meaningful colour check
# Ring size + losing the player:
MERGE_GROW = 1.3  # tracker box suddenly this much bigger (or MERGE_SHRINK smaller) than the player's usual
MERGE_SHRINK = 0.65  # size = another player came close: keep the size and re-find the marked player inside it
SIZE_SMOOTH = 0.12  # how fast the ring size may follow the tracker per update (camera zoom is gradual)
CUT_SCORE = 0.3  # frame-to-frame colour change above this = camera cut (normal frames ~0.08, cuts 0.3-0.8)
TEMPLATE_MIN = 0.45  # appearance match needed to re-find / re-confirm the player
RECOVER_CHECKS = 3  # a hidden marking comes back only after this many confirmed updates in a row
LOST_TIMEOUT_S = 8.0  # hidden this long -> given up (re-click to follow again)
GRASS_HUE_TOL = 12  # click-to-box: pitch hue +- this
GRASS_V_LO = 0.80  # click-to-box: pixels darker than this x local grass brightness are player, not grass

VIT_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx"


def similarity(a, b) -> float:
    """Similarity of two normalised histograms (1 = identical, 0 = nothing in common)."""
    if a is None or b is None:
        return 0.5
    return 1.0 - float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


def _ensure_vit() -> Path:
    path = SOT_DIR / "vittrack.onnx"
    if not path.exists():  # 0.7 MB, downloaded once
        import urllib.request

        SOT_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(VIT_URL, path)
    return path


def make_tracker():
    p = cv2.TrackerVit_Params()
    p.net = str(_ensure_vit())
    return cv2.TrackerVit_create(p)


def _encode(frame):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else b""


def _grass_hue(frame):
    """Dominant green-ish hue of the pitch (OpenCV hue units, 0-179)."""
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = (s > 50) & (v > 40) & (h > 25) & (h < 95)
    return int(np.median(h[m])) if m.sum() > 200 else 50


def _player_mask(roi, hue):
    """Non-grass pixels. Grass = pitch hue AND about as bright as the grass around it, so dark kits
    (even dark green ones) and shadows are judged against the local pitch, not one global colour."""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(np.int16), hsv[..., 1], hsv[..., 2]
    green = (np.abs(h - hue) <= GRASS_HUE_TOL) & (s > 40)
    if green.sum() < 50:
        return np.zeros(roi.shape[:2], np.uint8)
    v_grass = np.median(v[green])
    grass = green & (v > GRASS_V_LO * v_grass) & (v < 1.6 * v_grass + 10)
    return (~grass).astype(np.uint8) * 255


def appearance(frame, box, hue=None):
    """Colour signature of a player's torso (hue x saturation x brightness histogram).

    Grass is removed by comparing against the pitch *around* the box (hue + brightness), so a
    dark-green kit is not mistaken for grass, and brightness is part of the signature so
    dark vs light kits of similar hue stay distinguishable."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 8:
        return None
    hue = _grass_hue(frame) if hue is None else hue
    ex0, ey0, ex1, ey1 = max(0, x1 - bw), max(0, y1 - bh // 2), min(W, x2 + bw), min(H, y2 + bh // 2)
    region = cv2.cvtColor(frame[ey0:ey1, ex0:ex1], cv2.COLOR_BGR2HSV)
    rh, rs, rv = region[..., 0].astype(np.int16), region[..., 1], region[..., 2]
    outside = np.ones(rh.shape, bool)
    outside[max(0, y1 - ey0):y2 - ey0, max(0, x1 - ex0):x2 - ex0] = False
    green = (np.abs(rh - hue) <= GRASS_HUE_TOL) & (rs > 40)
    ref = green & outside
    v_grass = np.median(rv[ref]) if ref.sum() > 20 else None
    tx1, tx2 = x1 + int(bw * 0.2), x2 - int(bw * 0.2)
    ty1, ty2 = y1 + int(bh * 0.15), y1 + int(bh * 0.55)
    torso = cv2.cvtColor(frame[max(0, ty1):min(H, ty2), max(0, tx1):min(W, tx2)], cv2.COLOR_BGR2HSV)
    if torso.shape[0] < 2 or torso.shape[1] < 2:
        return None
    th, ts, tv = torso[..., 0].astype(np.int16), torso[..., 1], torso[..., 2]
    is_grass = (np.abs(th - hue) <= GRASS_HUE_TOL) & (ts > 40)
    if v_grass is not None:
        is_grass &= (tv > GRASS_V_LO * v_grass) & (tv < 1.6 * v_grass + 10)
    mask = (~is_grass).astype(np.uint8)
    if mask.sum() < 4:
        return None
    hist = cv2.calcHist([torso], [0, 1, 2], mask, [12, 4, 4], [0, 180, 0, 256, 0, 256])
    return (hist / hist.sum()).flatten().astype(np.float32)


def box_from_click(frame, x, y):
    """Bounding box of the player under (x, y): the non-grass blob around the click."""
    H, W = frame.shape[:2]
    rh, rw = int(H * 0.2), int(H * 0.12)  # search window (players are at most ~40% of frame height here)
    x0, x1, y0, y1 = max(0, int(x) - rw), min(W, int(x) + rw), max(0, int(y) - rh), min(H, int(y) + rh)
    fg = _player_mask(frame[y0:y1, x0:x1], _grass_hue(frame))
    k = max(3, int(H / 270)) | 1  # removes pitch lines / grass texture, keeps bodies
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k - 1, k - 1)))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
    cx = min(max(int(x) - x0, 0), lab.shape[1] - 1)
    cy = min(max(int(y) - y0, 0), lab.shape[0] - 1)
    best = lab[cy, cx]
    if best == 0:  # click landed just off the body: snap to the nearest player pixel
        r = max(3, int(H * 0.015))
        ys, xs = np.nonzero(lab[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1])
        if len(xs):
            i = np.argmin((xs - min(cx, r)) ** 2 + (ys - min(cy, r)) ** 2)
            best = lab[max(0, cy - r) + ys[i], max(0, cx - r) + xs[i]]
    if best:
        bx, by, bw, bh = stats[best, :4]
        if H * 0.025 <= bh <= H * 0.45:
            if bw > 0.8 * bh:  # two players side by side: keep a player-shaped box around the click
                bw = int(0.45 * bh)
                bx = cx - bw // 2
            if bh > 4.5 * bw:  # two players stacked: keep the part around the click
                bh = int(3.0 * bw)
                by = max(by, cy - int(0.45 * bh))
            pad_x, pad_y = max(2, int(bw * 0.12)), max(2, int(bh * 0.08))
            return [x0 + bx - pad_x, y0 + by - pad_y, x0 + bx + bw + pad_x, y0 + by + bh + pad_y]
    # Fallback (no pitch around, e.g. close-up): a player-sized box, assuming the click hit the torso.
    bh = H * 0.1
    bw = bh * 0.45
    return [x - bw / 2, y - bh * 0.4, x + bw / 2, y + bh * 0.6]


def _patch(gray, cx, cy, w, h):
    """Grey pixels of a w x h box centred on (cx, cy), or None if it sticks out of the frame."""
    x1, y1 = int(round(cx - w / 2)), int(round(cy - h / 2))
    x2, y2 = x1 + int(round(w)), y1 + int(round(h))
    H, W = gray.shape
    if x1 < 0 or y1 < 0 or x2 > W or y2 > H or x2 - x1 < 4 or y2 - y1 < 6:
        return None
    return gray[y1:y2, x1:x2]


def _thumb_hist(frame):
    t = cv2.cvtColor(cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([t], [0, 1], None, [16, 8], [0, 180, 0, 256])
    return cv2.normalize(h, h)


class _Mark:
    """One followed player. The ring is drawn from (cx, cy, w, h): the centre follows the tracker, but the
    size changes only slowly, so a neighbour stretching the tracker box can't make the ring balloon."""

    def __init__(self, tracker, box, hist, hue, gray):
        self.tracker, self.hist, self.hue = tracker, hist, hue
        self.cx, self.cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        self.w, self.h = box[2] - box[0], box[3] - box[1]
        p = _patch(gray, self.cx, self.cy, self.w, self.h)
        self.template = p.astype(np.float32) if p is not None else None  # how the player looks (grey)
        self.state = "tracking"  # tracking | lost (hidden, may come back) | gone (hidden until re-clicked)
        self.bad = 0  # consecutive colour-check failures
        self.recover = 0  # consecutive confirmations while lost
        self.lost_since = None
        self.vel = (0.0, 0.0)  # px per frame
        self.usual_sim = None  # running average of this player's colour similarity
        self.last_idx = None
        self.box = list(box)

    def set_box(self):
        self.box = [self.cx - self.w / 2, self.cy - self.h / 2, self.cx + self.w / 2, self.cy + self.h / 2]

    def predicted(self, idx):
        n = idx - (self.last_idx if self.last_idx is not None else idx)
        dx, dy = self.vel[0] * n, self.vel[1] * n
        b = self.box
        return {"box": [round(float(v), 1) for v in (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy)],
                "state": self.state}

    def match(self, gray, cx, cy):
        """Normalised similarity (-1..1) between the remembered look and the box at (cx, cy)."""
        if self.template is None:
            return None
        p = _patch(gray, cx, cy, self.w, self.h)
        if p is None:
            return None
        t = cv2.resize(self.template, (p.shape[1], p.shape[0]), interpolation=cv2.INTER_AREA)
        return float(cv2.matchTemplate(p.astype(np.float32), t, cv2.TM_CCOEFF_NORMED)[0, 0])

    def relocate(self, gray, tbox, pred):
        """The tracker box swallowed a neighbour: find the marked player inside it (appearance match,
        preferring the expected position). Returns a centre or None."""
        if self.template is None:
            return None
        x, y, w, h = tbox
        H, W = gray.shape
        mx, my = self.w * 0.3, self.h * 0.3
        x1, y1 = int(max(0, x - mx)), int(max(0, y - my))
        x2, y2 = int(min(W, x + w + mx)), int(min(H, y + h + my))
        tw, th = int(round(self.w)), int(round(self.h))
        if x2 - x1 <= tw or y2 - y1 <= th or tw < 4 or th < 6:
            return None
        t = cv2.resize(self.template, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(gray[y1:y2, x1:x2].astype(np.float32), t, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.mgrid[0:res.shape[0], 0:res.shape[1]]
        cxs, cys = x1 + xs + tw / 2, y1 + ys + th / 2
        score = res - 0.4 * np.hypot(cxs - pred[0], cys - pred[1]) / max(self.h, 1)
        i = np.unravel_index(np.argmax(score), score.shape)
        if res[i] < TEMPLATE_MIN * 0.8:
            return None
        return float(cxs[i]), float(cys[i])

    def json(self):
        return {"box": [round(float(v), 1) for v in self.box], "state": self.state}


class FollowPlayer:
    """Plays one source and follows the marked players. Used by server/app.py (one per browser connection)."""

    def __init__(self, source):
        self.src = source
        self.fps = source.fps
        self.live = source.live
        self.speed = 1.0
        self.stride = 1
        self.cv = threading.Condition()
        self.encoder = ThreadPoolExecutor(max_workers=2, thread_name_prefix="jpeg")
        self.frames = collections.OrderedDict()  # idx -> {"jpeg": Future, "size": (w, h), "anchors": {...}}
        self.ring = collections.OrderedDict()  # shown idx -> Future[jpeg]
        self.marks: dict[str, _Mark] = {}  # touched only by the reader thread
        self.ops = collections.deque()  # attach/detach/reset requests for the reader thread
        self.done_ops = {}  # op id -> result
        self.display_idx = 0
        self.shown_idx = None
        self.shown_anchors: dict = {}
        self.eof = False
        self.gen = 0
        self.seek_req = None
        self.upd_times = collections.deque(maxlen=200)
        self.round_cost = 0.0  # seconds to update all marked players once (EMA)
        self.prev_hist = None  # for camera-cut detection
        self.last_round = -10**9
        self._stop = False
        _ensure_vit()
        self.reader = threading.Thread(target=self._read_loop, daemon=True, name="follow-reader")
        self.reader.start()

    # --- stats used by the Session header ------------------------------------------------
    def det_rate(self):
        """Tracker update rounds per second (one round = all marked players on one frame)."""
        cutoff = time.perf_counter() - 4.0
        return sum(1 for t in self.upd_times if t > cutoff) / 4.0

    # --- reader thread -------------------------------------------------------------------
    def _read_loop(self):
        max_ahead = max(3, int(AHEAD_S * self.fps))
        while not self._stop:
            with self.cv:
                if self.seek_req is not None:
                    self.src.seek(self.seek_req)
                    self.seek_req = None
                while (not self._stop and not self.ops and self.seek_req is None and not self.live
                       and (self.eof or len(self.frames) > max_ahead)):
                    self.cv.wait(0.05)
                if self._stop:
                    break
                ops = list(self.ops)
                self.ops.clear()
                gen = self.gen
            for op in ops:
                self._apply(op)
            if ops:
                continue
            if self.seek_req is not None:
                continue

            frame, idx = self.src.read()
            if frame is None:
                if not self.live:
                    with self.cv:
                        if gen == self.gen:
                            self.eof = True
                            self.cv.notify_all()
                continue
            h, w = frame.shape[:2]
            if w > DISPLAY_MAX_W:
                frame = cv2.resize(frame, (DISPLAY_MAX_W, round(h * DISPLAY_MAX_W / w)), interpolation=cv2.INTER_AREA)
            anchors = self._update_all(frame, idx)
            fut = self.encoder.submit(_encode, frame)
            with self.cv:
                if gen != self.gen:
                    continue
                # Only the JPEG is kept (~100 KB vs ~1.5 MB raw): a 2.5 s buffer costs ~6 MB.
                self.frames[idx] = {"jpeg": fut, "size": (frame.shape[1], frame.shape[0]), "anchors": anchors}
                self.cv.notify_all()

    def _update_all(self, frame, idx):
        """Update every marked player. With many markers on a slow CPU, trackers run every Nth frame
        (N adapts to the measured cost) and in-between frames continue each player's recent motion,
        so playback keeps its frame rate. (Running trackers in parallel threads is not safe with
        OpenCV 5's DNN engine: results got corrupted.)"""
        hist = _thumb_hist(frame)  # camera cut? (a hard change of the whole picture)
        cut = self.prev_hist is not None and cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA) > CUT_SCORE
        self.prev_hist = hist
        if not self.marks:
            return {}
        if cut:  # the marked players' positions mean nothing in the new shot: hide until re-clicked
            for m in self.marks.values():
                m.state = "gone"
        budget = 0.8 / (self.fps * max(self.speed, 0.1))  # seconds per frame we can spend tracking
        self.stride = max(1, min(4, int(np.ceil(self.round_cost / budget)))) if self.round_cost else 1
        if self.stride > 1 and idx - self.last_round < self.stride:
            return {aid: m.predicted(idx) for aid, m in self.marks.items()}
        t0 = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out = {aid: self._update_one(m, frame, idx, gray) for aid, m in self.marks.items()}
        cost = time.perf_counter() - t0
        self.round_cost = cost if not self.round_cost else 0.8 * self.round_cost + 0.2 * cost
        self.last_round = idx
        self.upd_times.append(time.perf_counter())
        return out

    def _update_one(self, m: _Mark, frame, idx, gray=None):
        if m.state == "gone":
            return m.json()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if gray is None else gray
        H, W = gray.shape
        ok, (x, y, w, h) = m.tracker.update(frame)
        ok = bool(ok) and w > 1 and h > 1
        n = max(1, idx - m.last_idx) if m.last_idx is not None else 1
        px, py = m.cx + m.vel[0] * n, m.cy + m.vel[1] * n  # where we expect the player

        # 1) Ring size: another player close by stretches the tracker box. Keep the player's size, find him
        #    inside the box, and restart the tracker on him alone so it lets go of the neighbour.
        merged = ok and (max(w / m.w, h / m.h) > MERGE_GROW or min(w / m.w, h / m.h) < MERGE_SHRINK)
        if not ok:  # tracker failed this frame (it may return a junk box): hold the expected position
            cx, cy = px, py
        elif merged:
            found = m.relocate(gray, (x, y, w, h), (px, py))
            cx, cy = found if found else (px, py)
            m.tracker.init(frame, (int(cx - m.w / 2), int(cy - m.h / 2), max(4, int(m.w)), max(6, int(m.h))))
        else:
            cx, cy = x + w / 2, y + h / 2
            m.w += SIZE_SMOOTH * (w - m.w)  # follows camera zoom slowly
            m.h += SIZE_SMOOTH * (h - m.h)
        jump = ok and math.hypot(cx - px, cy - py) > (1.2 * m.h + 0.04 * W) * n  # tracker teleported elsewhere
        near_edge = min(m.cx, m.cy, W - m.cx, H - m.cy) < 1.5 * max(m.w, m.h)  # where he was before this frame
        if ok and not jump:
            vx, vy = (cx - m.cx) / n, (cy - m.cy) / n
            m.vel = (0.6 * m.vel[0] + 0.4 * vx, 0.6 * m.vel[1] + 0.4 * vy)
        m.cx, m.cy, m.last_idx = cx, cy, idx
        m.set_box()

        # 2) Player left the picture (walked out over an edge, or the camera turned away from him):
        #    hide until re-clicked, because the tracker would latch onto someone else.
        bx1, by1, bx2, by2 = m.box
        vis_w, vis_h = max(0, min(bx2, W) - max(bx1, 0)), max(0, min(by2, H) - max(by1, 0))
        if ok and near_edge and (not (0 <= cx < W and 0 <= cy < H) or vis_w * vis_h < 0.5 * m.w * m.h):
            m.state = "gone"
            return m.json()

        # 3) Shirt-colour check (skipped for tiny far-away players).
        if ok and idx % COLOUR_CHECK_EVERY == 0 and m.h >= COLOUR_MIN_PX and m.hist is not None:
            cur = appearance(frame, m.box, m.hue)
            if cur is not None:
                sim = similarity(m.hist, cur)
                usual = m.usual_sim if m.usual_sim is not None else sim
                dropped = sim < COLOUR_DROP * usual and sim < COLOUR_MAX_SIM
                m.bad = m.bad + 1 if dropped else 0
                if not dropped and m.state == "tracking":
                    m.usual_sim = sim if m.usual_sim is None else 0.85 * m.usual_sim + 0.15 * sim
                    if sim > 0.5:  # adapt slowly to lighting changes
                        m.hist = 0.95 * m.hist + 0.05 * cur

        # 4) Hide when unsure; come back only after repeated confirmation.
        t = idx / self.fps
        if m.state == "tracking":
            if not ok or jump or m.bad >= 2:
                m.state, m.recover, m.lost_since = "lost", 0, t
            elif not merged and idx % 5 == 0:  # keep the remembered look up to date (pose, light)
                p = _patch(gray, m.cx, m.cy, m.w, m.h)
                if p is not None and m.template is not None:
                    tp = cv2.resize(m.template, (p.shape[1], p.shape[0]), interpolation=cv2.INTER_AREA)
                    m.template = 0.9 * tp + 0.1 * p.astype(np.float32)
        else:
            score = m.match(gray, m.cx, m.cy)
            confirmed = ok and not jump and not merged and m.bad == 0 and (score is None or score >= TEMPLATE_MIN)
            m.recover = m.recover + 1 if confirmed else 0
            if m.recover >= RECOVER_CHECKS:
                m.state, m.lost_since = "tracking", None
            elif t - (m.lost_since or t) > LOST_TIMEOUT_S:
                m.state = "gone"
        return m.json()

    def _frame_image(self, idx):
        """Pixels of frame idx if still available (buffered or recently shown, decoded from its JPEG)."""
        with self.cv:
            f = self.frames.get(idx)
            fut = f["jpeg"] if f is not None else self.ring.get(idx)
        if fut is None:
            return None
        return cv2.imdecode(np.frombuffer(fut.result(), np.uint8), cv2.IMREAD_COLOR)

    def _apply(self, op):
        kind = op[0]
        if kind == "attach":
            _, op_id, aid, n, point, box = op
            ok = self._attach(aid, n, point, box)
            with self.cv:
                self.done_ops[op_id] = ok
                self.cv.notify_all()
        elif kind == "detach":
            self.marks.pop(op[1], None)
            with self.cv:
                for f in self.frames.values():
                    f["anchors"].pop(op[1], None)
                self.shown_anchors.pop(op[1], None)
        elif kind == "reset":  # after a seek the trackers' last positions are meaningless: re-click to follow
            self.marks.clear()

    def _attach(self, aid, n, point, box):
        img = self._frame_image(n)
        if img is None:  # frame no longer available: use the newest shown frame
            with self.cv:
                n = self.shown_idx if self.shown_idx is not None else n
            img = self._frame_image(n)
            if img is None:
                return None
        if box is None:
            box = box_from_click(img, *point)
        H, W = img.shape[:2]
        x1, y1 = max(0, box[0]), max(0, box[1])
        x2, y2 = min(W - 1, box[2]), min(H - 1, box[3])
        if x2 - x1 < 3 or y2 - y1 < 3:
            return None
        tracker = make_tracker()
        tracker.init(img, (int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        hue = _grass_hue(img)
        m = _Mark(tracker, [x1, y1, x2, y2], appearance(img, [x1, y1, x2, y2], hue), hue,
                  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        m.last_idx = n

        # Catch up through frames already decoded after n (ahead buffer, or frames shown since a live freeze).
        with self.cv:
            later = sorted(k for k in list(self.ring) + list(self.frames) if k > n)
        step = max(1, len(later) // CATCHUP_MAX_UPDATES)
        todo = later[::step]
        if later and todo[-1] != later[-1]:
            todo.append(later[-1])
        for k in todo:
            frame = self._frame_image(k)
            if frame is None:
                continue
            a = self._update_one(m, frame, k)
            with self.cv:
                if k in self.frames:
                    self.frames[k]["anchors"][aid] = a
        self.marks[aid] = m
        with self.cv:  # frames between catch-up steps: hold the latest box
            for k in later:
                if k in self.frames and aid not in self.frames[k]["anchors"]:
                    self.frames[k]["anchors"][aid] = m.json()
            if n == self.shown_idx:
                self.shown_anchors[aid] = {"box": [round(float(v), 1) for v in (x1, y1, x2, y2)],
                                           "state": "tracking"}
        return [round(float(v), 1) for v in (x1, y1, x2, y2)]

    # --- display side --------------------------------------------------------------------
    def next_frame(self, timeout=0.5):
        deadline = time.perf_counter() + timeout
        with self.cv:
            while True:
                if self._stop:
                    return None
                if self.live and len(self.frames) > self.fps:  # fell behind a live feed: skip ahead
                    while len(self.frames) > 2:
                        k, f = self.frames.popitem(last=False)
                        self.ring[k] = f["jpeg"]
                n = next((k for k in self.frames if k >= self.display_idx), None)
                if n is not None and (self.shown_idx is not None or self.eof or self.live or len(self.frames) >= 3):
                    break
                left = deadline - time.perf_counter()
                if left <= 0:
                    return None
                self.cv.wait(min(left, 0.02))
            for k in [k for k in self.frames if k < n]:
                self.ring[k] = self.frames.pop(k)["jpeg"]
            f = self.frames.pop(n)
            self.ring[n] = f["jpeg"]
            while len(self.ring) > RING_S * self.fps:
                self.ring.popitem(last=False)
            self.shown_idx, self.display_idx = n, n + 1
            self.shown_anchors = dict(f["anchors"])
            self.cv.notify_all()
        w, h = f["size"]
        return n, f["jpeg"].result(), w, h, [], self.shown_anchors

    def finished(self):
        with self.cv:
            return self.eof and not self.frames and not self.live

    def current(self):
        with self.cv:
            return [], dict(self.shown_anchors)

    # --- control ---------------------------------------------------------------------------
    def attach_at(self, anchor_id, frame_idx, point=None, box=None, timeout=5.0):
        """Start following the player at `point` (or inside `box`) on frame `frame_idx`.
        Returns the initial box, or None if it could not be placed."""
        with self.cv:
            op_id = object()
            self.ops.append(("attach", op_id, anchor_id, frame_idx, point, box))
            self.cv.notify_all()
            deadline = time.perf_counter() + timeout
            while True:
                if op_id in self.done_ops:
                    return self.done_ops.pop(op_id)
                left = deadline - time.perf_counter()
                if left <= 0:
                    return None
                self.cv.wait(left)

    def detach(self, anchor_id):
        with self.cv:
            self.ops.append(("detach", anchor_id))
            self.cv.notify_all()

    def seek(self, frame_idx):
        if self.live:
            return
        with self.cv:
            self.gen += 1
            self.seek_req = frame_idx
            for f in self.frames.values():
                f["jpeg"].cancel()
            self.frames.clear()
            self.display_idx = frame_idx
            self.shown_idx = None
            self.shown_anchors = {}
            self.eof = False
            self.ops.append(("reset",))
            self.prev_hist = None
            self.cv.notify_all()

    def close(self):
        with self.cv:
            self._stop = True
            self.cv.notify_all()
        self.reader.join(timeout=3)
        self.encoder.shutdown(wait=False, cancel_futures=True)
        self.src.close()
