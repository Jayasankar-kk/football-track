# Football Telestrator

Pause a football video (or a live feed), click a player, and drop a marking — ring, spotlight,
name tag, arrow, link line or zone. The marking **follows the player** as the video plays, sits
**under** the player's feet, and the result can be **saved as an MP4** to share.

No GPU and no AI model downloads: a tiny tracker (OpenCV ViTTrack, 0.7 MB, included) follows each
clicked player on the CPU.

## Run

Requires **Python 3.9+** and **Chrome or Edge**.

**Windows:** double-click `run.bat`. The first run downloads and installs the requirements (~70 MB download); the app then
opens at http://localhost:8000.

**Any OS:**

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # Windows: .venv\Scripts\pip install -r requirements.txt
.venv/bin/python server/app.py                   # Windows: .venv\Scripts\python server\app.py
```

Then open http://localhost:8000.

## Use

1. Choose a source: **Upload…** a video file (or copy it into `videos/`), a webcam / capture card,
   or a stream URL (RTSP / HTTP / HLS) — and press **Start**.
2. Press **Space** to pause, pick a tool and **click a player** — or **drag a box** around them for
   a precise start.
3. Press **Space** to resume: the marking follows the player.
4. Press **● Rec**, play / pause / draw as usual, then **■ Stop** — the MP4 is saved to your
   Downloads folder.

| Key | Action |
|---|---|
| `1` Ring · `2` Spotlight · `3` Tag | click a player (or drag a box around them) |
| `4` Arrow | drag from a player (hold **Shift** for a dashed arrow) |
| `5` Link · `6` Zone | click players, **Enter** / double-click to finish |
| `V` Select | click a marking; **Del** deletes it, **R** re-attaches it to a player |
| `Space` · `→` | pause / resume · next frame |
| `Esc`, right-click | cancel |

The **speed** selector (0.25×–1×) plays files in slow motion — handy for fast moments.

## How it behaves

* **Click → player:** the player is found as the "not grass" shape around the click. If a click lands
  badly, drag a box around the player instead.
* **Stable rings:** when another player comes close, the ring keeps its size and stays on the clicked player.
* **Markings disappear instead of jumping to someone else:**
  * *off screen · R* — camera cut, or the player left the picture. When he is back, select the marking,
    press **R** and click him.
  * *hidden* — hidden behind another player / not confirmed. Comes back by itself once he is found again.
* **Swaps:** two players with the same shirt colour crossing can occasionally swap — press **R** and click
  the right player.
* **After a seek**, re-click the players.

## Recording

* Frame-accurate: every video frame is written with its real duration, so a busy computer never shows up
  as slow motion in the saved clip. Pauses (freeze-frames with live drawing) are recorded at real time.
* MP4 (H.264) in Chrome / Edge; plays on phones, WhatsApp, email and PowerPoint. Other browsers record WebM.
* Keeps recording when the window is minimised. No audio is recorded.

## Settings (environment variables)

| Variable | Default | |
|---|---|---|
| `FT_DISPLAY_W` | `960` | Width of the video shown and recorded. `1280` or `1920` = sharper, more CPU. |
| `PORT` / `HOST` | `8000` / `127.0.0.1` | The port can also be given as an argument: `python server/app.py 8001`. `HOST=0.0.0.0` makes it reachable from other machines. |

Example (Windows): `set FT_DISPLAY_W=1280 && .venv\Scripts\python server\app.py`

## Project layout

```
server/app.py      web server: video upload/list, WebSocket playback + controls
server/follow.py   decoding, the per-player trackers, click → player box, swap / cut / off-screen handling
server/sources.py  video files, webcams and live streams
web/index.html     the page
web/app.js         drawing tools, markings list, playback controls, recording
web/draw.js        how markings are drawn, and keying them under the players
models/vittrack.onnx  the tracker model
```

Third-party licenses: see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
