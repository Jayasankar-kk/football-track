import { TOOL_NAMES, drawAnnotation, drawPending, keyLayer, calibrateGrass, hitAnnotation } from './draw.js';

const $ = (id) => document.getElementById(id);
const COLORS = ['#ffd400', '#ff3b3b', '#2ea8ff', '#3ddc84', '#ffffff', '#ff7ad9', '#ff8c1a', '#9b6bff'];
const PLAYER_TOOLS = new Set(['ring', 'spotlight', 'tag', 'arrow', 'link', 'zone']);
const GESTURE_TOOLS = new Set(['ring', 'spotlight', 'tag', 'link', 'zone']); // click or drag a box on a player
const GROUND_TYPES = new Set(['ring', 'spotlight', 'arrow', 'link', 'zone']); // drawn on the pitch, under players

const S = {
  ws: null,
  live: false,
  paused: false,
  total: 0,
  videoFps: 25, // the video's frame rate
  fps: 0, // measured display rate
  fpsCount: 0,
  view: null, // what is on screen: {w, h, frame, anchors}
  latest: null, // newest header from the server (may be ahead of the view while a live feed is paused)
  seq: 0,
  applied: 0,
  anns: [], // markings: {id, type, color, anchors:[anchorId], text?, offsetPx?, dashed?}
  tool: 'select',
  color: COLORS[0],
  sel: null,
  pending: null, // link/zone being built {type, targets:[]} | arrow being dragged {type:'arrow', target, start, cur}
  drag: null, // click-or-drag gesture on a player {start, cur}
  mouse: null,
  reassign: false,
  frozen: {}, // anchorId -> {frame, box} placed on the paused/frozen frame
  smooth: {},
  grass: { hue: 110, tol: 30 },
  needCalib: true,
  maskCache: new Map(),
  nextId: 1,
  seeking: false,
  bmp: null,
  frameReady: false,
  dirty: true,
  lastDraw: 0,
};
// Any input may change what is drawn (previews, selection).
for (const ev of ['mousemove', 'mousedown', 'mouseup', 'keydown', 'click', 'input']) {
  window.addEventListener(ev, () => { S.dirty = true; }, true);
}
setInterval(() => { S.fps = S.fpsCount; S.fpsCount = 0; }, 1000);

// --- canvases ---------------------------------------------------------------------
const view = $('view');
const ctx = view.getContext('2d');
const frameCanvas = document.createElement('canvas'); // CPU copy of the frame, read by the keying
const fctx = frameCanvas.getContext('2d', { willReadFrequently: true });
const groundCanvas = document.createElement('canvas'); // markings painted on the pitch
const gctx = groundCanvas.getContext('2d', { willReadFrequently: true });

function setSize(w, h) {
  if (view.width === w && view.height === h) return;
  for (const c of [view, frameCanvas, groundCanvas]) { c.width = w; c.height = h; }
}

// --- websocket ----------------------------------------------------------------------
const outbox = []; // "open" requests sent before the socket is ready
function connect() {
  const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => {
    status('Connected — pick a source and press Start');
    while (outbox.length) ws.send(JSON.stringify(outbox.shift()));
  };
  ws.onclose = () => { status('Disconnected — retrying…', true); setTimeout(connect, 1500); };
  ws.onmessage = (e) => (typeof e.data === 'string' ? onControl(JSON.parse(e.data)) : onFrame(e.data));
  S.ws = ws;
}

function send(obj) {
  if (S.ws?.readyState === WebSocket.OPEN) S.ws.send(JSON.stringify(obj));
  else if (obj.type === 'open') outbox.push(obj);
}

function onControl(msg) {
  if (msg.type === 'error') status(msg.message, true);
  else if (msg.type === 'status') status(msg.message);
  else if (msg.type === 'attached') {
    if (msg.box) S.frozen[msg.anchor_id] = { frame: msg.frame, box: msg.box };
    S.dirty = true;
  } else if (msg.type === 'info') {
    S.live = msg.live;
    S.total = msg.total;
    S.videoFps = msg.fps;
    S.needCalib = true;
    $('seek').max = Math.max(0, msg.total - 1);
    $('seek').disabled = msg.live;
    $('stepBtn').disabled = msg.live;
    $('speed').disabled = msg.live;
    if (!msg.live) send({ type: 'speed', value: +$('speed').value });
    $('liveBadge').hidden = !msg.live;
    $('hint').hidden = true;
    status(`${msg.live ? 'Live source running' : 'Video loaded'} — pick a tool, then click a player (or drag a box around them)`);
  }
}

async function onFrame(buf) {
  const seq = ++S.seq;
  const hl = new DataView(buf).getUint32(0, true);
  const hdr = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl)));
  const jpgLen = buf.byteLength - 4 - hl;
  S.latest = hdr;
  S.live = hdr.live;
  if (S.paused !== hdr.paused) {
    S.paused = hdr.paused;
    if (!S.paused) S.frozen = {};
  }
  updateTransport(hdr);

  if (jpgLen > 0) {
    const bmp = await createImageBitmap(new Blob([new Uint8Array(buf, 4 + hl)], { type: 'image/jpeg' }));
    if (seq < S.applied) { bmp.close(); return; } // an older frame finished decoding late
    S.applied = seq;
    setView(hdr, bmp);
  } else if (S.view && !hdr.live) {
    // Paused file: markings were added/removed on the frame that is being held.
    S.view.anchors = hdr.anchors;
    S.dirty = true;
  }
}

function setView(hdr, bmp) {
  setSize(hdr.w, hdr.h);
  S.bmp?.close();
  S.bmp = bmp; // drawn straight to the screen; copied to the CPU canvas only when keying needs pixels
  S.frameReady = false;
  S.maskCache.clear();
  S.dirty = true;
  S.fpsCount++;
  const smoothOn = $('optSmooth').checked && !hdr.paused;
  for (const [aid, a] of Object.entries(hdr.anchors)) {
    const prev = S.smooth[aid];
    const b = a.box;
    const jump = !prev || Math.hypot((prev[0] + prev[2]) / 2 - (b[0] + b[2]) / 2, prev[3] - b[3]) > (b[3] - b[1]) * 0.8;
    S.smooth[aid] = !smoothOn || jump ? b.slice() : prev.map((v, i) => v + (b[i] - v) * 0.6);
  }
  S.view = { w: hdr.w, h: hdr.h, frame: hdr.frame, anchors: hdr.anchors };
  if (rec.active) draw(performance.now(), true); // record every frame the moment it arrives
  if (S.needCalib) autoCalibrate(false);
}

// --- where each marking's player is ------------------------------------------------------
function anchorBox(aid) {
  const v = S.view;
  if (!v) return null;
  if (S.paused && S.frozen[aid] && S.frozen[aid].frame === v.frame) {
    return { box: S.frozen[aid].box, state: 'tracking' }; // just placed on the frozen frame
  }
  const a = v.anchors[aid];
  if (!a) return null;
  return { box: S.smooth[aid] || a.box, state: a.state };
}

const resolveAnn = (ann) => ann.anchors.map(anchorBox);

function annState(ann) {
  const src = S.latest?.anchors || {};
  const states = ann.anchors.map((aid) => (S.frozen[aid] && S.paused ? 'tracking' : src[aid]?.state || 'pending'));
  for (const s of ['gone', 'lost', 'pending']) if (states.includes(s)) return s;
  return 'tracking';
}

// --- rendering -----------------------------------------------------------------------
function frameCtx() {
  if (!S.frameReady && S.bmp) {
    fctx.drawImage(S.bmp, 0, 0);
    S.frameReady = true;
  }
  return fctx;
}

// Redraw only when something changed: a new frame, input, or (at ~30 fps) animated markings.
function render(ts) {
  requestAnimationFrame(render);
  draw(ts, false);
}

// force=true draws now even without a screen refresh (used while recording, so a hidden or
// minimised window still records every frame).
function draw(ts, force) {
  const v = S.view;
  if (!v || !S.bmp) return;
  const animating = S.anns.length > 0 || S.pending || rec.active;
  if (!force && !S.dirty && !(animating && ts - S.lastDraw > 33)) return;
  S.dirty = false;
  S.lastDraw = ts;
  const u = v.h / 720, t = ts / 1000;
  ctx.drawImage(S.bmp, 0, 0);

  const resolved = S.anns.map((a) => [a, resolveAnn(a)]);
  const showLost = $('optLost').checked;
  if (S.pending || S.anns.some((a) => GROUND_TYPES.has(a.type))) {
    gctx.clearRect(0, 0, v.w, v.h);
    for (const [a, r] of resolved) drawAnnotation(gctx, 'ground', a, r, u, t, showLost);
    if (S.pending) {
      const boxes = (S.pending.type === 'arrow' ? [S.pending.target] : S.pending.targets).map(targetBox);
      drawPending(gctx, S.pending, boxes, S.pending.type === 'arrow' ? null : S.mouse, S.color, u, t);
    }
    // Keying: cut the pitch markings where the followed players' bodies are, so they appear underneath.
    const people = resolved.flatMap(([a, r]) => r.filter((x) => x && x.state === 'tracking')
      .map((x, i) => ({ id: `${a.id}:${i}`, box: x.box, cls: 'player' })));
    if ($('optKey').checked) keyLayer(gctx, frameCtx(), people, S.grass, v.w, v.h, S.maskCache);
    ctx.drawImage(groundCanvas, 0, 0);
  }
  for (const [a, r] of resolved) drawAnnotation(ctx, 'top', a, r, u, t, showLost);

  captureRecordingFrame();
  if (rec.active) return; // recording: leave UI-only overlays (drag rectangle, selection) out of the video
  if (S.drag && dragged(S.drag)) {
    const [x1, y1, x2, y2] = dragBox(S.drag);
    ctx.save();
    ctx.strokeStyle = S.color;
    ctx.setLineDash([5 * u, 4 * u]);
    ctx.lineWidth = 1.5 * u;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.restore();
  }
  const ann = S.anns.find((a) => a.id === S.sel);
  if (ann) {
    ctx.save();
    ctx.strokeStyle = '#fff';
    ctx.setLineDash([6 * u, 4 * u]);
    ctx.lineWidth = 1.5 * u;
    for (const r of resolveAnn(ann)) {
      if (!r || r.state === 'gone') continue;
      const b = r.box;
      ctx.strokeRect(b[0] - 6 * u, b[1] - 6 * u, b[2] - b[0] + 12 * u, b[3] - b[1] + 12 * u);
    }
    ctx.restore();
  }
}

// --- interaction ---------------------------------------------------------------------
function toFrame(e) {
  const r = view.getBoundingClientRect();
  return [((e.clientX - r.left) / r.width) * view.width, ((e.clientY - r.top) / r.height) * view.height];
}

// A "target" is what was pointed at: a click {point} or a dragged box {box}. The server turns it into
// a followed player.
function targetBox(t) {
  if (!t) return null;
  if (t.box) return t.box;
  const h = (S.view?.h || 540) * 0.08, [x, y] = t.point; // rough player-sized box for previews
  return [x - h * 0.22, y - h * 0.4, x + h * 0.22, y + h * 0.6];
}

const dragged = (d) => Math.hypot(d.cur[0] - d.start[0], d.cur[1] - d.start[1]) > 8 * ((S.view?.h || 540) / 720);
const dragBox = (d) => [Math.min(d.start[0], d.cur[0]), Math.min(d.start[1], d.cur[1]),
  Math.max(d.start[0], d.cur[0]), Math.max(d.start[1], d.cur[1])];

function sendAttach(aid, target) {
  send({ type: 'attach', anchor_id: aid, frame: S.view.frame, ...(target.box ? { box: target.box } : { point: target.point }) });
}

function newAnchor(target) {
  const aid = `a${S.nextId++}`;
  sendAttach(aid, target);
  return aid;
}

function addAnn(ann) {
  ann.id = `m${S.nextId++}`;
  ann.color ||= S.color;
  S.anns.push(ann);
  S.sel = ann.id;
  renderList();
}

function removeAnn(id) {
  const ann = S.anns.find((a) => a.id === id);
  if (!ann) return;
  for (const aid of ann.anchors) {
    send({ type: 'detach', anchor_id: aid });
    delete S.smooth[aid];
    delete S.frozen[aid];
  }
  S.anns = S.anns.filter((a) => a.id !== id);
  if (S.sel === id) S.sel = null;
  renderList();
}

function finishPending() {
  const p = S.pending;
  if (!p || p.type === 'arrow') return;
  const need = p.type === 'zone' ? 3 : 2;
  if (p.targets.length >= need) addAnn({ type: p.type, anchors: p.targets.map(newAnchor) });
  else status(`${TOOL_NAMES[p.type]} needs at least ${need} players`, true);
  S.pending = null;
}

function reassignTo(target) {
  const ann = S.anns.find((a) => a.id === S.sel);
  S.reassign = false;
  if (!ann) return;
  // Re-attach the first player of the marking that is not followed any more (or the only one).
  const src = S.latest?.anchors || {};
  let i = ann.anchors.findIndex((aid) => src[aid] && src[aid].state !== 'tracking');
  if (i < 0) i = ann.anchors.length === 1 ? 0 : -1;
  if (i < 0) { status('All players of this marking are followed — delete and redraw to change it'); return; }
  const aid = ann.anchors[i];
  delete S.smooth[aid];
  delete S.frozen[aid];
  sendAttach(aid, target);
  status('Re-attached');
  renderList();
}

// What a finished click/drag on a player does with the current tool.
function actOn(target) {
  if (S.reassign) { reassignTo(target); return; }
  switch (S.tool) {
    case 'ring':
    case 'spotlight':
      addAnn({ type: S.tool, anchors: [newAnchor(target)] });
      break;
    case 'tag': {
      const text = prompt('Tag text (player name / number):', '');
      if (text) addAnn({ type: 'tag', text, anchors: [newAnchor(target)] });
      break;
    }
    case 'link':
    case 'zone': {
      if (!S.pending || S.pending.type !== S.tool) S.pending = { type: S.tool, targets: [] };
      const ts = S.pending.targets;
      const same = (a, b) => Math.hypot(targetBox(a)[0] - targetBox(b)[0], targetBox(a)[1] - targetBox(b)[1]) < 6;
      if (ts.length && same(ts[0], target)) finishPending();
      else if (!ts.some((x) => same(x, target))) ts.push(target);
      break;
    }
  }
}

view.addEventListener('mousemove', (e) => {
  S.mouse = toFrame(e);
  if (S.pending?.type === 'arrow') S.pending.cur = S.mouse;
  if (S.drag) S.drag.cur = S.mouse;
});
view.addEventListener('mouseleave', () => { S.mouse = null; });

view.addEventListener('mousedown', (e) => {
  if (e.button !== 0 || !S.view) return;
  const p = toFrame(e);
  if (S.reassign || GESTURE_TOOLS.has(S.tool)) {
    S.drag = { start: p, cur: p }; // resolved on mouseup: a click, or a dragged box around the player
  } else if (S.tool === 'arrow') {
    S.pending = { type: 'arrow', target: { point: p }, start: p, cur: p };
  } else if (S.tool === 'select') {
    const u = S.view.h / 720;
    const hit = [...S.anns].reverse().find((a) => hitAnnotation(a, resolveAnn(a), p, u));
    S.sel = hit?.id ?? null;
    renderList();
  }
});

window.addEventListener('mouseup', (e) => {
  if (S.drag) {
    const d = S.drag;
    S.drag = null;
    if (S.view) actOn(dragged(d) ? { box: dragBox(d) } : { point: d.start });
    return;
  }
  const p = S.pending;
  if (p?.type !== 'arrow' || !S.view) return;
  S.pending = null;
  const end = toFrame(e);
  if (Math.hypot(end[0] - p.start[0], end[1] - p.start[1]) < 15 * (S.view.h / 720)) return; // just a click
  // The arrow keeps the dragged direction and length, starting at the player's feet. Shift = dashed.
  addAnn({ type: 'arrow', offsetPx: [end[0] - p.start[0], end[1] - p.start[1]], dashed: e.shiftKey, anchors: [newAnchor(p.target)] });
});

view.addEventListener('dblclick', () => finishPending());
view.addEventListener('contextmenu', (e) => { e.preventDefault(); S.pending = null; S.reassign = false; S.drag = null; });

function setTool(tool) {
  S.tool = tool;
  S.pending = null;
  S.reassign = false;
  document.querySelectorAll('#tools [data-tool]').forEach((b) => b.classList.toggle('active', b.dataset.tool === tool));
  $('stage').classList.toggle('picking', PLAYER_TOOLS.has(tool));
}

function togglePause() {
  send({ type: S.paused ? 'resume' : 'pause' });
}

window.addEventListener('keydown', (e) => {
  if (e.target.matches('input[type=text], input[type=number], select')) return;
  const k = e.key;
  if (k === ' ') { e.preventDefault(); document.activeElement?.blur?.(); togglePause(); }
  else if (k === 'Escape') { S.pending = null; S.reassign = false; S.drag = null; }
  else if (k === 'Enter') finishPending();
  else if ((k === 'Delete' || k === 'Backspace') && S.sel) removeAnn(S.sel);
  else if (k === 'ArrowRight' && !S.live) send({ type: 'step' });
  else if ((k === 'r' || k === 'R') && S.sel) { S.reassign = true; status('Click the player to attach the selected marking to'); }
  else if (k === 'v' || k === 'V') setTool('select');
  else {
    const tools = ['ring', 'spotlight', 'tag', 'arrow', 'link', 'zone'];
    const n = parseInt(k, 10);
    if (n >= 1 && n <= tools.length) setTool(tools[n - 1]);
  }
});

// --- recording: save the video with markings to share ------------------------------------
// Frame-accurate: every video frame is written with its real duration (1/fps, or longer if you
// deliberately play at 0.5x), however late it reached the browser, so a busy computer never turns
// into slow motion in the saved clip. While paused, the frozen picture and live drawing are recorded
// at real time. Uses WebCodecs + mp4-muxer (Chrome/Edge); other browsers fall back to real-time
// screen recording (MediaRecorder).
const rec = {
  active: false, mode: null, encoder: null, muxer: null, t: 0, lastFrame: null, lastWall: 0, lastPaused: false,
  frames: 0, recorder: null, chunks: [], ext: 'mp4', t0: 0, timer: null, pauseTimer: null,
};

async function pickH264(width, height) {
  if (!window.VideoEncoder || !window.Mp4Muxer) return null;
  for (const codec of ['avc1.640028', 'avc1.4d0028', 'avc1.42002a', 'avc1.42001f']) {
    const cfg = { codec, width, height, bitrate: 6_000_000, framerate: 30, avc: { format: 'avc' } };
    try {
      if ((await VideoEncoder.isConfigSupported(cfg)).supported) return cfg;
    } catch { /* try the next profile */ }
  }
  return null;
}

async function startRecording() {
  if (!S.view) { status('Start a video first', true); return; }
  const width = view.width & ~1, height = view.height & ~1; // H.264 needs even sizes
  const cfg = await pickH264(width, height);
  if (cfg) {
    rec.muxer = new Mp4Muxer.Muxer({
      target: new Mp4Muxer.ArrayBufferTarget(),
      video: { codec: 'avc', width, height },
      fastStart: 'in-memory',
      firstTimestampBehavior: 'offset',
    });
    rec.encoder = new VideoEncoder({
      output: (chunk, meta) => rec.muxer.addVideoChunk(chunk, meta),
      error: (e) => status(`Recording error: ${e.message}`, true),
    });
    rec.encoder.configure(cfg);
    rec.mode = 'frames';
    rec.ext = 'mp4';
  } else if (window.MediaRecorder) {
    const options = ['video/mp4;codecs=avc1', 'video/mp4', 'video/webm;codecs=vp9', 'video/webm'];
    const mime = options.find((m) => MediaRecorder.isTypeSupported(m)) || '';
    rec.recorder = new MediaRecorder(view.captureStream(30), { mimeType: mime, videoBitsPerSecond: 8_000_000 });
    rec.ext = mime.includes('mp4') ? 'mp4' : 'webm';
    rec.chunks = [];
    rec.recorder.ondataavailable = (e) => { if (e.data.size) rec.chunks.push(e.data); };
    rec.recorder.start(1000);
    rec.mode = 'realtime';
  } else {
    status('This browser cannot record video (use Chrome or Edge)', true);
    return;
  }
  Object.assign(rec, { active: true, t: 0, frames: 0, lastFrame: null, lastWall: performance.now(), lastPaused: S.paused, t0: performance.now() });
  $('recBtn').classList.add('on');
  const tick = () => {
    const s = Math.floor(rec.mode === 'frames' ? rec.t : (performance.now() - rec.t0) / 1000);
    $('recBtn').textContent = `■ Stop ${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
  };
  tick();
  rec.timer = setInterval(tick, 500);
  // While paused there are no new frames: keep recording the frozen picture (and live drawing) at 30 fps.
  rec.pauseTimer = setInterval(() => { if (rec.active && S.paused) draw(performance.now(), true); }, 1000 / 30);
  S.dirty = true;
  status('Recording… play, pause and draw as usual; press Stop to save the video');
}

// Called by draw() right after the picture and the markings are drawn.
function captureRecordingFrame() {
  if (!rec.active || rec.mode !== 'frames' || !S.view) return;
  const now = performance.now();
  const frame = S.view.frame;
  let dt;
  if (rec.lastFrame === null) {
    dt = 0;
  } else if (!S.paused && frame !== rec.lastFrame) {
    if (rec.lastPaused) {
      dt = Math.min((now - rec.lastWall) / 1000, 0.1); // first frame after resuming
    } else {
      // Video time between the two frames (skipped frames keep their time; a seek counts as one frame).
      const step = frame > rec.lastFrame ? Math.min(frame - rec.lastFrame, S.videoFps) : 1;
      const speed = S.live ? 1 : (S.latest?.speed || 1);
      dt = step / (S.videoFps * speed);
    }
  } else if (S.paused && now - rec.lastWall >= 1000 / 30) {
    dt = Math.min((now - rec.lastWall) / 1000, 0.5); // frozen picture / live drawing: real time
  } else {
    return; // nothing new to record
  }
  rec.t += dt;
  const vf = new VideoFrame(view, { timestamp: Math.round(rec.t * 1e6) });
  rec.encoder.encode(vf, { keyFrame: rec.frames % 60 === 0 });
  vf.close();
  rec.frames += 1;
  rec.lastFrame = frame;
  rec.lastWall = now;
  rec.lastPaused = S.paused;
}

async function stopRecording() {
  if (!rec.active) return;
  rec.active = false;
  clearInterval(rec.timer);
  clearInterval(rec.pauseTimer);
  $('recBtn').classList.remove('on');
  $('recBtn').textContent = '● Rec';
  S.dirty = true;
  if (rec.mode === 'frames' && rec.frames === 0) {
    rec.encoder.close();
    rec.encoder = rec.muxer = null;
    status('Nothing was recorded', true);
  } else if (rec.mode === 'frames') {
    status('Finishing the video…');
    await rec.encoder.flush();
    rec.encoder.close();
    rec.muxer.finalize();
    saveBlob(new Blob([rec.muxer.target.buffer], { type: 'video/mp4' }), 'mp4', rec.t);
    rec.encoder = rec.muxer = null;
  } else {
    const r = rec.recorder;
    r.onstop = () => saveBlob(new Blob(rec.chunks, { type: r.mimeType || `video/${rec.ext}` }), rec.ext, (performance.now() - rec.t0) / 1000);
    r.stop();
    rec.recorder = null;
  }
}

function saveBlob(blob, ext, seconds) {
  const src = ($('videoList').value || 'video').replace(/\.[^.]+$/, '').replace(/[^\w\- ()]+/g, '_');
  const stamp = new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-');
  const name = `telestration_${src}_${stamp}.${ext}`;
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 60_000);
  status(`Saved ${name} (${(blob.size / 1e6).toFixed(1)} MB, ${seconds.toFixed(1)} s) — check your Downloads folder`);
}

// --- markings list --------------------------------------------------------------------
const STATE_TEXT = { tracking: '', pending: '…', lost: 'hidden', gone: 'off screen · R' };
const STATE_HELP = {
  lost: 'Player not confirmed (hidden behind someone?). The marking comes back by itself when he is found again.',
  gone: 'Player left the picture or the camera cut away. Select the marking, press R and click the player to follow him again.',
};

function renderList() {
  const ul = $('annList');
  ul.innerHTML = '';
  if (!S.anns.length) {
    ul.innerHTML = '<li class="empty">No markings yet. Pause, pick a tool, click a player.</li>';
    return;
  }
  for (const ann of S.anns) {
    const li = document.createElement('li');
    li.className = ann.id === S.sel ? 'sel' : '';
    const st = annState(ann);
    li.innerHTML = `<span class="dot" style="background:${ann.color}"></span>
      <span class="name"></span><span class="state ${st}" title="${STATE_HELP[st] || ''}">${STATE_TEXT[st] ?? st}</span>
      <button class="x" title="Delete">×</button>`;
    li.querySelector('.name').textContent = `${TOOL_NAMES[ann.type]}${ann.text ? ` "${ann.text}"` : ''}`;
    li.onclick = (e) => {
      if (e.target.classList.contains('x')) { removeAnn(ann.id); return; }
      S.sel = ann.id;
      renderList();
    };
    ul.appendChild(li);
  }
}
setInterval(renderList, 500);

function updateTransport(hdr) {
  $('playBtn').textContent = hdr.paused ? '▶' : '⏸';
  const sp = hdr.speed && hdr.speed !== 1 && !hdr.live ? ` · ${hdr.speed}×` : '';
  const following = Object.values(hdr.anchors || {}).filter((a) => a.state === 'tracking').length;
  const perf = `${S.fps} fps${sp} · following ${following} player${following === 1 ? '' : 's'}`;
  if (hdr.live) {
    $('clock').textContent = perf;
  } else {
    if (!S.seeking) $('seek').value = hdr.frame;
    const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
    $('clock').textContent = `${fmt(hdr.frame / hdr.fps)} / ${fmt(S.total / hdr.fps)} · ${perf}`;
  }
}

function status(text, err = false) {
  const el = $('status');
  el.textContent = text;
  el.title = text;
  el.classList.toggle('err', err);
}

function autoCalibrate(report) {
  if (!S.view) return;
  const hue = calibrateGrass(frameCtx(), S.view.w, S.view.h);
  S.needCalib = false;
  if (hue != null) {
    S.grass.hue = hue;
    S.maskCache.clear();
    if (report) status(`Grass hue set to ${Math.round(hue)}°`);
  } else if (report) status('Could not find a pitch colour in this frame', true);
}

// --- source controls -------------------------------------------------------------------
async function loadVideos(select) {
  const list = await (await fetch('/api/videos')).json();
  const sel = $('videoList');
  sel.innerHTML = list.length ? '' : '<option value="">(upload a video)</option>';
  for (const name of list) sel.add(new Option(name, name));
  if (select) sel.value = select;
}

$('srcKind').onchange = () => {
  const k = $('srcKind').value;
  $('srcFile').hidden = k !== 'file';
  $('srcWebcam').hidden = k !== 'webcam';
  $('srcUrl').hidden = k !== 'url';
};

$('upload').onchange = async () => {
  const f = $('upload').files[0];
  if (!f) return;
  status(`Uploading ${f.name}…`);
  const fd = new FormData();
  fd.append('file', f);
  const res = await (await fetch('/api/upload', { method: 'POST', body: fd })).json();
  if (res.error) { status(res.error, true); return; }
  await loadVideos(res.name);
  status(`Uploaded ${res.name}`);
};

$('openBtn').onclick = () => {
  const kind = $('srcKind').value;
  const value = kind === 'file' ? $('videoList').value : kind === 'webcam' ? $('webcamIdx').value : $('streamUrl').value.trim();
  if (!value && kind !== 'webcam') { status('Choose a video or enter a URL', true); return; }
  for (const a of [...S.anns]) removeAnn(a.id);
  S.view = null;
  S.smooth = {};
  status('Opening source…');
  send({ type: 'open', kind, value, simulate_live: $('simLive').checked });
};

$('playBtn').onclick = togglePause;
$('recBtn').onclick = () => { (rec.active ? stopRecording : startRecording)(); $('recBtn').blur(); };
$('speed').onchange = () => { send({ type: 'speed', value: +$('speed').value }); $('speed').blur(); };
$('stepBtn').onclick = () => send({ type: 'step' });
$('seek').addEventListener('input', () => { S.seeking = true; });
$('seek').addEventListener('change', () => { S.seeking = false; send({ type: 'seek', frame: +$('seek').value }); });
$('clearBtn').onclick = () => { for (const a of [...S.anns]) removeAnn(a.id); };
$('calibBtn').onclick = () => autoCalibrate(true);
$('optTol').oninput = () => { S.grass.tol = +$('optTol').value; $('tolVal').textContent = S.grass.tol; S.maskCache.clear(); };

document.querySelectorAll('#tools [data-tool]').forEach((b) => (b.onclick = () => setTool(b.dataset.tool)));
const sw = $('swatches');
for (const c of COLORS) {
  const b = document.createElement('button');
  b.style.background = c;
  b.title = c;
  b.onclick = () => {
    S.color = c;
    sw.querySelectorAll('button').forEach((x) => x.classList.toggle('active', x === b));
    const ann = S.anns.find((a) => a.id === S.sel); // recolour the selected marking too
    if (ann) ann.color = c;
    renderList();
  };
  sw.appendChild(b);
}
sw.firstChild.classList.add('active');

setTool('select');
renderList();
loadVideos();
connect();
requestAnimationFrame(render);
