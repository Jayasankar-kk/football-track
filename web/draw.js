// Telestration drawing + "keying" (markings appear on the grass, under the players).
//
// Each marking draws into one or two layers:
//   ground – things painted on the pitch (rings, arrows, links, zones). This layer is
//            keyed: inside every player box, pixels that are not grass are cut out so
//            the player's body occludes the marking, like a broadcast chroma-key.
//   top    – things floating above everything (tags, spotlight beam).

export const TOOL_NAMES = {
  ring: 'Ring', spotlight: 'Spotlight', tag: 'Tag', arrow: 'Arrow', link: 'Link', zone: 'Zone',
};

export function rgba(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

export const feet = (b) => [(b[0] + b[2]) / 2, b[3]];
const height = (b) => b[3] - b[1];

function ringSize(b, u) {
  const rx = Math.max((b[2] - b[0]) * 0.75, height(b) * 0.32, 12 * u);
  return [rx, rx * 0.36];
}

function ellipse(ctx, cx, cy, rx, ry) {
  ctx.beginPath();
  ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
}

function arrowHead(ctx, x1, y1, x2, y2, size) {
  const a = Math.atan2(y2 - y1, x2 - x1);
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - size * Math.cos(a - 0.45), y2 - size * Math.sin(a - 0.45));
  ctx.lineTo(x2 - size * Math.cos(a + 0.45), y2 - size * Math.sin(a + 0.45));
  ctx.closePath();
  ctx.fill();
}

function glow(ctx, color, u) {
  ctx.shadowColor = color;
  ctx.shadowBlur = 10 * u;
}

export function arrowEnd(ann, b) {
  const [fx, fy] = feet(b);
  if (ann.offsetPx) return [fx + ann.offsetPx[0], fy + ann.offsetPx[1]];
  const h = height(b);
  return [fx + ann.offset[0] * h, fy + ann.offset[1] * h];
}

// --- ground layer -----------------------------------------------------------

function groundRing(ctx, b, color, u, t, spot) {
  const [cx, cy] = feet(b);
  const [rx, ry] = ringSize(b, u);
  ctx.save();
  if (spot) {
    const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, rx * 1.3);
    g.addColorStop(0, rgba(color, 0.65));
    g.addColorStop(1, rgba(color, 0));
    ctx.fillStyle = g;
    ctx.translate(cx, cy);
    ctx.scale(1, ry / rx);
    ctx.beginPath();
    ctx.arc(0, 0, rx * 1.3, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    return;
  }
  ctx.fillStyle = rgba(color, 0.22);
  ellipse(ctx, cx, cy, rx, ry);
  ctx.fill();
  glow(ctx, color, u);
  ctx.strokeStyle = color;
  ctx.lineWidth = 3.5 * u;
  ctx.stroke();
  ctx.shadowBlur = 0;
  ctx.setLineDash([10 * u, 7 * u]);
  ctx.lineDashOffset = -t * 25 * u;
  ctx.lineWidth = 2 * u;
  ctx.globalAlpha *= 0.85;
  ellipse(ctx, cx, cy, rx * 1.28, ry * 1.28);
  ctx.stroke();
  ctx.restore();
}

function groundArrow(ctx, ann, b, color, u) {
  const [fx, fy] = feet(b);
  const [ex, ey] = arrowEnd(ann, b);
  ctx.save();
  glow(ctx, color, u);
  ctx.strokeStyle = ctx.fillStyle = color;
  ctx.lineWidth = 4.5 * u;
  ctx.lineCap = 'round';
  if (ann.dashed) ctx.setLineDash([14 * u, 10 * u]);
  const len = Math.hypot(ex - fx, ey - fy) || 1;
  const head = Math.min(22 * u, len * 0.5);
  const sx = ex - ((ex - fx) / len) * head * 0.6, sy = ey - ((ey - fy) / len) * head * 0.6;
  ctx.beginPath();
  ctx.moveTo(fx, fy);
  ctx.lineTo(sx, sy);
  ctx.stroke();
  ctx.setLineDash([]);
  arrowHead(ctx, fx, fy, ex, ey, head);
  ellipse(ctx, fx, fy, 7 * u, 3 * u);
  ctx.fill();
  ctx.restore();
}

function groundChain(ctx, pts, color, u, closed, t) {
  if (pts.length < 2) return;
  ctx.save();
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  if (closed) {
    ctx.closePath();
    ctx.fillStyle = rgba(color, 0.25);
    ctx.fill();
  }
  glow(ctx, color, u);
  ctx.strokeStyle = color;
  ctx.lineWidth = 3.5 * u;
  ctx.lineJoin = 'round';
  if (!closed) {
    ctx.setLineDash([16 * u, 8 * u]);
    ctx.lineDashOffset = -t * 30 * u;
  }
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = color;
  for (const [x, y] of pts) {
    ellipse(ctx, x, y, 9 * u, 3.5 * u);
    ctx.fill();
  }
  ctx.restore();
}

// --- top layer ----------------------------------------------------------------

function topSpotlight(ctx, b, color, u) {
  const [cx, cy] = feet(b);
  const [rx] = ringSize(b, u);
  const top = Math.max(0, b[1] - height(b) * 2.2);
  const tw = rx * 0.25;
  ctx.save();
  ctx.globalCompositeOperation = 'screen';
  const g = ctx.createLinearGradient(0, top, 0, cy);
  g.addColorStop(0, rgba(color, 0.0));
  g.addColorStop(0.35, rgba(color, 0.12));
  g.addColorStop(1, rgba(color, 0.38));
  ctx.fillStyle = g;
  ctx.beginPath();
  ctx.moveTo(cx - tw, top);
  ctx.lineTo(cx + tw, top);
  ctx.lineTo(cx + rx * 1.1, cy);
  ctx.lineTo(cx - rx * 1.1, cy);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function topTag(ctx, b, text, color, u) {
  const cx = (b[0] + b[2]) / 2;
  const fs = Math.round(15 * u);
  ctx.save();
  ctx.font = `700 ${fs}px system-ui, "Segoe UI", sans-serif`;
  const tw = ctx.measureText(text).width;
  const pw = 8 * u, ph = 5 * u, gap = 12 * u;
  const w = tw + pw * 2, h = fs + ph * 2;
  const x = cx - w / 2, y = Math.max(2, b[1] - gap - h);
  ctx.fillStyle = color;
  ctx.shadowColor = 'rgba(0,0,0,.5)';
  ctx.shadowBlur = 6 * u;
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, 4 * u);
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(cx - 6 * u, y + h);
  ctx.lineTo(cx + 6 * u, y + h);
  ctx.lineTo(cx, y + h + 7 * u);
  ctx.fill();
  ctx.shadowBlur = 0;
  ctx.fillStyle = luminance(color) > 0.55 ? '#111' : '#fff';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, x + pw, y + h / 2 + 1);
  ctx.restore();
}

function luminance(hex) {
  const n = parseInt(hex.slice(1), 16);
  return (0.299 * ((n >> 16) & 255) + 0.587 * ((n >> 8) & 255) + 0.114 * (n & 255)) / 255;
}

// --- public -----------------------------------------------------------------------

/**
 * Draw one marking's layer.
 * @param resolved array (one per anchor) of {box, state} or null if unknown.
 */
export function drawAnnotation(ctx, layer, ann, resolved, u, t, showLost = false) {
  // A marking is only drawn on players that are currently followed (unless showLost).
  const live = resolved.filter((r) => r && (r.state === 'tracking' || (showLost && r.state === 'lost')));
  if (!live.length) return;
  const lost = live.some((r) => r.state === 'lost');
  ctx.save();
  ctx.globalAlpha = lost ? 0.45 : 1;
  const b = live[0].box;
  const c = ann.color;
  if (layer === 'ground') {
    if (ann.type === 'ring') groundRing(ctx, b, c, u, t, false);
    else if (ann.type === 'spotlight') groundRing(ctx, b, c, u, t, true);
    else if (ann.type === 'arrow') groundArrow(ctx, ann, b, c, u);
    else if (ann.type === 'link' || ann.type === 'zone') {
      groundChain(ctx, live.map((r) => feet(r.box)), c, u, ann.type === 'zone' && live.length > 2, t);
    }
  } else {
    if (ann.type === 'spotlight') topSpotlight(ctx, b, c, u);
    else if (ann.type === 'tag') topTag(ctx, b, ann.text, c, u);
  }
  ctx.restore();
}

/** Preview while building a link / zone / arrow. */
export function drawPending(ctx, pending, boxes, mouse, color, u, t) {
  if (!pending) return;
  ctx.save();
  ctx.globalAlpha = 0.8;
  if (pending.type === 'arrow') {
    const b = boxes[0];
    if (b && pending.cur) {
      const ann = pending.target && pending.target.tid == null
        ? { offsetPx: [pending.cur[0] - pending.start[0], pending.cur[1] - pending.start[1]] }
        : { offset: [(pending.cur[0] - feet(b)[0]) / height(b), (pending.cur[1] - feet(b)[1]) / height(b)] };
      groundArrow(ctx, ann, b, color, u);
    }
  } else {
    const pts = boxes.filter(Boolean).map(feet);
    if (mouse) pts.push(mouse);
    groundChain(ctx, pts, color, u, pending.type === 'zone' && pts.length > 2, t);
  }
  ctx.restore();
}

function grassTester({ hue, tol }) {
  const lo = hue - tol, hi = hue + tol;
  return (r, g, b) => {
    const max = r > g ? (r > b ? r : b) : g > b ? g : b;
    const min = r < g ? (r < b ? r : b) : g < b ? g : b;
    if (max < 28) return false;
    const d = max - min;
    if (d < max * 0.1) return false; // grey/white (lines, socks, shirts)
    let h;
    if (max === r) h = ((g - b) / d) % 6;
    else if (max === g) h = (b - r) / d + 2;
    else h = (r - g) / d + 4;
    h *= 60;
    if (h < 0) h += 360;
    return h >= lo && h <= hi;
  };
}

/**
 * Cut the ground layer wherever a player's (non-grass) pixels are.
 * `cache` maps track id -> mask for the current frame (cleared when the frame changes).
 */
export function keyLayer(gctx, fctx, tracks, grass, W, H, cache) {
  let isGrass = null;
  for (const tr of tracks) {
    if (tr.cls === 'sports ball' || tr.cls === 'ball') continue;
    const x1 = Math.max(0, Math.floor(tr.box[0]) - 2), y1 = Math.max(0, Math.floor(tr.box[1]) - 2);
    const x2 = Math.min(W, Math.ceil(tr.box[2]) + 2), y2 = Math.min(H, Math.ceil(tr.box[3]) + 3);
    const w = x2 - x1, h = y2 - y1;
    if (w <= 0 || h <= 0) continue;
    const gd = gctx.getImageData(x1, y1, w, h);
    const gp = gd.data;
    let any = false;
    for (let i = 3; i < gp.length; i += 4) if (gp[i]) { any = true; break; }
    if (!any) continue;

    const key = `${tr.id}:${x1},${y1},${w},${h}`;
    let mask = cache.get(key);
    if (!mask) {
      isGrass ||= grassTester(grass);
      const fp = fctx.getImageData(x1, y1, w, h).data;
      const raw = new Uint8Array(w * h);
      for (let p = 0, i = 0; p < raw.length; p++, i += 4) raw[p] = isGrass(fp[i], fp[i + 1], fp[i + 2]) ? 0 : 1;
      // 3x3 majority filter: removes speckle from grass texture / compression noise.
      mask = new Uint8Array(w * h);
      for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
          const p = y * w + x;
          const s = raw[p - w - 1] + raw[p - w] + raw[p - w + 1] + raw[p - 1] + raw[p] + raw[p + 1] +
            raw[p + w - 1] + raw[p + w] + raw[p + w + 1];
          mask[p] = s >= 5 ? 1 : 0;
        }
      }
      cache.set(key, mask);
    }
    for (let p = 0, i = 3; p < mask.length; p++, i += 4) if (mask[p]) gp[i] = 0;
    gctx.putImageData(gd, x1, y1);
  }
}

/** Find the dominant pitch hue in the frame (degrees). */
export function calibrateGrass(fctx, W, H) {
  const data = fctx.getImageData(0, 0, W, H).data;
  const bins = new Float32Array(72);
  for (let i = 0; i < data.length; i += 4 * 7) {
    const r = data[i], g = data[i + 1], b = data[i + 2];
    const max = Math.max(r, g, b), min = Math.min(r, g, b), d = max - min;
    if (max < 40 || d < max * 0.15) continue;
    let h;
    if (max === r) h = ((g - b) / d) % 6;
    else if (max === g) h = (b - r) / d + 2;
    else h = (r - g) / d + 4;
    h *= 60;
    if (h < 0) h += 360;
    bins[Math.floor(h / 5) % 72]++;
  }
  let best = -1, bestV = 0;
  for (let k = 8; k < 36; k++) { // 40°..180°: plausible grass hues
    const v = bins[k - 1] + bins[k] + bins[k + 1];
    if (v > bestV) { bestV = v; best = k; }
  }
  return best < 0 ? null : best * 5 + 2.5;
}

// --- hit testing -------------------------------------------------------------------

function distToSeg(p, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const l = dx * dx + dy * dy || 1;
  const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l));
  return Math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy);
}

export function hitAnnotation(ann, resolved, p, u) {
  const live = resolved.filter((r) => r && r.state !== 'gone');
  const pad = 10 * u;
  for (const r of live) {
    const b = r.box;
    if (p[0] >= b[0] - pad && p[0] <= b[2] + pad && p[1] >= b[1] - pad - (ann.type === 'tag' ? 40 * u : 0) && p[1] <= b[3] + pad) return true;
  }
  if (ann.type === 'arrow' && live[0]) {
    if (distToSeg(p, feet(live[0].box), arrowEnd(ann, live[0].box)) < 10 * u) return true;
  }
  if (ann.type === 'link' || ann.type === 'zone') {
    const pts = live.map((r) => feet(r.box));
    if (ann.type === 'zone' && pts.length > 2) pts.push(pts[0]);
    for (let i = 1; i < pts.length; i++) if (distToSeg(p, pts[i - 1], pts[i]) < 10 * u) return true;
  }
  return false;
}
