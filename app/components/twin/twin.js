// Operations Twin: schematic axonometric of Chamchuri 5 drawn as SVG.
// Streamlit components v2 calls the default export on mount and again whenever `data` changes.
// Geometry is original line work from approximate zone footprints; nothing is to scale.

const NS = "http://www.w3.org/2000/svg";
const TH = (32 * Math.PI) / 180, PH = (24 * Math.PI) / 180;
const ST = Math.sin(TH), CT = Math.cos(TH), SP = Math.sin(PH), CP = Math.cos(PH);
const GAP = 36, SLAB = 3, LIFT = 8, ROOF = 9, FLOORS = 7;
const BUILDING_BOX = [64, 34, 420, 650], FOCUS_BOX = [36, 60, 724, 520];
const DONE = "Completed (demo)", DONE_LABEL = "Inspection marked complete (demo)";
const CALLOUT_X = 474, BUILDING = "__building__", MS = 620;

const el = (name, attrs = {}, parent) => {
  const n = document.createElementNS(NS, name);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(n);
  return n;
};
const raw = (x, y, z) => {
  const dx = x - 50, dy = y - 32;
  return [dx * CT - dy * ST, (dx * ST + dy * CT) * SP - z * CP];
};
const fit = (pts, box) => {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of pts) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
  const s = Math.min((box[2] - box[0]) / (x1 - x0), (box[3] - box[1]) / (y1 - y0));
  return { s, ox: (box[0] + box[2]) / 2 - (s * (x0 + x1)) / 2, oy: (box[1] + box[3]) / 2 - (s * (y0 + y1)) / 2 };
};
const inside = (poly, x, y) => {
  let c = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) c = !c;
  }
  return c;
};
const centroid = (poly) => {
  let a = 0, cx = 0, cy = 0;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const f = poly[j][0] * poly[i][1] - poly[i][0] * poly[j][1];
    a += f; cx += (poly[j][0] + poly[i][0]) * f; cy += (poly[j][1] + poly[i][1]) * f;
  }
  return [cx / (3 * a), cy / (3 * a)];
};
const frontEdges = (poly) => {
  const out = [];
  for (let i = 0; i < poly.length; i++) {
    const p = poly[i], q = poly[(i + 1) % poly.length];
    let nx = q[1] - p[1], ny = -(q[0] - p[0]);
    const len = Math.hypot(nx, ny) || 1; nx /= len; ny /= len;
    const mx = (p[0] + q[0]) / 2, my = (p[1] + q[1]) / 2;
    if (inside(poly, mx + nx * 0.2, my + ny * 0.2)) { nx = -nx; ny = -ny; }
    if (nx * ST + ny * CT > 1e-6) out.push([p, q]);
  }
  return out;
};
// east-west hatch segments inside a footprint (plan coordinates), for "no data" zones
const hatchSegments = (poly, step = 3.5) => {
  const ys = poly.map((p) => p[1]), out = [];
  for (let y = Math.min(...ys) + step / 2; y < Math.max(...ys); y += step) {
    const xs = [];
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [xi, yi] = poly[i], [xj, yj] = poly[j];
      if (yi > y !== yj > y) xs.push(xi + ((y - yi) * (xj - xi)) / (yj - yi));
    }
    xs.sort((a, b) => a - b);
    for (let k = 0; k + 1 < xs.length; k += 2) out.push([[xs[k], y], [xs[k + 1], y]]);
  }
  return out;
};
const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const lerp = (a, b, t) => a + (b - a) * t;
const kwh = (v) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(1) + " kWh";

function create(root) {
  const svg = root.querySelector("svg");
  const backBtn = root.querySelector(".twin-back");
  const titleEl = root.querySelector(".twin-title");
  const subEl = root.querySelector(".twin-sub");
  const S = { zones: {}, floors: {}, sel: null, view: null, anim: null, data: null, emit: () => {} };

  const corners = [[0, 0], [100, 0], [100, 64], [0, 64]];
  const topZ = (FLOORS - 1) * GAP + SLAB + ROOF;
  const buildingCam = fit(corners.flatMap(([x, y]) => [raw(x, y, 0), raw(x, y, topZ)]), BUILDING_BOX);
  const focusCam = (floor) => {
    const z = (floor - 1) * GAP;
    return fit(corners.flatMap(([x, y]) => [raw(x, y, z), raw(x, y, z + LIFT + SLAB)]), FOCUS_BOX);
  };
  const viewFor = (sel) => {
    if (!sel || !S.zones[sel]) {
      return { ...buildingCam, focus: 0, alpha: Array(FLOORS).fill(1), ext: {} };
    }
    const f = S.zones[sel].floor;
    return { ...focusCam(f), focus: 1, alpha: Array.from({ length: FLOORS }, (_, i) => (i + 1 === f ? 1 : 0)),
             ext: { [sel]: LIFT } };
  };

  // ---- static scaffold
  const gGround = el("g", { class: "ground" }, svg);
  const ground = el("polygon", {}, gGround);
  const posts = corners.map(() => el("line", { class: "post" }, gGround));
  const gFloors = el("g", {}, svg);
  const roof = el("polygon", { class: "roof" }, svg);
  const gFloorLabels = el("g", { class: "floor-labels" }, svg);
  const gCallouts = el("g", { class: "callouts" }, svg);
  const gLocator = el("g", { class: "locator" }, svg);
  const locBars = [];
  for (let f = 1; f <= FLOORS; f++) {
    locBars.push(el("rect", { x: 30, y: 652 - f * 11, width: 34, height: 7, rx: 1.5 }, gLocator));
  }
  const locText = el("text", { x: 72, y: 648, class: "loc-text" }, gLocator);

  function build(zones) {
    gFloors.textContent = ""; gFloorLabels.textContent = ""; S.zones = {}; S.floors = {};
    for (let f = 1; f <= FLOORS; f++) {
      const g = el("g", { class: "floor" }, gFloors);
      S.floors[f] = { g, sides: el("g", {}, g), tops: el("g", {}, g), lift: el("g", {}, g), labels: el("g", {}, g),
                      badges: el("g", {}, g), label: el("text", { class: "floor-label" }, gFloorLabels) };
    }
    const order = [...zones].sort((a, b) => {
      const ca = centroid(a.poly), cb = centroid(b.poly);
      return ca[0] * ST + ca[1] * CT - (cb[0] * ST + cb[1] * CT);
    });
    for (const z of order) {
      const F = S.floors[z.floor];
      const gs = el("g", {}, F.sides), gt = el("g", { class: "zone" }, F.tops);
      const zone = { ...z, c: z.label || centroid(z.poly), edges: frontEdges(z.poly), gs, gt,
                     sides: [], under: el("polygon", { class: "under" }, gt), top: el("polygon", { class: "top" }, gt) };
      zone.sides = zone.edges.map(() => el("polygon", { class: "side" }, gs));
      zone.hatch = el("path", { class: "hatch" }, gt);
      zone.hatchSegs = z.core ? [] : hatchSegments(z.poly);
      zone.title = el("title", {}, gt);
      if (!z.core) {
        zone.lab = el("g", { class: "zlabel" }, F.labels);
        zone.l1 = el("text", { class: "l1" }, zone.lab);
        zone.l2 = el("text", { class: "l2" }, zone.lab);
        zone.l3 = el("text", { class: "l3" }, zone.lab);
        zone.badge = el("g", { class: "badge" }, F.badges);
        zone.bc = el("circle", { r: 9.5 }, zone.badge);
        zone.bt = el("text", {}, zone.badge);
        const go = () => zone.clickable && pick(zone.id);
        gt.addEventListener("click", go);
        zone.badge.addEventListener("click", go);
        gt.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } });
      }
      S.zones[z.id] = zone;
    }
  }

  // ---- per-frame geometry
  function draw() {
    const v = S.view;
    const P = (x, y, z) => { const r = raw(x, y, z); return [v.ox + v.s * r[0], v.oy + v.s * r[1]]; };
    const pts = (a) => a.map((p) => p[0].toFixed(1) + "," + p[1].toFixed(1)).join(" ");
    ground.setAttribute("points", pts([[-5, -5], [105, -5], [105, 69], [-5, 69]].map(([x, y]) => P(x, y, 0))));
    corners.forEach(([x, y], i) => {
      const a = P(x, y, 0), b = P(x, y, topZ);
      posts[i].setAttribute("x1", a[0]); posts[i].setAttribute("y1", a[1]);
      posts[i].setAttribute("x2", b[0]); posts[i].setAttribute("y2", b[1]);
    });
    roof.setAttribute("points", pts(corners.map(([x, y]) => P(x, y, topZ))));
    const chrome = 1 - v.focus;
    gGround.style.opacity = chrome; roof.style.opacity = chrome;
    gFloorLabels.style.opacity = chrome; gCallouts.style.opacity = chrome;
    gCallouts.style.pointerEvents = chrome > 0.5 ? "auto" : "none";
    gLocator.style.opacity = v.focus;

    for (let f = 1; f <= FLOORS; f++) {
      const F = S.floors[f], a = v.alpha[f - 1], z0 = (f - 1) * GAP;
      F.g.style.opacity = a; F.g.style.pointerEvents = a > 0.5 ? "auto" : "none";
      F.labels.style.opacity = v.focus;
      const lp = P(0, 64, z0);
      F.label.setAttribute("x", lp[0] - 12); F.label.setAttribute("y", lp[1] + 4);
    }
    for (const id in S.zones) {
      const z = S.zones[id], z0 = (z.floor - 1) * GAP, h = SLAB + (v.ext[id] || 0);
      const top = pts(z.poly.map(([x, y]) => P(x, y, z0 + h)));
      z.top.setAttribute("points", top); z.under.setAttribute("points", top);
      z.edges.forEach(([p, q], i) => z.sides[i].setAttribute("points",
        pts([P(p[0], p[1], z0 + h), P(q[0], q[1], z0 + h), P(q[0], q[1], z0), P(p[0], p[1], z0)])));
      if (z.core) continue;
      z.hatch.setAttribute("d", z.status === "no_data" ? z.hatchSegs.map(([p, q]) => {
        const a = P(p[0], p[1], z0 + h), b = P(q[0], q[1], z0 + h);
        return `M${a[0].toFixed(1)},${a[1].toFixed(1)}L${b[0].toFixed(1)},${b[1].toFixed(1)}`;
      }).join("") : "");
      const c = P(z.c[0], z.c[1], z0 + h);
      z.anchor = c;
      z.badge.setAttribute("transform", `translate(${c[0].toFixed(1)},${(c[1] - 17 * v.focus).toFixed(1)})`);
      const ly = c[1] + (z.priority ? 8 : -4) * v.focus;
      [z.l1, z.l2, z.l3].forEach((t, i) => { t.setAttribute("x", c[0]); t.setAttribute("y", ly + i * 17); });
    }
    for (const c of S.callouts || []) {
      const a = S.zones[c.id].anchor;
      c.line.setAttribute("points", `${a[0].toFixed(1)},${a[1].toFixed(1)} ${CALLOUT_X - 26},${c.y} ${CALLOUT_X - 8},${c.y}`);
    }
  }

  function setTarget(sel, animate) {
    S.sel = sel && S.zones[sel] ? sel : null;
    const target = viewFor(S.sel);
    // extruded zones paint above their floor's other zones
    for (const id in S.zones) {
      const z = S.zones[id], F = S.floors[z.floor];
      const lifted = id === S.sel || (S.view && (S.view.ext[id] || 0) > 0.01);
      (lifted ? F.lift : F.sides).appendChild(z.gs);
      (lifted ? F.lift : F.tops).appendChild(z.gt);
      z.gt.classList.toggle("selected", id === S.sel);
    }
    root.classList.toggle("focused", !!S.sel);
    backBtn.hidden = !S.sel;
    header();
    const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (!animate || !S.view || reduce) { cancelAnimationFrame(S.anim); S.view = target; draw(); return; }
    const from = { ...S.view, alpha: [...S.view.alpha], ext: { ...S.view.ext } };
    const ids = new Set([...Object.keys(from.ext), ...Object.keys(target.ext)]);
    const t0 = performance.now();
    cancelAnimationFrame(S.anim);
    const step = (now) => {
      const t = Math.min(1, (now - t0) / MS), e = ease(t), ext = {};
      ids.forEach((id) => { ext[id] = lerp(from.ext[id] || 0, target.ext[id] || 0, e); });
      S.view = { s: lerp(from.s, target.s, e), ox: lerp(from.ox, target.ox, e), oy: lerp(from.oy, target.oy, e),
                 focus: lerp(from.focus, target.focus, e), ext,
                 alpha: from.alpha.map((a, i) => lerp(a, target.alpha[i], Math.min(1, e * 1.6))) };
      draw();
      if (t < 1) S.anim = requestAnimationFrame(step);
    };
    S.anim = requestAnimationFrame(step);
  }

  function pick(id) {
    if (id === S.sel) return;
    setTarget(id, true);
    S.emit(id || BUILDING);
  }
  backBtn.addEventListener("click", () => pick(null));

  function header() {
    const d = S.data; if (!d) return;
    if (S.sel) {
      const z = S.zones[S.sel];
      titleEl.textContent = `Floor ${z.floor} · Zone ${z.zone}`;
      subEl.textContent = `${d.date_label} · ${z.statusText}`;
    } else {
      titleEl.textContent = `Chamchuri 5 · ${d.date_label}`;
      subEl.textContent = d.summary;
    }
    locBars.forEach((b, i) => b.classList.toggle("on", S.sel && S.zones[S.sel].floor === i + 1));
    locText.textContent = S.sel ? `Floor ${S.zones[S.sel].floor} of ${FLOORS}` : "";
  }

  function update(data) {
    const first = !S.data;
    const sig = data.zones.map((z) => z.id).join("|");
    if (first || S.sig !== sig) { build(data.zones); S.sig = sig; }
    S.data = data;
    const nFind = data.zones.filter((z) => z.priority).length;
    const floorHasData = {};
    for (const zd of data.zones) {
      const z = S.zones[zd.id];
      Object.assign(z, { status: zd.status, priority: zd.priority || 0, wo: zd.wo || null, core: !!zd.core });
      const done = zd.wo === DONE;
      z.gt.setAttribute("class", "zone " + (zd.core ? "z-core" : "z-" + zd.status) + (done ? " wf-completed" : "") +
        (zd.id === S.sel ? " selected" : ""));
      if (zd.core) { z.title.textContent = "Service core (not metered as a zone)"; continue; }
      z.clickable = zd.status !== "no_ac";
      z.gt.classList.toggle("clickable", z.clickable);
      if (z.clickable) { z.gt.setAttribute("tabindex", "0"); z.gt.setAttribute("role", "button"); }
      else { z.gt.removeAttribute("tabindex"); z.gt.removeAttribute("role"); }
      if (zd.status !== "no_ac" && zd.status !== "no_data") floorHasData[zd.floor] = true;
      const name = `Floor ${zd.floor}, Zone ${zd.zone}`;
      if (z.priority) {
        const o = z.priority === 1 ? 0.5 : Math.max(0.16, 0.42 - (0.26 * (z.priority - 2)) / Math.max(1, nFind - 2));
        z.top.style.fillOpacity = o;
        z.sides.forEach((s) => (s.style.fillOpacity = Math.min(0.75, o + 0.18)));
        z.statusText = `review priority ${z.priority}: ${kwh(zd.departure_kwh)} ${zd.departure_kwh > 0 ? "above" : "below"} expected range`;
        z.l2.textContent = `${kwh(zd.departure_kwh)} ${zd.departure_kwh > 0 ? "above" : "below"} range`;
      } else {
        z.top.style.fillOpacity = ""; z.sides.forEach((s) => (s.style.fillOpacity = ""));
        z.statusText = { within: "within expected range", no_data: "no data on this date", no_ac: "no AC meter" }[zd.status];
        z.l2.textContent = z.statusText;
      }
      z.l1.textContent = `Zone ${zd.zone}`;
      z.l3.textContent = done ? DONE_LABEL : z.wo ? `Demo work order: ${z.wo}` : "";
      z.title.textContent = `${name}: ${z.statusText}` + (done ? ` · ${DONE_LABEL}` : z.wo ? ` · demo work order ${z.wo}` : "");
      z.gt.setAttribute("aria-label", z.title.textContent);
      z.badge.style.display = z.priority ? "" : "none";
      z.badge.setAttribute("class", "badge " + (done ? "b-done" : z.priority === 1 ? "b-top" : "b-finding") + (z.wo ? " has-wo" : ""));
      z.bt.textContent = done ? "✓" : z.priority || "";
    }
    for (let f = 1; f <= FLOORS; f++) {
      S.floors[f].label.textContent = `F${f}` + (floorHasData[f] ? "" : " · no data");
    }
    // callouts: one row per finding, ordered by screen height so leaders do not cross
    gCallouts.textContent = "";
    const cam = buildingCam;
    const rows = data.zones.filter((z) => z.priority).map((zd) => {
      const z = S.zones[zd.id], r = raw(z.c[0], z.c[1], (z.floor - 1) * GAP + SLAB);
      return { zd, ay: cam.oy + cam.s * r[1] };
    }).sort((a, b) => a.ay - b.ay);
    const step = rows.length > 14 ? 34 : 38;
    let y = Math.min(rows.length ? rows[0].ay - 30 : 0, 650 - rows.length * step) - step;
    y = Math.max(y, 30 - step);
    S.callouts = rows.map(({ zd, ay }, i) => {
      y = Math.min(Math.max(y + step, ay - 30), 650 - (rows.length - 1 - i) * step);
      const cDone = zd.wo === DONE;
      const g = el("g", { class: "callout " + (cDone ? "c-done" : zd.priority === 1 ? "c-top" : "c-finding"), tabindex: 0, role: "button" }, gCallouts);
      const line = el("polyline", {}, g);
      el("circle", { cx: CALLOUT_X + 4, cy: y, r: 9.5 }, g);
      el("text", { x: CALLOUT_X + 4, y: y + 3.6, class: "c-rank" }, g).textContent = cDone ? "✓" : zd.priority;
      el("text", { x: CALLOUT_X + 22, y: y - 1, class: "c-name" }, g).textContent =
        `Floor ${zd.floor} · Zone ${zd.zone}` + (cDone ? `  ·  ${kwh(zd.departure_kwh)}` : zd.priority === 1 ? "  ·  review first" : "");
      el("text", { x: CALLOUT_X + 22, y: y + 14, class: "c-val" }, g).textContent =
        cDone ? DONE_LABEL : `${kwh(zd.departure_kwh)} ${zd.departure_kwh > 0 ? "above" : "below"} range` +
          (zd.wo ? ` · WO ${zd.wo === "Created" ? "created" : "assigned"}` : "");
      g.addEventListener("click", () => pick(zd.id));
      g.addEventListener("keydown", (e) => { if (e.key === "Enter") pick(zd.id); });
      return { id: zd.id, line, y };
    });
    const want = data.selected && S.zones[data.selected] ? data.selected : null;
    if (first || want !== S.sel) setTarget(want, !first);
    else { header(); draw(); }
  }

  S.update = update;
  return S;
}

export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  const root = parentElement.querySelector(".twin");
  if (!root || !data) return;
  if (!root.__twin) root.__twin = create(root);
  root.__twin.emit = (id) => setTriggerValue("pick", id);
  root.__twin.update(data);
}
