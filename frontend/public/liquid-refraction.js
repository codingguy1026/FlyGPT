/* Generated refraction map, chromatic-neutral sRGB SVG optics. Chromium-only. */
(function opticalBoot() {
  "use strict";
  if (window.__drawingLiquidRefraction) return;
  window.__drawingLiquidRefraction = true;

  const SELECTORS = ".flightRail, .flightHeader, .telemetryDock, .launchComposer, .settingsModal, .connectionChip, .historyItem";
  const ua = navigator.userAgent;
  const chromium = /(?:Chrome|Chromium|Edg|OPR|SamsungBrowser)\//.test(ua) &&
    !/(?:CriOS|EdgiOS|FxiOS)\//.test(ua) && !/iP(?:hone|ad|od)/.test(ua);
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");
  const reduceTransparency = matchMedia("(prefers-reduced-transparency: reduce)");
  if (!chromium || reduceMotion.matches || reduceTransparency.matches || navigator.connection?.saveData) return;

  const SVG = "http://www.w3.org/2000/svg";
  const mobile = matchMedia("(max-width: 640px)").matches;
  const MAX_PANES = mobile ? 6 : 16;
  const MAP_SIZE = mobile ? 216 : 328;
  const REFRACTION = mobile ? 14 : 20;
  const attached = new Map();
  let counter = 0;
  let scheduled = false;
  const root = document.createElementNS(SVG, "svg");
  root.setAttribute("aria-hidden", "true");
  root.setAttribute("focusable", "false");
  root.setAttribute("width", "0");
  root.setAttribute("height", "0");
  root.style.cssText = "position:absolute;width:0;height:0;overflow:hidden;pointer-events:none";
  const defs = document.createElementNS(SVG, "defs");
  root.append(defs);
  document.body.append(root);

  function signedDistance(x, y, w, h, r) {
    const dx = Math.abs(x - w / 2) - (w / 2 - r);
    const dy = Math.abs(y - h / 2) - (h / 2 - r);
    return Math.hypot(Math.max(0, dx), Math.max(0, dy)) + Math.min(Math.max(dx, dy), 0) - r;
  }

  function mapImage(w, h, radius) {
    const ratio = Math.min(1, MAP_SIZE / Math.max(w, h));
    const cw = Math.max(1, Math.round(w * ratio));
    const ch = Math.max(1, Math.round(h * ratio));
    const canvas = document.createElement("canvas");
    canvas.width = cw;
    canvas.height = ch;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    const image = ctx.createImageData(cw, ch);
    const data = image.data;
    const rim = Math.min(21, Math.min(w, h) * .23);
    for (let row = 0; row < ch; row++) {
      for (let col = 0; col < cw; col++) {
        const x = (col + .5) * w / cw;
        const y = (row + .5) * h / ch;
        const d = -signedDistance(x, y, w, h, radius);
        const intensity = d > 0 && d < rim ? Math.pow(1 - d / rim, 1.7) : 0;
        const nx = signedDistance(x + 1, y, w, h, radius) - signedDistance(x - 1, y, w, h, radius);
        const ny = signedDistance(x, y + 1, w, h, radius) - signedDistance(x, y - 1, w, h, radius);
        const length = Math.hypot(nx, ny) || 1;
        const offset = (row * cw + col) * 4;
        const strength = intensity * 255 * .43;
        data[offset] = Math.round(Math.max(0, Math.min(255, 128 + nx / length * strength)));
        data[offset + 1] = Math.round(Math.max(0, Math.min(255, 128 + ny / length * strength)));
        data[offset + 2] = 128;
        data[offset + 3] = 255;
      }
    }
    ctx.putImageData(image, 0, 0);
    return canvas.toDataURL("image/png");
  }

  function refresh(el, item) {
    const rect = el.getBoundingClientRect();
    const w = Math.round(rect.width), h = Math.round(rect.height);
    if (w < 60 || h < 32 || w * h > 1200000 || (item.w === w && item.h === h)) return;
    const radius = Math.min(w / 2, h / 2, Math.max(0, parseFloat(getComputedStyle(el).borderTopLeftRadius) || 16));
    const map = mapImage(w, h, radius);
    if (!map) return;
    item.w = w;
    item.h = h;
    item.filter.setAttribute("x", "-16");
    item.filter.setAttribute("y", "-16");
    item.filter.setAttribute("width", String(w + 32));
    item.filter.setAttribute("height", String(h + 32));
    item.image.setAttribute("width", String(w));
    item.image.setAttribute("height", String(h));
    item.image.setAttribute("href", map);
  }

  const resizer = new ResizeObserver((entries) => {
    for (const entry of entries) {
      const item = attached.get(entry.target);
      if (item) refresh(entry.target, item);
    }
  });

  function attach(el) {
    if (attached.has(el)) return;
    const id = `dg-liquid-optic-${++counter}`;
    const filter = document.createElementNS(SVG, "filter");
    filter.setAttribute("id", id);
    filter.setAttribute("filterUnits", "userSpaceOnUse");
    filter.setAttribute("primitiveUnits", "userSpaceOnUse");
    filter.setAttribute("color-interpolation-filters", "sRGB");
    const image = document.createElementNS(SVG, "feImage");
    image.setAttribute("x", "0");
    image.setAttribute("y", "0");
    image.setAttribute("result", "lens-map");
    image.setAttribute("preserveAspectRatio", "none");
    const bend = document.createElementNS(SVG, "feDisplacementMap");
    bend.setAttribute("in", "SourceGraphic");
    bend.setAttribute("in2", "lens-map");
    bend.setAttribute("scale", String(REFRACTION));
    bend.setAttribute("xChannelSelector", "R");
    bend.setAttribute("yChannelSelector", "G");
    filter.append(image, bend);
    defs.append(filter);

    const needsPosition = getComputedStyle(el).position === "static";
    const previousPosition = el.style.position;
    if (needsPosition) el.style.position = "relative";
    const shine = document.createElement("span");
    shine.className = "dg-liquid-shine";
    shine.setAttribute("aria-hidden", "true");
    el.append(shine);

    const item = { filter, image, shine, w: 0, h: 0, frame: 0, needsPosition, previousPosition };
    const move = (ev) => {
      if (ev.pointerType === "touch") return;
      if (item.frame) cancelAnimationFrame(item.frame);
      item.frame = requestAnimationFrame(() => {
        const rect = el.getBoundingClientRect();
        if (rect.width && rect.height) {
          shine.style.setProperty("--dg-light-x", `${((ev.clientX - rect.left) / rect.width * 100).toFixed(1)}%`);
          shine.style.setProperty("--dg-light-y", `${((ev.clientY - rect.top) / rect.height * 100).toFixed(1)}%`);
        }
        item.frame = 0;
      });
    };
    item.move = move;
    el.addEventListener("pointermove", move, { passive: true });
    el.classList.add("dg-liquid-optical");
    el.style.setProperty("--dg-liquid-filter", `url("#${id}") blur(.7px) saturate(1.35)`);
    attached.set(el, item);
    refresh(el, item);
    resizer.observe(el);
  }

  function detach(el, item) {
    resizer.unobserve(el);
    el.removeEventListener("pointermove", item.move);
    if (item.frame) cancelAnimationFrame(item.frame);
    item.shine.remove();
    item.filter.remove();
    el.classList.remove("dg-liquid-optical");
    el.style.removeProperty("--dg-liquid-filter");
    if (item.needsPosition) el.style.position = item.previousPosition;
    attached.delete(el);
  }

  function scan() {
    scheduled = false;
    for (const [el, item] of attached) {
      if (!el.isConnected || reduceMotion.matches || reduceTransparency.matches) detach(el, item);
      else if (item.shine.parentElement !== el) el.append(item.shine);
    }
    if (document.visibilityState === "hidden" || reduceMotion.matches || reduceTransparency.matches) return;
    for (const selector of SELECTORS.split(",").map(x => x.trim())) {
      for (const el of document.querySelectorAll(selector)) {
        if (attached.size >= MAX_PANES) return;
        if (attached.has(el) || el.getAttribute("aria-hidden") === "true" || el.closest(".dg-liquid-optical")) continue;
        const r = el.getBoundingClientRect();
        if (r.width < 60 || r.height < 32 || r.width * r.height > 1200000) continue;
        attach(el);
      }
    }
  }
  function queueScan() {
    if (!scheduled) {
      scheduled = true;
      requestAnimationFrame(scan);
    }
  }
  const observer = new MutationObserver(records => {
    if (records.some(x => x.addedNodes.length || x.removedNodes.length)) queueScan();
  });
  observer.observe(document.body, { childList: true, subtree: true });
  document.addEventListener("visibilitychange", queueScan);
  reduceMotion.addEventListener?.("change", queueScan);
  reduceTransparency.addEventListener?.("change", queueScan);
  queueScan();
})();
