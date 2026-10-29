/* Infinite-OCR web app. No libraries; talks only to the local server (web/server.py).
   Parts: helpers, page wipe, hero demo, status pill, effort slider, choosing a file, running a job, result. */

"use strict";

const $ = (id) => document.getElementById(id);
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ---- helpers --------------------------------------------------------------- */

async function api(path, options) {
  let res;
  try {
    res = await fetch(path, options);
  } catch (e) {
    throw new Error("The local server is not reachable. Start it with ./start-web.sh.");
  }
  const type = res.headers.get("content-type") || "";
  if (!type.includes("application/json")) {
    // a plain file server (python -m http.server) answers /api/... with an HTML error page or nothing
    throw new Error("This page is being served without the OCR server behind it. Stop it and run ./start-web.sh.");
  }
  const body = await res.json();
  if (!res.ok) throw new Error(body.detail || res.statusText);
  return body;
}

/* "12 s" for a short job, "3.4 min" for a longer one */
function duration(seconds) {
  return seconds < 90 ? `${seconds} s` : `${(seconds / 60).toFixed(1)} min`;
}

function clock(seconds) {
  const m = Math.floor(seconds / 60);
  return `${m}:${String(seconds % 60).padStart(2, "0")}`;
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/* ---- page wipe: the original on the left, the digitised text on the right ---- */

const MARGIN_LABELS = new Set(["Header", "Footer", "Page-number", "Folio"]);
const HEADING_LABELS = new Set(["Title", "Chapter-title", "Section-title", "Sub-section-title", "Sub-sub-section-title"]);

/* Largest font size at which the text still fits its box. */
function fitText(node, cap) {
  let lo = 5;
  let hi = Math.max(lo, Math.min(Math.floor(node.clientHeight), cap));
  let best = lo;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    node.style.fontSize = mid + "px";
    if (node.scrollHeight <= node.clientHeight + 1 && node.scrollWidth <= node.clientWidth + 1) {
      best = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  node.style.fontSize = best + "px";
}

function groupOf(b) {
  if (b.kind) return b.kind === "heading" ? "heading" : b.kind === "furniture" ? "margin" : "text";
  return MARGIN_LABELS.has(b.label) ? "margin" : HEADING_LABELS.has(b.label) ? "heading" : "text";
}

/* OCR engines write superscripts as LaTeX ($^{*1}$). Show those as real sup/sub and drop the dollar signs;
   anything else between dollars is shown as written. Text nodes only: nothing is ever interpreted as HTML. */
const MATH = /\${1,2}([^$]+)\${1,2}/g;

function richText(node, text) {
  let last = 0;
  for (const m of text.matchAll(MATH)) {
    node.append(text.slice(last, m.index));
    const inner = m[1].trim();
    const sup = inner.match(/^\^\{?([^{}]*)\}?$/);
    const sub = inner.match(/^_\{?([^{}]*)\}?$/);
    node.append(sup ? el("sup", "", sup[1]) : sub ? el("sub", "", sub[1]) : inner);
    last = m.index + m[0].length;
  }
  node.append(text.slice(last));
}

/* A figure has no text to show, so it is cut out of the page image: the page is drawn behind a box
   that is the figure's size, shifted so only the figure's region shows. */
/* A "figure" with text inside it is a page the layout model took for a picture (a scanned certificate came back as one
   "Diagram" over 66% of the page): the text is the content, so copying the picture would draw the page under it. */
function coversText(b, textBlocks) {
  const [x, y, w, h] = b.box;
  let inside = 0;
  for (const t of textBlocks) {
    const [tx, ty, tw, th] = t.box;
    inside += Math.max(0, Math.min(x + w, tx + tw) - Math.max(x, tx)) * Math.max(0, Math.min(y + h, ty + th) - Math.max(y, ty));
  }
  return inside > 0.15 * w * h;
}

function figureNode(b, src, place) {
  const [x, y, w, h] = b.box;
  const node = el("div", "twin-figure");
  node.style.backgroundImage = `url("${src}")`;
  node.style.backgroundSize = `${100 / w}% ${100 / h}%`;
  node.style.backgroundPosition = `${w < 1 ? (x / (1 - w)) * 100 : 0}% ${h < 1 ? (y / (1 - h)) * 100 : 0}%`;
  node.dataset.label = b.label;
  node.style.cssText += place;
  return node;
}

/* The text of a block, or a table built cell by cell. */
function twinNode(b, place) {
  let node;
  if (b.table) {
    node = el("div", "twin-block twin-table");
    node.style.gridTemplateColumns = `repeat(${b.table.cols}, minmax(0, 1fr))`;
    for (const c of b.table.cells) {
      const cell = el("div", "twin-cell");
      richText(cell, c.t);
      if (c.h) cell.dataset.head = "";
      cell.style.gridRow = `${c.r + 1} / span ${c.rs}`;
      cell.style.gridColumn = `${c.c + 1} / span ${c.cs}`;
      node.appendChild(cell);
    }
  } else {
    node = el("div", "twin-block");
    richText(node, b.text);
  }
  node.dataset.label = b.label;
  if (b.kind) node.dataset.kind = b.kind;
  node.style.cssText += place;
  return node;
}

class PageWipe {
  constructor({ wrap, stage, twin, boxes, img, handle, toggle }) {
    Object.assign(this, { wrap, stage, twin, boxes, img, handle, toggle });
    this.position = 100; // percent of the page shown as the original
    this.dragging = false;

    stage.addEventListener("pointerdown", (e) => {
      this.dragging = true;
      stage.setPointerCapture(e.pointerId);
      this.setPosition(this.fromEvent(e));
    });
    stage.addEventListener("pointermove", (e) => { if (this.dragging) this.setPosition(this.fromEvent(e)); });
    const end = () => { this.dragging = false; };
    stage.addEventListener("pointerup", end);
    stage.addEventListener("pointercancel", end);

    handle.addEventListener("keydown", (e) => {
      const moves = { ArrowLeft: -3, ArrowDown: -3, ArrowRight: 3, ArrowUp: 3 };
      if (e.key in moves) this.setPosition(this.position + moves[e.key]);
      else if (e.key === "Home") this.setPosition(0);
      else if (e.key === "End") this.setPosition(100);
      else return;
      e.preventDefault();
    });

    toggle.addEventListener("change", () => this.setBoxes(toggle.checked));
    new ResizeObserver(() => this.refit()).observe(stage);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => this.refit());
  }

  fromEvent(e) {
    const r = this.stage.getBoundingClientRect();
    return ((e.clientX - r.left) / r.width) * 100;
  }

  setPosition(p) {
    this.position = Math.max(0, Math.min(100, p));
    this.stage.style.setProperty("--p", this.position + "%");
    this.handle.setAttribute("aria-valuenow", String(Math.round(this.position)));
  }

  setBoxes(on) {
    this.stage.dataset.boxes = on ? "on" : "off";
    this.toggle.checked = on;
  }

  refit() {
    const cap = this.stage.clientWidth * 0.05;
    this.twin.querySelectorAll(".twin-block").forEach((n) => fitText(n, cap));
  }

  /* Show a page: `data.blocks` with boxes as [x, y, w, h] in 0-1, and the image at `src`. */
  load(data, src) {
    return new Promise((resolve, reject) => {
      this.img.onload = () => {
        const nw = this.img.naturalWidth;
        const nh = this.img.naturalHeight;
        this.stage.style.setProperty("--ar", `${nw} / ${nh}`);
        this.wrap.style.setProperty("--ar-num", String(nw / nh));
        this.twin.replaceChildren();
        this.boxes.replaceChildren();
        const textBlocks = data.blocks.filter((t) => t.shown);
        let copied = 0; // figures drawn from the page
        for (const b of data.blocks) {
          const [x, y, w, h] = b.box.map((v) => v * 100);
          const place = `left:${x}%;top:${y}%;width:${w}%;height:${h}%;`;
          const figure = b.kind === "figure" && !coversText(b, textBlocks);
          if (figure) { this.twin.appendChild(figureNode(b, src, place)); copied++; }
          else if (b.shown) this.twin.appendChild(twinNode(b, place));
          const o = el("div", "box");
          o.dataset.label = b.label;
          o.dataset.group = groupOf(b);
          if (x > 60) o.dataset.edge = "right";        // keep the label inside the page
          if (!b.shown && !figure) o.dataset.quiet = ""; // not read (empty header or footer): outline only
          o.style.cssText = place;
          this.boxes.appendChild(o);
        }
        this.refit();
        resolve(copied);
      };
      this.img.onerror = () => reject(new Error("The page image could not be loaded."));
      this.img.src = src;
    });
  }

  /* Open with the line sweeping in from the right, once. */
  sweepIn(to = 52) {
    if (reducedMotion) {
      this.setPosition(to);
      return;
    }
    const t0 = performance.now();
    const step = (now) => {
      const k = Math.min(1, (now - t0) / 1300);
      this.setPosition(100 + (to - 100) * (1 - Math.pow(1 - k, 3)));
      if (k < 1 && !this.dragging) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }
}

const parts = (prefix) => ({
  wrap: $(prefix + "wrap"), stage: $(prefix + "stage"), twin: $(prefix + "twin"), boxes: $(prefix + "boxes"),
  img: $(prefix + "img"), handle: $(prefix + "handle"), toggle: $(prefix + "boxes-toggle"),
});

/* ---- hero demo: a real page from an earlier run ----------------------------- */

const hero = new PageWipe({
  wrap: $("stage-wrap"), stage: $("stage"), twin: $("twin"), boxes: $("boxes"),
  img: $("scan-img"), handle: $("handle"), toggle: $("boxes-toggle"),
});
hero.setBoxes(new URLSearchParams(location.search).get("boxes") === "on");

fetch("demo/page.json")
  .then((r) => r.json())
  .then((data) => {
    $("demo-title").textContent = data.source;
    $("demo-sub").textContent = `${data.language}, read locally, ₹0`;
    return hero.load(data, "demo/page.jpg").then(() => hero.sweepIn());
  })
  .catch(() => { $("demo-sub").textContent = "The sample page could not be loaded."; });

/* ---- status pill: is the model, and so the GPU, in use? ---------------------- */

let serverUp = false;

function renderStatus(s) {
  const gb = s.gpu_used_mb == null ? null : (s.gpu_used_mb / 1024).toFixed(1);
  const texts = {
    off: gb !== null && s.gpu_used_mb > 1500 ? `Model off, GPU ${gb} GB in use` : "Model off, GPU idle",
    starting: "Starting the model on the GPU",
    ready: gb !== null ? `Model running, GPU ${gb} GB` : "Model running",
    sleeping: gb !== null ? `Model asleep, GPU ${gb} GB` : "Model asleep, GPU free",
    stopping: "Releasing the GPU",
  };
  $("gpu").dataset.state = s.model.state;
  $("gpu-text").textContent = texts[s.model.state] || s.model.state;

}

async function refreshStatus() {
  try {
    renderStatus(await api("/api/status"));
    serverUp = true;
  } catch (e) {
    serverUp = false;
    $("gpu").dataset.state = "offline";
    $("gpu-text").textContent = "Server not connected";
  }
  updateStart();
}

setInterval(() => { if (document.visibilityState === "visible") refreshStatus(); }, 3000);

/* ---- effort slider ---------------------------------------------------------- */

const MODE_KEYS = ["fast", "normal", "think"]; // the three stops, in the order of the modes in src/modes.py

const MODES = [
  { name: "Fast", what: "Bodhan and Sarvam readers go through the document and a free checker using raw PDF analysis compares every table. No verifier model is asked." },
  { name: "Medium", what: "Fast, plus an image check using two consensus verifier models on the table cells still in doubt, and a second look at a few cells both readers agree on but that look risky." },
  { name: "High", what: "Both Sarvam and Bodhan scan every page. A third model is also included in the consensus decision on scanned tables, and more of the risky agreements looked at again." },
];

const effort = $("effort");
let fileRecord = null; // the checked upload: id, name, pages

function showEffort(i) {
  const m = MODES[i];
  effort.value = String(i);
  effort.style.setProperty("--fill", (i / (MODES.length - 1)) * 100 + "%");
  document.querySelectorAll(".stop").forEach((b) => b.setAttribute("aria-pressed", String(Number(b.dataset.stop) === i)));
  $("readout-what").textContent = m.what;
}

effort.addEventListener("input", () => showEffort(Number(effort.value)));
document.querySelectorAll(".stop").forEach((b) => b.addEventListener("click", () => showEffort(Number(b.dataset.stop))));
showEffort(Number(effort.value));

/* ---- choosing a file -------------------------------------------------------- */

const drop = $("drop");
const fileInput = $("file");
let job = null; // the running job, if any

function showFileLabel(text) {
  const label = $("drop-file");
  label.textContent = text || "";
  label.hidden = !text;
}

function showDropError(text) {
  const label = $("drop-error");
  label.textContent = text || "";
  label.hidden = !text;
}

async function chooseFile(file) {
  if (!file || job) return;
  showDropError("");
  fileRecord = null;
  const mb = file.size / 1048576;
  showFileLabel(`${file.name} (${mb < 0.1 ? "under 0.1" : mb.toFixed(1)} MB), checking`);
  updateStart();
  try {
    const form = new FormData();
    form.append("file", file);
    fileRecord = await api("/api/files", { method: "POST", body: form });
    showFileLabel(`${fileRecord.name}, ${fileRecord.pages} ${fileRecord.pages === 1 ? "page" : "pages"}`);
  } catch (e) {
    showFileLabel("");
    showDropError(e.message);
  }
  showEffort(Number(effort.value));
  updateStart();
}

fileInput.addEventListener("change", () => chooseFile(fileInput.files[0]));
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.dataset.over = "true"; }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, () => { drop.dataset.over = "false"; }));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  if (e.dataTransfer.files.length) chooseFile(e.dataTransfer.files[0]);
});

/* ---- start, progress, cancel ------------------------------------------------- */

const TERMINAL = new Set(["done", "failed", "cancelled"]);
let pollTimer = null;
let lostContact = 0;

function updateStart() {
  const busy = job !== null;
  $("start").disabled = busy || !fileRecord || !serverUp;
  $("start-note").textContent = !serverUp ? "The local server is not connected."
    : busy ? "Reading in progress."
    : fileRecord ? "Reading uses the GPU while it runs, and frees it when it is done."
    : "Choose a file first.";
  fileInput.disabled = busy;
}

$("start").addEventListener("click", async () => {
  const mode = MODE_KEYS[Number(effort.value)];
  try {
    job = await api("/api/jobs", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ file_id: fileRecord.id, mode }),
    });
  } catch (e) {
    $("start-note").textContent = e.message;
    return;
  }
  lostContact = 0;
  $("results").hidden = true;
  $("progress").hidden = false;
  $("progress-error").hidden = true;
  renderJob(job);
  updateStart();
  $("progress").scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth", block: "center" });
  poll();
});

$("cancel").addEventListener("click", async () => {
  $("cancel").disabled = true;
  try { await api(`/api/jobs/${job.id}/cancel`, { method: "POST" }); } catch (e) { /* the next poll shows the state */ }
});

const STAGES = {
  queued: ["Getting ready", "Checking the file."],
  starting_model: ["Starting the model on the GPU", "The GPU was idle until you pressed start. This takes about a minute."],
  waking_model: ["Waking the model", "It was kept in memory since the last document, so this takes a moment."],
  reading: ["Reading pages", ""],
  checking: ["Checking tables", "Comparing the two readers cell by cell."],
  done: ["Finished", ""],
  failed: ["It stopped with an error", ""],
  cancelled: ["Cancelled", "The model was stopped and the GPU is free."],
};

function renderJob(j) {
  const [stage, detail] = STAGES[j.state === "starting_model" && j.model_action === "wake" ? "waking_model" : j.state];
  $("progress-stage").textContent = stage;
  let sub = detail;
  if (j.state === "reading") sub = `${j.pages_done} of ${j.pages_total} pages${j.eta && j.eta !== "-" ? `, about ${j.eta} left` : ""}.`;
  if (j.state === "done") sub = `Read ${j.result.pages_done} of ${j.result.pages_total} pages. The GPU is free again.`;
  $("progress-detail").textContent = sub;

  const bar = $("progress-bar");
  const waiting = j.state === "starting_model" || j.state === "queued";
  const pct = j.state === "done" || j.state === "checking" ? 100 : j.pages_total ? (j.pages_done / j.pages_total) * 100 : 0;
  if (waiting) bar.dataset.wait = ""; else delete bar.dataset.wait;
  $("progress-fill").style.width = waiting ? "" : pct + "%";
  bar.setAttribute("aria-valuenow", String(Math.round(waiting ? 0 : pct)));

  $("p-pages").textContent = `${j.pages_done} / ${j.pages_total}`;
  $("p-speed").textContent = j.pages_per_min ? `${j.pages_per_min.toFixed(1)} per min` : "–";
  $("p-cost").textContent = `₹${j.cost_inr.toFixed(2)}`;
  $("p-time").textContent = clock(j.elapsed_s);
  $("progress-log").textContent = j.log.join("\n");
  const err = $("progress-error");
  err.hidden = !j.error;
  err.textContent = j.error;
  $("cancel").hidden = TERMINAL.has(j.state);
  $("cancel").disabled = false;
}

async function poll() {
  clearTimeout(pollTimer);
  try {
    const j = await api(`/api/jobs/${job.id}`);
    lostContact = 0;
    renderJob(j);
    if (TERMINAL.has(j.state)) {
      job = null;
      updateStart();
      refreshStatus();
      if (j.state === "done") showResult(j);
      return;
    }
  } catch (e) {
    if (++lostContact >= 5) {
      $("progress-error").textContent = "Lost contact with the server. Check that ./start-web.sh is still running.";
      $("progress-error").hidden = false;
    }
  }
  pollTimer = setTimeout(poll, 1000);
}

/* ---- result: figures, downloads and the page-by-page inspector ----------------- */

const inspector = new PageWipe(parts("r-"));
const view = { jobId: null, total: 1, token: 0 };

function figure(label, value, sub) {
  const box = el("div", "figure");
  box.appendChild(el("dt", "", label));
  const dd = el("dd", "", value);
  if (sub) dd.appendChild(el("span", "figure-sub", sub));
  box.appendChild(dd);
  return box;
}

function showResult(j) {
  const r = j.result;
  const statuses = Object.entries(r.tables.status || {}).map(([k, n]) => `${n} ${k.replace("_", " ")}`).join(", ");
  $("results-title").textContent = `${j.name}, read`;
  $("results-lede").textContent = r.complete
    ? `${r.pages_done === 1 ? "The page was" : `All ${r.pages_done} pages were`} read in ${duration(r.seconds)}.`
    : `${r.pages_done} of ${r.pages_total} pages were read before it stopped (${r.stopped || "stopped"}).`;
  const figures = $("result-figures");
  figures.replaceChildren(
    figure("Cost", `₹${r.cost_inr.toFixed(2)}`),
    figure("Reading time", duration(r.seconds)),
    figure("Start to finish", clock(j.elapsed_s)),
    figure("Tables", r.tables.found ? `${r.tables.after_joining} found` : "None found", statuses || ""),
  );
  $("dl-doc").href = `/api/jobs/${j.id}/download/doc`;
  $("dl-report").href = `/api/jobs/${j.id}/download/report`;
  $("results").hidden = false;
  Object.assign(view, { jobId: j.id, total: r.pages_total });
  $("r-range").max = String(view.total);
  inspector.setPosition(50);
  showPage(1);
  $("results").scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth", block: "start" });
}

async function showPage(n) {
  n = Math.max(1, Math.min(view.total, n));
  const token = ++view.token;
  $("r-range").value = String(n);
  $("r-prev").disabled = n <= 1;
  $("r-next").disabled = n >= view.total;
  $("r-title").textContent = `Page ${n} of ${view.total}`;
  try {
    const data = await api(`/api/jobs/${view.jobId}/pages/${n}`);
    if (token !== view.token) return; // a newer page was asked for meanwhile
    const copied = await inspector.load(data, `/api/jobs/${view.jobId}/pages/${n}/image`);
    $("r-sub").textContent = `Read by ${data.engine === "bodhan" ? "Bodhan" : "Sarvam"}, ${data.blocks.filter((b) => b.shown).length} blocks with text${copied ? ", figures copied from the page" : ""}`;
  } catch (e) {
    if (token === view.token) $("r-sub").textContent = e.message;
  }
}

$("r-prev").addEventListener("click", () => showPage(Number($("r-range").value) - 1));
$("r-next").addEventListener("click", () => showPage(Number($("r-range").value) + 1));
$("r-range").addEventListener("input", () => showPage(Number($("r-range").value)));

refreshStatus();
updateStart();
