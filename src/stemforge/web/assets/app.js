/* StemForge — single-page workstation (vanilla JS, no build step). */
"use strict";

const WORKFLOWS = {
  extract: {
    title: "Extract Stems", sub: "Full mix → isolated stems",
    glyph: "◇", endpoint: "/api/extract", cta: "Extract Stems",
    quality: true, melodic: false, dz: "Drop a mix",
  },
  drums: {
    title: "Drum Teardown", sub: "Drum loop → per-hit stems + GM drum MIDI",
    glyph: "◈", endpoint: "/api/drum-teardown", cta: "Tear Down Kit",
    quality: false, melodic: false, dz: "Drop a drum loop or drums stem",
  },
  melodic: {
    title: "Melodic → MIDI", sub: "Stem or mix → transcribed .mid",
    glyph: "◉", endpoint: "/api/melodic-midi", cta: "Transcribe MIDI",
    quality: false, melodic: true, dz: "Drop a stem or a mix",
  },
  full: {
    title: "Full Teardown", sub: "One drop → stems + drum hits + MIDI + tempo",
    glyph: "✦", endpoint: "/api/full-teardown", cta: "Full Teardown",
    quality: true, melodic: false, dz: "Drop a track",
  },
  matchbpm: {
    title: "Match BPM", sub: "Stretch a whole file to a target BPM · pitch preserved",
    glyph: "⇋", endpoint: "/api/match-bpm", cta: "Match BPM",
    quality: false, melodic: false, matchbpm: true, dz: "Drop a track",
  },
};
const QUALITY = ["Fast", "Best", "SOTA", "Max"];

const state = {
  wf: "extract",
  file: null,
  quality: "Best",
  device: "auto",
  inputWave: null,
  cardWaves: [],
  lastOutDir: null,
};

const $ = (id) => document.getElementById(id);

/* ------------------------------ rail ------------------------------------ */
function renderRail() {
  const nav = $("rail-nav");
  nav.innerHTML = "";
  for (const [id, wf] of Object.entries(WORKFLOWS)) {
    const item = document.createElement("div");
    item.className = "rail-item" + (id === state.wf ? " active" : "");
    item.dataset.wf = id;
    item.innerHTML = `<span class="ri-glyph">${wf.glyph}</span><span>${wf.title}</span>`;
    item.addEventListener("click", () => selectWorkflow(id));
    nav.appendChild(item);
  }
}

function renderQuality() {
  const seg = $("quality");
  seg.innerHTML = "";
  for (const q of QUALITY) {
    const b = document.createElement("button");
    b.className = "pill" + (q === state.quality ? " active" : "");
    b.textContent = q;
    b.addEventListener("click", () => {
      state.quality = q;
      renderQuality();
    });
    seg.appendChild(b);
  }
}

// Show/hide a control block. Sets inline display too: the `.ctrl { display:flex }`
// author rule outranks the UA `[hidden]{display:none}`, so the `hidden` attribute
// alone would NOT hide it — inline display is what actually scopes controls.
function showCtrl(id, show) {
  const el = $(id);
  el.hidden = !show;
  el.style.display = show ? "" : "none";
}

function selectWorkflow(id) {
  state.wf = id;
  const wf = WORKFLOWS[id];
  $("wf-title").textContent = wf.title;
  $("wf-sub").textContent = wf.sub;
  $("dz-title").textContent = wf.dz;
  $("run-btn").textContent = wf.cta;
  // Each panel shows ONLY its own controls.
  showCtrl("ctrl-quality", !!wf.quality);   // Extract Stems + Full Teardown
  showCtrl("ctrl-melodic", !!wf.melodic);    // Melodic → MIDI
  showCtrl("ctrl-matchbpm", !!wf.matchbpm);  // Match BPM (Drum Teardown: none)
  $("results").hidden = true;
  $("head-meta").hidden = true;
  if (wf.matchbpm) {
    $("detect-note").textContent = "drop a file, then detect (or type the source below)";
    $("detect-note").classList.remove("detected");
  }
  renderRail();
}

/* ---------------------------- dropzone ---------------------------------- */
function setupDropzone() {
  const dz = $("dropzone");
  const input = $("file-input");
  dz.addEventListener("click", () => input.click());
  dz.addEventListener("keydown", (e) => { if (e.key === "Enter") input.click(); });
  input.addEventListener("change", () => input.files[0] && loadFile(input.files[0]));
  ["dragover", "dragenter"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files[0];
    if (f) loadFile(f);
  });
}

function loadFile(file) {
  state.file = file;
  $("dz-file").hidden = false;
  $("dz-file").textContent = `▸ ${file.name}`;
  $("run-btn").disabled = false;

  const waveEl = $("input-wave");
  waveEl.hidden = false;
  if (state.inputWave) state.inputWave.destroy();
  state.inputWave = WaveSurfer.create({
    container: waveEl, height: 64, waveColor: "#39456a", progressColor: "#22d3ee",
    cursorColor: "#a78bfa", barWidth: 2, barGap: 1, barRadius: 2,
    url: URL.createObjectURL(file),
  });
  bindPlayClick(waveEl, state.inputWave);
}

function bindPlayClick(el, ws) {
  el.style.cursor = "pointer";
  el.addEventListener("click", (e) => { e.stopPropagation(); ws.playPause(); });
}

/* ------------------------------ run ------------------------------------- */
async function run() {
  if (!state.file) return;
  const wf = WORKFLOWS[state.wf];
  const fd = new FormData();
  fd.append("file", state.file);
  fd.append("device", state.device);
  if (wf.quality) fd.append("preset", state.quality.toLowerCase());
  if (wf.melodic) {
    fd.append("monophonic", $("opt-mono").checked);
    fd.append("quantize", $("opt-quant").checked);
  }
  if (wf.matchbpm) {
    const target = parseFloat($("target-bpm").value);
    if (!target || target <= 0) {
      $("detect-note").textContent = "enter a target BPM first";
      $("detect-note").classList.remove("detected");
      return;
    }
    fd.append("target_bpm", target);
    const src = parseFloat($("source-bpm").value);
    if (src > 0) fd.append("source_bpm", src);
    fd.append("engine", $("mb-engine").value);
  }

  showProgress(0.05, "Uploading…");
  $("run-btn").disabled = true;
  try {
    const res = await fetch(wf.endpoint, { method: "POST", body: fd });
    const { job_id } = await res.json();
    await poll(job_id);
  } catch (err) {
    finishError(String(err));
  } finally {
    $("run-btn").disabled = false;
  }
}

function showProgress(frac, msg) {
  $("progress").hidden = false;
  $("bar-fill").style.width = `${Math.round(frac * 100)}%`;
  $("progress-msg").textContent = msg || "Working…";
}

async function poll(jobId) {
  for (;;) {
    await new Promise((r) => setTimeout(r, 400));
    const job = await (await fetch(`/api/job/${jobId}`)).json();
    showProgress(job.progress || 0.05, job.message);
    if (job.status === "done") return renderResults(job.result);
    if (job.status === "error") return finishError(job.error || job.message);
  }
}

function finishError(msg) {
  $("progress").hidden = true;
  $("results").hidden = false;
  $("results-title").textContent = "Something went wrong";
  $("cards").innerHTML = "";
  const line = $("status-line");
  line.textContent = msg;
  line.classList.add("err");
}

/* ---------------------------- results ----------------------------------- */
function renderResults(result) {
  $("progress").hidden = true;
  state.lastOutDir = result.out_dir;
  state.cardWaves.forEach((w) => w.destroy());
  state.cardWaves = [];

  // BPM / key chip
  const meta = $("head-meta");
  if (result.bpm != null || result.key != null) {
    meta.hidden = false;
    $("chip-bpm").textContent = result.bpm != null ? `${Math.round(result.bpm)} BPM` : "— BPM";
    $("chip-key").textContent = result.key ? `key ${result.key}` : "key —";
  } else {
    meta.hidden = true;
  }

  $("results").hidden = false;
  $("results-title").textContent = `Results · ${result.cards.length} file${result.cards.length === 1 ? "" : "s"}`;
  const line = $("status-line");
  line.classList.remove("err");
  line.textContent = result.out_dir ? `Saved to ${result.out_dir}` : "";

  const wrap = $("cards");
  wrap.innerHTML = "";
  if (!result.cards.length) {
    line.textContent = "No output produced — a stage was skipped (see manifest). " + line.textContent;
    return;
  }
  for (const card of result.cards) wrap.appendChild(buildCard(card));
}

function buildCard(card) {
  const el = document.createElement("div");
  el.className = "card" + (card.kind === "midi" ? " midi" : "");
  el.innerHTML = `
    <div class="card-top">
      <span class="card-name">${card.name}</span>
      <span class="card-group">${card.group}</span>
    </div>
    <div class="card-wave"></div>
    <div class="card-actions"></div>`;
  const waveEl = el.querySelector(".card-wave");
  const actions = el.querySelector(".card-actions");

  let ws = null;
  if (card.kind === "audio") {
    ws = WaveSurfer.create({
      container: waveEl, height: 52, waveColor: "#39456a", progressColor: "#7c8bff",
      cursorColor: "#22d3ee", barWidth: 2, barGap: 1, barRadius: 2, url: card.url,
    });
    state.cardWaves.push(ws);
    actions.appendChild(iconBtn("▶", "play", () => ws.playPause()));
    actions.appendChild(soloBtn(ws));
  } else {
    waveEl.textContent = "MIDI";
  }
  actions.appendChild(downloadBtn(card.url, card.filename));
  return el;
}

function iconBtn(glyph, cls, onClick) {
  const b = document.createElement("button");
  b.className = `icon-btn ${cls}`;
  b.textContent = glyph;
  b.addEventListener("click", onClick);
  return b;
}

function soloBtn(ws) {
  const b = iconBtn("S", "solo", () => {
    const on = b.classList.toggle("on");
    // solo = mute every other audio card
    state.cardWaves.forEach((w) => { if (w !== ws) w.setMuted(on); });
    if (on && !ws.isPlaying()) ws.play();
  });
  return b;
}

function downloadBtn(url, filename) {
  const b = iconBtn("↓", "dl", () => {
    const a = document.createElement("a");
    a.href = url; a.download = filename || "";
    document.body.appendChild(a); a.click(); a.remove();
  });
  return b;
}

/* ---------------------------- detect BPM -------------------------------- */
async function detectBpm() {
  if (!state.file) {
    $("detect-note").textContent = "drop a file first";
    return;
  }
  const note = $("detect-note");
  note.classList.remove("detected");
  note.textContent = "detecting…";
  const fd = new FormData();
  fd.append("file", state.file);
  fd.append("device", state.device);
  try {
    const r = await (await fetch("/api/detect-bpm", { method: "POST", body: fd })).json();
    if (!r.bpm) {
      note.textContent = "couldn't detect a BPM — type the source override below";
      return;
    }
    if (!$("target-bpm").value) $("target-bpm").value = Math.round(r.bpm);
    note.classList.add("detected");
    note.textContent = `detected ${r.bpm} BPM · half ${r.half} · double ${r.double} — wrong? set the source override`;
  } catch {
    note.textContent = "detection failed — type the source override below";
  }
}

/* ------------------------- footer actions ------------------------------- */
function setupActions() {
  $("run-btn").addEventListener("click", run);
  $("detect-btn").addEventListener("click", detectBpm);
  $("download-all").addEventListener("click", () => {
    if (state.lastOutDir) window.location = `/api/download-all?dir=${encodeURIComponent(state.lastOutDir)}`;
  });
  $("open-folder").addEventListener("click", async () => {
    if (!state.lastOutDir) return;
    await fetch("/api/open-folder", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: state.lastOutDir }),
    });
  });
}

/* ------------------------------ health ---------------------------------- */
async function loadHealth() {
  try {
    const h = await (await fetch("/api/health")).json();
    state.device = h.device || "auto";
    const dev = $("chip-device");
    dev.classList.toggle("ok", !!h.cuda);
    dev.querySelector("span:last-child").textContent =
      h.cuda ? `GPU · ${h.gpu}` : `CPU (${h.device})`;
    const model = $("chip-model");
    const ready = h.drumsep_model || h.uvr_venv;
    model.classList.toggle("ok", !!ready);
    model.classList.toggle("warn", !ready);
    model.querySelector("span:last-child").textContent =
      h.drumsep_model ? "drumsep ready" : (h.uvr_venv ? "uvr ready" : "models: setup needed");
  } catch {
    /* health is best-effort */
  }
}

/* ------------------------------ boot ------------------------------------ */
window.addEventListener("DOMContentLoaded", () => {
  renderRail();
  renderQuality();
  selectWorkflow("extract");
  setupDropzone();
  setupActions();
  loadHealth();
});
