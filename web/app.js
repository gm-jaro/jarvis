"use strict";

// Jarvis web page: server events (SSE /events) drawn as a timeline, plus the camera preview,
// model selection (the "brain" controls the camera, the "eyes" describe the photo) and system status.

const $ = (id) => document.getElementById(id);
const timeline = $("timeline");
const cards = new Map(); // tool call id -> card on the timeline
let current = null;      // block of the current brain step
let config = null;       // selected models and the lists to choose from ("config" event)
let busy = false;
let visionLive = 0;

// ---------- helpers ----------

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

function nearBottom() {
  return timeline.scrollHeight - timeline.scrollTop - timeline.clientHeight < 120;
}

function keepBottom(change) {
  const stick = nearBottom();
  change();
  if (stick) timeline.scrollTop = timeline.scrollHeight;
}

function append(node) {
  keepBottom(() => timeline.appendChild(node));
  return node;
}

function header(tagClass, tagText, metaText) {
  const who = el("div", "who");
  who.append(el("span", `tag ${tagClass}`, tagText));
  if (metaText != null) who.append(el("span", "meta", metaText));
  return who;
}

function mdLite(text) {
  const esc = text.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  return esc.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/\n/g, "<br>");
}

function setNow(text, tone = "") {
  $("now").textContent = text;
  $("now").dataset.tone = tone;
}

function setChip(id, state, tone = "", model) {
  const chip = $(id);
  chip.querySelector(".state").textContent = state;
  if (model !== undefined) chip.querySelector(".model").textContent = model;
  chip.dataset.tone = tone;
}

function brainLabel() {
  return config ? config.brains[config.brain] : "Brain";
}

function visionLabel() {
  return config ? config.visions[config.vision] : "Eyes";
}

async function post(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch { /* response without JSON */ }
    append(el("div", "note error", msg));
  }
  return r;
}

// ---------- events ----------

function clearTimeline() {
  timeline.replaceChildren();
  cards.clear();
  current = null;
  visionLive = 0;
}

function fillSelect(select, options, value) {
  select.replaceChildren(...Object.entries(options).map(([key, label]) => {
    const o = el("option", null, label);
    o.value = key;
    return o;
  }));
  select.value = value;
}

function onConfig(ev) {
  config = ev;
  fillSelect($("sel-brain"), ev.brains, ev.brain);
  fillSelect($("sel-vision"), ev.visions, ev.vision);
  setChip("chip-brain", "…", "", brainLabel());
  setChip("chip-vision", "…", "", visionLabel());
  pollSys();
}

function onStatus(ev) {
  busy = ev.busy;
  $("btn-stop").hidden = !busy;
  for (const b of document.querySelectorAll("[data-lock]")) b.disabled = busy;
  if (!busy) setNow("Waiting for a command.", "");
}

function onUser(ev) {
  const box = el("div", "item user");
  box.append(el("div", "bubble", ev.text));
  append(box);
}

function onBrainStart(ev) {
  const step = `step ${ev.step + 1}`;
  const box = el("div", "item brain");
  const who = header("brain", ev.model, ev.think ? `${step} · thinking…` : `${step} · writing…`);
  const think = el("details", "think");
  think.open = true;
  think.hidden = !ev.think;
  const thinkText = el("div", "think-text");
  think.append(el("summary", null, "Reasoning"), thinkText);
  const draft = el("div", "draft");
  box.append(who, think, draft);
  current = { meta: who.querySelector(".meta"), think, thinkText, draft, content: "" };
  append(box);
  setChip("chip-brain", "thinking", "active");
  setNow(ev.think ? `${ev.model} is thinking (${step})…` : `${ev.model} is answering (${step})…`, "brain");
}

function onBrainToken(ev) {
  if (!current) return;
  keepBottom(() => {
    if (ev.kind === "think") {
      current.think.hidden = false;
      current.thinkText.textContent += ev.text;
      current.thinkText.scrollTop = current.thinkText.scrollHeight;
      return;
    }
    current.content += ev.text;
    const writingTool = current.content.includes("<tool_call>");
    const visible = current.content.replace(/<tool_call>[\s\S]*?(<\/tool_call>|$)/g, "").trim();
    current.draft.textContent = visible + (writingTool ? "\n✎ preparing a tool call…" : "");
    if (writingTool) setNow("The brain is picking a tool…", "brain");
  });
}

function onBrainEnd(ev) {
  if (!current) return;
  current.meta.textContent = `step ${ev.step + 1} · ${ev.tokens} tok · ${ev.seconds} s · ${ev.tok_s} tok/s`;
  current.think.open = false;
  if (!current.thinkText.textContent.trim()) current.think.hidden = true;
  current.draft.remove(); // calls and the answer are shown by separate cards
  current = null;
  setChip("chip-brain", "ready", "ok");
}

function onToolCall(ev) {
  const box = el("div", "item tool");
  const who = header("tool", "Tool");
  who.append(el("code", null, `${ev.name}(${JSON.stringify(ev.args)})`));
  box.append(who, el("div", "label", ev.label));
  cards.set(ev.id, box);
  append(box);
  setNow(`${ev.label}…`, "tool");
}

function onToolResult(ev) {
  const box = cards.get(ev.id);
  if (!box || ev.name === "describe_view") return; // the description is already shown in the eyes section
  keepBottom(() => box.append(el("div", "result", "→ " + ev.result.split(" Widok: ")[0])));
}

function visionSection(id) {
  let box = cards.get(id);
  if (!box) { // automatic description (between turns) or manual from the button
    box = el("div", "item tool");
    box.append(header("vision", "Eyes", id.startsWith("manual") ? "description on request" : "automatic, between turns"));
    cards.set(id, box);
    append(box);
  }
  let v = box.querySelector(".vision");
  if (!v) {
    v = el("div", "vision");
    const link = el("a", "thumb");
    link.target = "_blank";
    const img = el("img");
    img.alt = "Photo sent to the vision model";
    link.append(img);
    const body = el("div");
    body.append(el("div", "vision-head", "Eyes are looking"), el("div", "vision-text"));
    v.append(link, body);
    keepBottom(() => box.append(v));
  }
  return v;
}

function onSnapshot(ev) {
  const v = visionSection(ev.id);
  v.querySelector(".thumb").href = ev.url;
  v.querySelector("img").src = ev.url;
  v.dataset.where = ev.where;
}

function onVisionStart(ev) {
  const v = visionSection(ev.id);
  let head = `${ev.model} is looking · camera: ${v.dataset.where || "?"}`;
  if (ev.question) head += ` · question: “${ev.question}”`;
  v.querySelector(".vision-head").textContent = head;
  v.classList.add("live");
  visionLive++;
  setChip("chip-vision", "describing", "active");
  setNow(`${ev.model} is looking at the photo and describing it…`, "vision");
}

function onVisionToken(ev) {
  const v = visionSection(ev.id);
  keepBottom(() => { v.querySelector(".vision-text").textContent += ev.text; });
}

function onVisionEnd(ev) {
  const v = visionSection(ev.id);
  v.classList.remove("live");
  v.querySelector(".vision-head").textContent += ` · ${ev.seconds} s`;
  visionLive = Math.max(0, visionLive - 1);
  setChip("chip-vision", "in VRAM", "ok");
}

function onAnswer(ev) {
  const box = el("div", "item answer");
  const bubble = el("div", "bubble");
  bubble.innerHTML = mdLite(ev.text);
  box.append(header("jarvis", "Jarvis"), bubble);
  append(box);
  setNow("Done.", "ok");
}

function onHead(ev) {
  $("cam-where").textContent = `camera: ${ev.where}`;
  setChip("chip-head", ev.where);
  drawDial(ev);
}

const handlers = {
  hello: clearTimeline,
  reset: clearTimeline,
  config: onConfig,
  status: onStatus,
  user: onUser,
  brain_start: onBrainStart,
  brain_token: onBrainToken,
  brain_end: onBrainEnd,
  tool_call: onToolCall,
  tool_result: onToolResult,
  snapshot: onSnapshot,
  vision_start: onVisionStart,
  vision_token: onVisionToken,
  vision_end: onVisionEnd,
  answer: onAnswer,
  head: onHead,
  manual: (ev) => append(el("div", "note", ev.label)),
  notice: (ev) => append(el("div", "note", ev.text)),
  error: (ev) => append(el("div", "note error", "Error: " + ev.message)),
  cancelled: () => append(el("div", "note", "Stopped.")),
};

// ---------- camera direction dials ----------

// "pan": top view, 0° = straight ahead (up in the drawing), positive to the right.
// "tilt": side view, 0° = straight ahead (right in the drawing), positive upwards.
function polar(cx, cy, r, deg, mode) {
  const a = (deg * Math.PI) / 180;
  return mode === "pan" ? [cx + r * Math.sin(a), cy - r * Math.cos(a)] : [cx + r * Math.cos(a), cy - r * Math.sin(a)];
}

function sector(cx, cy, r, a0, a1, mode) {
  const [x0, y0] = polar(cx, cy, r, a0, mode);
  const [x1, y1] = polar(cx, cy, r, a1, mode);
  const sweep = mode === "pan" ? 1 : 0; // increasing angle: clockwise for pan, counterclockwise for tilt
  return `M${cx} ${cy} L${x0.toFixed(1)} ${y0.toFixed(1)} A${r} ${r} 0 0 ${sweep} ${x1.toFixed(1)} ${y1.toFixed(1)} Z`;
}

function gauge(cx, cy, r, value, range, mode) {
  const [x, y] = polar(cx, cy, r - 4, value, mode);
  return `<path class="g-bg" d="${sector(cx, cy, r, -90, 90, mode)}" opacity=".45"/>
    <path class="g-range" d="${sector(cx, cy, r, range[0], range[1], mode)}"/>
    <line class="g-needle" x1="${cx}" y1="${cy}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}"/>
    <circle class="g-hub" cx="${cx}" cy="${cy}" r="4"/>`;
}

function drawDial(s) {
  $("dial").innerHTML = `
    ${gauge(62, 70, 52, s.right, s.right_range, "pan")}
    <text class="g-label" x="10" y="84">left</text>
    <text class="g-label" x="84" y="84">right</text>
    <text class="g-label" x="62" y="104" text-anchor="middle">rotation</text>
    <text class="g-label" x="62" y="120" text-anchor="middle">${s.right > 0 ? s.right + "° right" : s.right < 0 ? -s.right + "° left" : "straight ahead"}</text>
    ${gauge(146, 66, 52, s.up, s.up_range, "tilt")}
    <text class="g-label" x="150" y="12">up</text>
    <text class="g-label" x="150" y="128">down</text>
    <text class="g-label" x="200" y="62" text-anchor="middle">front</text>`;
  $("dial").setAttribute("aria-label", `Camera: ${s.where}`);
}

// ---------- system status ----------

const MODEL_STATE = {
  ready: ["ready", "ok"],
  loaded: ["in VRAM", "ok"],
  idle: ["not in VRAM", ""],
  offline: ["offline", "bad"],
};

async function pollSys() {
  try {
    const s = await (await fetch("/api/sys")).json();
    if (!current) setChip("chip-brain", ...MODEL_STATE[s.brain]);
    if (!visionLive) setChip("chip-vision", ...MODEL_STATE[s.vision]);
    const g = s.gpu;
    $("sys").textContent = g
      ? `GPU ${g.busy}% · VRAM ${(g.vram_used / 1e9).toFixed(1)}/${(g.vram_total / 1e9).toFixed(1)} GB · ` +
        `${g.temp_c}°C · ${g.power_w} W · CPU load ${s.load}`
      : `CPU load ${s.load}`;
  } catch {
    $("sys").textContent = "no connection to the server";
  }
}

// ---------- connection and controls ----------

function connect() {
  const es = new EventSource("/events");
  es.onopen = () => {
    $("conn-dot").dataset.tone = "ok";
    $("cam").src = "/stream.mjpg?" + Date.now(); // after a server restart the preview has to be opened again
  };
  es.onerror = () => {
    $("conn-dot").dataset.tone = "bad";
    setNow("No connection to the Jarvis server, retrying…", "bad");
  };
  es.onmessage = (m) => {
    const ev = JSON.parse(m.data);
    handlers[ev.type]?.(ev);
  };
}

async function send(text) {
  text = text.trim();
  if (!text || busy) return;
  const r = await post("/api/chat", { text, think: $("think").checked });
  if (r.ok) $("input").value = "";
}

async function changeModel(role, value) {
  const r = await post("/api/config", { [role]: value });
  if (!r.ok && config) onConfig(config); // restore the previous selection in the lists
}

$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  send($("input").value);
});
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send($("input").value);
  }
});
for (const b of $("suggest").querySelectorAll("button")) b.addEventListener("click", () => send(b.textContent));
for (const b of document.querySelectorAll("[data-dir]")) {
  b.addEventListener("click", () => post("/api/look", { direction: b.dataset.dir, degrees: Number($("step").value) }));
}
$("sel-brain").addEventListener("change", (e) => changeModel("brain", e.target.value));
$("sel-vision").addEventListener("change", (e) => changeModel("vision", e.target.value));
$("btn-describe").addEventListener("click", () => post("/api/describe"));
$("btn-reset").addEventListener("click", () => post("/api/reset"));
$("btn-stop").addEventListener("click", () => post("/api/stop"));

connect();
setInterval(pollSys, 2000);
