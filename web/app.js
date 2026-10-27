// Recording + waveform: Mohammad's wavesurfer code (see citations in index.html / Readme.md).

import WaveSurfer from 'https://unpkg.com/wavesurfer.js@7/dist/wavesurfer.esm.js';
import RecordPlugin from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/record.esm.js';

const DEFAULT_SPACE = "https://pineappleananas-speechcorrection.hf.space";

const apiParam = new URLSearchParams(location.search).get('api');
const SPACE_URL = apiParam && /^http:\/\/(127\.0\.0\.1|localhost)(:\d+)?$/.test(apiParam) ? apiParam : DEFAULT_SPACE;
const API_PREFIXES = ["/gradio_api", ""];
const UNSURE = 0.6;
const MAX_HEALTH_RETRIES = 45;
const SKIP_WORDS = new Set(["a", "an", "the", "to", "of", "and", "i"]);

const $ = (id) => document.getElementById(id);
const btn = $("btn"), btnWrap = $("btnWrap"), status = $("status"), spinner = $("spinner");
const liquidEl = document.querySelector('.liquid-body');

let isRecording = false, serverReady = false, isBusy = false, mango = 0;
let current = null;
let apiPrefix = API_PREFIXES[0];

const STORE_KEY = "articulate.v1";
function loadStore() {
  try {
    const s = JSON.parse(localStorage.getItem(STORE_KEY) || "null");
    if (s && Array.isArray(s.entries)) return Object.assign({ practice: {}, favorites: [], settings: {} }, s);
  } catch (e) {   }
  return { entries: [], practice: {}, favorites: [], settings: { askBeforeSpeaking: true } };
}
const store = loadStore();
if (store.settings.askBeforeSpeaking === undefined) store.settings.askBeforeSpeaking = true;
if (!Array.isArray(store.favorites)) store.favorites = [];
function saveStore() {
  const cutoff = Date.now() - 60 * 864e5;
  store.entries = store.entries.filter(e => e.ts >= cutoff).slice(-2000);
  try { localStorage.setItem(STORE_KEY, JSON.stringify(store)); } catch (e) {   }
}

const isHighContrast = () => document.body.classList.contains("active");
const buttonImage = (kind) => `Images/${kind}_Button_${isHighContrast() ? "hc" : "teal"}.png`;
let AloVera = "rgba(255,250,240,0.7)";
function applyTheme() {
  AloVera = getComputedStyle(document.body).getPropertyValue("--wave").trim() || AloVera;
  const b = document.getElementById("btn");
  if (b) b.src = buttonImage(isRecording ? "Pause" : "Play");
  const t = document.getElementById("toggle-contrast");
  if (t) t.textContent = `High contrast: ${isHighContrast() ? "On" : "Off"}`;
  const m = document.querySelector('meta[name="theme-color"]');
  if (m) m.content = isHighContrast() ? "#000000" : "#0f4c4c";
}
document.addEventListener("themechange", applyTheme);
applyTheme();

const LEBRON = new Array(28).fill(4);
const wavesurfer = WaveSurfer.create({
  container: '#waveform', waveColor: 'rgba(255,255,255,0.6)', progressColor: 'transparent',
  height: 80, interact: false,
  renderFunction: (channels, ctx) => {
    const { width, height } = ctx.canvas;
    const BAR_COUNT = 28, BAR_WIDTH = 12, BAR_GAP = 9, BAR_RADIUS = 6, centerY = height / 2;
    const data = channels[0];
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = AloVera;
    for (let i = 0; i < BAR_COUNT; i++) {
      const index = Math.floor(i * data.length / BAR_COUNT);
      if (isRecording) {
        const raw = Math.abs(data[index]) * height * 1.5;
        LEBRON[i] = LEBRON[i] * 0.85 + raw * 0.15;
      }
      const edge = Math.min(i, BAR_COUNT - 1 - i);
      const taper = edge < 2 ? 0.4 : 1;
      const barH = Math.max(4, LEBRON[i] * taper);
      const x = i * (BAR_WIDTH + BAR_GAP) + (width - BAR_COUNT * (BAR_WIDTH + BAR_GAP)) / 2;
      ctx.beginPath();
      ctx.roundRect(x, centerY - barH / 2, BAR_WIDTH, barH, BAR_RADIUS);
      ctx.fill();
    }
  },
});
const recorder = wavesurfer.registerPlugin(RecordPlugin.create({ scrollingWaveform: false }));
recorder.on('record-end', async (blob) => { await handleStop(blob); });

function setStatus(msg, cls) { status.textContent = msg; status.className = "status " + (cls || ""); }
function setButtonEnabled(on) { btnWrap.classList.toggle('disabled', !on); }
function scaleUp() { liquidEl.style.transform = 'scale(1.1)'; liquidEl.classList.add('pulsing'); }
function scaleDown() { liquidEl.style.transform = 'scale(1)'; liquidEl.classList.remove('pulsing'); }
function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
const normWord = (w) => (w || "").toLowerCase().replace(/’/g, "'").replace(/[^a-z0-9']/g, "");
const sentenceOf = (words) => words.map(w => w.text).filter(Boolean).join(" ");
const dayKey = (ts) => { const d = new Date(ts); return `${d.getFullYear()}-${d.getMonth() + 1}-${d.getDate()}`; };

function speak(text, rate) {
  if (!text) return;
  if (!window.speechSynthesis || typeof window.SpeechSynthesisUtterance !== "function") {
    setStatus("Speech isn't supported in this browser", "error"); return;
  }
  const u = new SpeechSynthesisUtterance(text);
  u.lang = "en-US"; u.rate = rate || 0.95;
  u.onend = () => { if (!isBusy) setStatus("Tap to speak"); };
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(u);
  setStatus("Speaking...", "active");
}

async function fetchT(url, opts, ms) {
  const c = new AbortController(); const t = setTimeout(() => c.abort(), ms || 20000);
  try { return await fetch(url, Object.assign({}, opts || {}, { signal: c.signal })); }
  finally { clearTimeout(t); }
}

async function uploadBlob(blob, name) {
  const fd = new FormData();
  fd.append("files", blob, name);
  let lastErr;
  for (const pre of [apiPrefix, ...API_PREFIXES.filter(p => p !== apiPrefix)]) {
    try {
      const r = await fetchT(SPACE_URL + pre + "/upload", { method: "POST", body: fd }, 30000);
      if (r.status === 404) { lastErr = new Error("upload not found"); continue; }
      if (!r.ok) throw new Error("Upload failed (" + r.status + ")");
      const data = await r.json();
      apiPrefix = pre;
      const path = Array.isArray(data) ? (typeof data[0] === "string" ? data[0] : data[0] && data[0].path) : null;
      if (!path) throw new Error("Upload response missing file path");
      return { path, orig_name: name, meta: { _type: "gradio.FileData" } };
    } catch (e) { lastErr = e; }
  }
  throw lastErr;
}

async function callApi(name, data, ms) {
  const r = await fetchT(SPACE_URL + apiPrefix + "/call/" + name, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ data }),
  }, 30000);
  if (!r.ok) throw new Error("Request failed (" + r.status + ")");
  const { event_id } = await r.json();
  const res = await fetchT(SPACE_URL + apiPrefix + "/call/" + name + "/" + event_id, {}, ms || 120000);
  const text = await res.text();
  let ev = "", out = null, err = null;
  for (const line of text.split("\n")) {
    if (line.startsWith("event:")) ev = line.slice(6).trim();
    else if (line.startsWith("data:")) {
      const raw = line.slice(5).trim();
      if (ev === "complete") { try { out = JSON.parse(raw); } catch (e) {   } }
      if (ev === "error") err = raw && raw !== "null" ? raw : "Server error";
    }
  }
  if (out == null) throw new Error(err || "No result from server");
  return Array.isArray(out) ? out[0] : out;
}

function waitForServer() {
  fetchT(SPACE_URL + "/", {}, 8000).then(res => {
    if (res.ok) { serverReady = true; setButtonEnabled(true); setStatus("Tap to speak"); return; }
    throw new Error("not ready");
  }).catch(() => {
    mango++;
    if (mango >= MAX_HEALTH_RETRIES) { setStatus("Server unavailable", "error"); return; }
    if (mango === 3) setStatus("Waking up the server", "warming dot-pulse");
    setTimeout(waitForServer, 4000);
  });
}
window.addEventListener("load", waitForServer);

async function startRecording() {
  try {
    await recorder.startRecording();
    isRecording = true;
    btn.src = buttonImage("Pause"); btn.alt = "Stop recording";
    scaleUp(); setStatus("Recording...", "active");
  } catch (err) { setStatus("Microphone access denied", "error"); }
}
function stopRecording() {
  recorder.stopRecording();
  isRecording = false; LEBRON.fill(4);
  btn.src = buttonImage("Play"); btn.alt = "Start recording";
  scaleDown();
}
function onMainButton() {
  if (!serverReady || isBusy) return;
  if (isRecording) stopRecording(); else startRecording();
}
btn.addEventListener("click", onMainButton);
btn.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onMainButton(); } });

async function handleStop(blob) {
  if (isBusy) return;
  if (!blob || blob.size === 0) { setStatus("No audio captured", "error"); return; }
  isBusy = true; setButtonEnabled(false); spinner.classList.add("visible"); setStatus("Processing...", "working");
  try {
    const ext = (blob.type || "").includes("mp4") ? "m4a" : "webm";
    const file = await uploadBlob(blob, "recording." + ext);
    const r = await callApi("process_audio", [file]);
    if (!r || r.error) throw new Error((r && r.error) || "No usable result from server");
    if (!r.corrected) throw new Error("No usable result from server");
    const words = (r.words && r.words.length) ? r.words : r.corrected.split(/\s+/).map(t => ({ text: t, conf: 1, alts: [] }));
    const entry = {
      id: Date.now() + "-" + Math.random().toString(36).slice(2, 7), ts: Date.now(),
      raw: r.raw || "", corrected: r.corrected, final: r.corrected,
      words: words.map(w => ({ text: w.text, conf: w.conf, alts: w.alts || [] })),
      alternatives: r.alternatives || [], swaps: [], audio: file, confidence: r.confidence,
      needsReview: !!r.needs_review,
    };
    store.entries.push(entry); saveStore();
    showResult(entry, true);
    spinner.classList.remove("visible");
    if (entry.needsReview && store.settings.askBeforeSpeaking) setStatus("Check it, then press Speak", "active");
    else speak(entry.final);
  } catch (err) {
    spinner.classList.remove("visible");
    setStatus(err && err.message ? err.message : "Unexpected failure", "error");
    console.error(err);
    setTimeout(() => { if (!isBusy) setStatus("Tap to speak"); }, 3500);
  } finally {
    isBusy = false; setButtonEnabled(serverReady);
  }
}

function showResult(entry, scroll) {
  current = entry;
  $("result").hidden = false;
  $("review-note").hidden = !(entry.needsReview && store.settings.askBeforeSpeaking);
  const s = $("result-sentence"); s.innerHTML = "";
  entry.words.forEach((w, i) => {
    if (!w.text) return;
    const b = el("button", "word" + (w.conf < UNSURE ? " unsure" : ""), w.text);
    b.setAttribute("aria-label", `${w.text}. Tap for other words.`);
    b.addEventListener("click", (ev) => openWordPop(i, ev.currentTarget));
    s.appendChild(b);
    s.appendChild(document.createTextNode(" "));
  });
  $("result-heard").textContent = entry.raw && normSentence(entry.raw) !== normSentence(entry.final) ? `Heard: "${entry.raw}"` : "";
  const alts = $("result-alts"); alts.innerHTML = "";
  if (entry.needsReview && entry.alternatives.length) {
    alts.appendChild(el("span", "alts-label", "Or did you mean:"));
    entry.alternatives.forEach(a => {
      const c = el("button", "pill small", a);
      c.addEventListener("click", () => {
        entry.swaps.push({ from: entry.final, to: a, whole: true });
        entry.words = a.split(/\s+/).map(t => ({ text: t, conf: 1, alts: [] }));
        entry.final = a; entry.needsReview = false; saveStore(); showResult(entry); speak(a);
      });
      alts.appendChild(c);
    });
  }
  renderFavButton();
  if (scroll) $("result").scrollIntoView({ behavior: "smooth", block: "nearest" });
}
const normSentence = (s) => (s || "").split(/\s+/).map(normWord).filter(Boolean).join(" ");
$("speak-btn").addEventListener("click", () => { if (current) speak(current.final); });

let popIndex = -1;
function openWordPop(i, anchor) {
  popIndex = i;
  const pop = $("word-pop"), list = $("word-pop-list");
  list.innerHTML = "";
  const w = current.words[i];
  const opts = (w.alts || []).filter(o => o && o.span);
  if (!opts.length) list.appendChild(el("p", "word-pop-empty", "No other guesses yet."));
  opts.forEach(o => list.appendChild(optionButton(o)));
  $("word-pop-more").disabled = false;
  $("word-pop-more").textContent = "More options";
  pop.hidden = false;
  const r = anchor.getBoundingClientRect();
  const left = Math.min(Math.max(8, r.left + window.scrollX), window.scrollX + document.documentElement.clientWidth - pop.offsetWidth - 8);
  pop.style.left = left + "px";

  const below = window.innerHeight - r.bottom - 90;
  pop.style.top = (below < pop.offsetHeight && r.top > pop.offsetHeight + 12)
    ? (r.top + window.scrollY - pop.offsetHeight - 6) + "px"
    : (r.bottom + window.scrollY + 6) + "px";
}
function optionButton(o) {
  const b = el("button", "word-opt", o.text ? o.text : "(remove this word)");
  b.addEventListener("click", () => applyOption(o));
  return b;
}
function closeWordPop() { $("word-pop").hidden = true; popIndex = -1; }
document.addEventListener("click", (e) => {
  const pop = $("word-pop");
  if (!pop.hidden && !pop.contains(e.target) && !e.target.classList.contains("word")) closeWordPop();
});
document.addEventListener("keydown", (e) => { if (e.key === "Escape") { closeWordPop(); closePanels(); } });

function applyOption(o) {
  const [a, b] = o.span;
  const oldText = current.words.slice(a, b).map(w => w.text).join(" ");
  const newWords = o.text ? o.text.split(/\s+/).map(t => ({ text: t, conf: 1, alts: [] })) : [];
  const delta = newWords.length - (b - a);
  current.words.forEach((w, i) => {
    if (i >= a && i < b) return;
    w.alts = (w.alts || []).filter(x => x.span[1] <= a || x.span[0] >= b)
      .map(x => x.span[0] >= b ? Object.assign({}, x, { span: [x.span[0] + delta, x.span[1] + delta] }) : x);
  });
  current.words.splice(a, b - a, ...newWords);
  current.swaps.push({ from: oldText, to: o.text });
  current.final = sentenceOf(current.words);
  current.needsReview = false;
  saveStore(); closeWordPop(); showResult(current);
}

$("word-pop-more").addEventListener("click", async (e) => {
  e.stopPropagation();
  if (!current || popIndex < 0) return;
  const btnMore = $("word-pop-more"); btnMore.disabled = true; btnMore.textContent = "Listening again...";
  try {
    const idx = popIndex;
    const r = await callApi("more_alternatives", [current.audio, current.raw, sentenceOf(current.words), idx], 60000);
    const list = $("word-pop-list");
    if (r.error) throw new Error(r.error);
    const have = new Set([...list.querySelectorAll(".word-opt")].map(b => normWord(b.textContent)));
    const fresh = (r.options || []).filter(o => !have.has(normWord(o.text || "(remove this word)")));
    const empty = list.querySelector(".word-pop-empty"); if (empty && fresh.length) empty.remove();
    fresh.forEach(o => list.appendChild(optionButton(o)));
    if (r.note) list.appendChild(el("p", "word-pop-empty", r.note));
    btnMore.textContent = fresh.length ? "More options" : "No more options";
    btnMore.disabled = !fresh.length;
  } catch (err) {
    btnMore.textContent = "Couldn't load options"; console.error(err);
  }
});

function renderReviewToggle() { $("toggle-review").textContent = "Ask before speaking: " + (store.settings.askBeforeSpeaking ? "On" : "Off"); }
$("toggle-review").addEventListener("click", () => { store.settings.askBeforeSpeaking = !store.settings.askBeforeSpeaking; saveStore(); renderReviewToggle(); });
renderReviewToggle();

function hardWords(sinceTs, untilTs) {
  const score = new Map(), times = new Map();
  const add = (w, pts) => {
    const k = normWord(w); if (k.length < 2 || SKIP_WORDS.has(k)) return;
    score.set(k, (score.get(k) || 0) + pts); times.set(k, (times.get(k) || 0) + 1);
  };
  for (const e of store.entries) {
    if (e.ts < sinceTs || (untilTs && e.ts >= untilTs)) continue;
    for (const w of e.words) if (w.conf < UNSURE) add(w.text, 1);
    for (const s of e.swaps) if (!s.whole) (s.to || "").split(/\s+/).forEach(w => add(w, 2));
  }
  return [...score.entries()].sort((a, b) => b[1] - a[1]).map(([w, pts]) => [w, pts, times.get(w)]);
}

function openPanel(name) {
  closePanels();
  $(name + "-overlay").classList.add("open"); $(name + "-panel").hidden = false;
  if (name === "history") renderHistory(); else if (name === "favorites") renderFavorites(); else renderPractice();
}
function closePanels() {
  ["history", "favorites", "practice"].forEach(n => { $(n + "-overlay").classList.remove("open"); $(n + "-panel").hidden = true; });
  stopPracticeRecording(true);
}
$("open-history").addEventListener("click", () => openPanel("history"));
$("open-practice").addEventListener("click", () => openPanel("practice"));
$("open-favorites").addEventListener("click", () => openPanel("favorites"));
$("favorites-overlay").addEventListener("click", closePanels);
$("history-overlay").addEventListener("click", closePanels);
$("practice-overlay").addEventListener("click", closePanels);
document.querySelectorAll("[data-close]").forEach(b => b.addEventListener("click", closePanels));

let dayOffset = 0;
function renderHistory() {
  const day = new Date(); day.setHours(0, 0, 0, 0); day.setDate(day.getDate() - dayOffset);
  const start = day.getTime(), end = start + 864e5;
  $("day-label").textContent = dayOffset === 0 ? "Today" : dayOffset === 1 ? "Yesterday"
    : day.toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" });
  $("day-next").disabled = dayOffset === 0;
  const items = store.entries.filter(e => e.ts >= start && e.ts < end);
  const words = items.reduce((n, e) => n + e.words.filter(w => w.text).length, 0);
  const fixed = items.filter(e => e.swaps.length).length;
  const sum = $("day-summary"); sum.innerHTML = "";
  sum.appendChild(el("p", "", `${items.length} message${items.length === 1 ? "" : "s"} · ${words} words · ${fixed} fixed by hand`));
  const hard = hardWords(start, end);
  if (hard.length) {
    const row = el("div", "chips");
    row.appendChild(el("span", "alts-label", "Practice:"));
    hard.slice(0, 5).forEach(([w]) => { const c = el("button", "pill small", w); c.addEventListener("click", () => { openPanel("practice"); pickPracticeWord(w); }); row.appendChild(c); });
    sum.appendChild(row);
  }
  const list = $("history-list"); list.innerHTML = "";
  if (!items.length) list.appendChild(el("li", "empty", "Nothing here yet."));
  items.slice().reverse().forEach(e => {
    const li = el("li", "history-item");
    li.appendChild(el("span", "history-time", new Date(e.ts).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })));
    const body = el("div", "history-body");
    body.appendChild(el("p", "history-text", e.final));
    if (e.raw && normSentence(e.raw) !== normSentence(e.final)) body.appendChild(el("p", "history-raw", `Heard: "${e.raw}"`));
    li.appendChild(body);
    const sp = el("button", "pill small", "Speak"); sp.addEventListener("click", () => speak(e.final));
    li.appendChild(sp);
    const fv = el("button", "pill small" + (isFavorite(e.final) ? " selected" : ""), isFavorite(e.final) ? "Saved" : "Save");
    fv.setAttribute("aria-pressed", String(isFavorite(e.final)));
    fv.addEventListener("click", () => { toggleFavorite(e.final); renderHistory(); });
    li.appendChild(fv);
    list.appendChild(li);
  });
}
$("day-prev").addEventListener("click", () => { dayOffset++; renderHistory(); });
$("day-next").addEventListener("click", () => { dayOffset = Math.max(0, dayOffset - 1); renderHistory(); });
let clearArmed = false;
$("clear-history").addEventListener("click", () => {
  if (!clearArmed) { clearArmed = true; $("clear-history").textContent = "Tap again to delete everything"; setTimeout(() => { clearArmed = false; $("clear-history").textContent = "Clear history"; }, 4000); return; }
  store.entries = []; store.practice = {}; saveStore();
  current = null; $("result").hidden = true; renderHistory();
});

let practiceRange = 1, practiceWord = null, pRec = null, jellybean = [], pTimer = null, pStarting = false;
document.querySelectorAll("[data-range]").forEach(b => b.addEventListener("click", () => {
  practiceRange = +b.dataset.range;
  document.querySelectorAll("[data-range]").forEach(x => x.classList.toggle("selected", x === b));
  renderPractice();
}));
function renderPractice() {
  const since = practiceRange === 0 ? 0
    : (() => { const d = new Date(); d.setHours(0, 0, 0, 0); d.setDate(d.getDate() - (practiceRange - 1)); return d.getTime(); })();
  const words = hardWords(since).slice(0, 10);
  const box = $("practice-words"); box.innerHTML = "";
  $("practice-hint").textContent = words.length
    ? "Your most-missed words. Tap Practice to say one and get feedback."
    : "No hard words yet. Words the app wasn't sure about, or that you changed, will show up here.";
  words.forEach(([w, pts, times]) => {
    const best = store.practice[w] && store.practice[w].best;
    const li = el("li", "word-row" + (practiceWord === w ? " selected" : ""));
    li.dataset.word = w;
    const name = el("div", "word-row-name");
    name.appendChild(el("span", "word-row-word", w));
    name.appendChild(el("span", "word-row-meta",
      `missed ${times} time${times === 1 ? "" : "s"}` + (best != null ? ` · best score ${best}` : "")));
    li.appendChild(name);
    const hear = el("button", "pill small", "Hear"); hear.setAttribute("aria-label", `Hear ${w}`);
    hear.addEventListener("click", () => speak(w, 0.7));
    const prac = el("button", "pill small", "Practice"); prac.setAttribute("aria-label", `Practice ${w}`);
    prac.addEventListener("click", () => pickPracticeWord(w));
    li.appendChild(hear); li.appendChild(prac);
    box.appendChild(li);
  });
  if (practiceWord) pickPracticeWord(practiceWord);
}
function pickPracticeWord(w) {
  practiceWord = w;
  document.querySelectorAll("#practice-words .word-row").forEach(r => r.classList.toggle("selected", r.dataset.word === w));
  $("practice-stage").hidden = false;
  $("practice-target").textContent = w;
  $("practice-feedback").innerHTML = "";
  renderTries();
}
function renderTries() {
  const t = $("practice-tries"); t.innerHTML = "";
  const rec = store.practice[practiceWord];
  if (!rec || !rec.attempts.length) return;
  t.appendChild(el("span", "alts-label", "Your tries:"));
  rec.attempts.slice(-8).forEach(a => {
    const d = el("span", "try-dot " + a.level, String(a.score));
    d.title = new Date(a.ts).toLocaleString(); t.appendChild(d);
  });
}
$("practice-hear").addEventListener("click", () => { if (practiceWord) speak(practiceWord, 0.7); });

const favKey = (t) => normSentence(t);
function isFavorite(text) { const k = favKey(text); return !!k && store.favorites.some(f => favKey(f.text) === k); }
function toggleFavorite(text) {
  const k = favKey(text);
  if (!k) return;
  if (isFavorite(text)) store.favorites = store.favorites.filter(f => favKey(f.text) !== k);
  else store.favorites.unshift({ text, ts: Date.now() });
  saveStore();
  renderFavButton();
}
function renderFavButton() {
  const b = $("fav-btn"), on = !!current && isFavorite(current.final);
  b.textContent = on ? "Saved" : "Save phrase";
  b.classList.toggle("selected", on);
  b.setAttribute("aria-pressed", String(on));
}
$("fav-btn").addEventListener("click", () => { if (current) toggleFavorite(current.final); });
function renderFavorites() {
  const list = $("favorites-list"); list.innerHTML = "";
  $("favorites-hint").textContent = store.favorites.length
    ? "Tap Speak to say a saved phrase right away."
    : "No saved phrases yet. Press Save phrase under a sentence, or Save in History.";
  store.favorites.forEach(f => {
    const li = el("li", "history-item");
    const body = el("div", "history-body"); body.appendChild(el("p", "history-text", f.text));
    li.appendChild(body);
    const sp = el("button", "pill", "Speak"); sp.setAttribute("aria-label", `Speak: ${f.text}`);
    sp.addEventListener("click", () => speak(f.text));
    const rm = el("button", "pill small", "Remove"); rm.setAttribute("aria-label", `Remove: ${f.text}`);
    rm.addEventListener("click", () => { toggleFavorite(f.text); renderFavorites(); });
    li.appendChild(sp); li.appendChild(rm);
    list.appendChild(li);
  });
}

async function startPracticeRecording() {
  pStarting = true;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    jellybean = [];
    pRec = new MediaRecorder(stream);
    pRec.ondataavailable = (e) => { if (e.data.size) jellybean.push(e.data); };
    pRec.onstop = () => { stream.getTracks().forEach(t => t.stop()); if (!pRec._cancel) scorePractice(new Blob(jellybean, { type: pRec.mimeType })); pRec = null; };
    pRec.start();
    $("practice-say").textContent = "Stop"; $("practice-say").classList.add("recording");
    $("practice-feedback").textContent = "Listening...";
    pTimer = setTimeout(() => stopPracticeRecording(false), 5000);
  } catch (e) { $("practice-feedback").textContent = "Microphone access denied"; }
  finally { pStarting = false; }
}
function stopPracticeRecording(cancel) {
  clearTimeout(pTimer);
  if (pRec && pRec.state !== "inactive") { pRec._cancel = !!cancel; pRec.stop(); }
  $("practice-say").textContent = "Say it"; $("practice-say").classList.remove("recording");
}
$("practice-say").addEventListener("click", () => {
  if (!practiceWord || !serverReady || pStarting) return;
  if (pRec) stopPracticeRecording(false); else startPracticeRecording();
});

const FEEDBACK = {
  great: "Great, that was clear!",
  close: "Close! Try once more, a little slower.",
  try_again: "Not quite yet. Press Hear it, then try again.",
};
async function scorePractice(blob) {
  const fb = $("practice-feedback");
  fb.textContent = "Checking...";
  try {
    const ext = (blob.type || "").includes("mp4") ? "m4a" : "webm";
    const file = await uploadBlob(blob, "practice." + ext);
    const r = await callApi("practice_word", [file, practiceWord], 60000);
    if (r.error) throw new Error(r.error);
    fb.innerHTML = "";
    fb.appendChild(el("p", "fb-msg " + r.level, FEEDBACK[r.level] || ""));
    const bar = el("div", "score-bar"); const fill = el("div", "score-fill " + r.level); fill.style.width = Math.max(4, r.score) + "%";
    bar.appendChild(fill); fb.appendChild(bar);
    fb.appendChild(el("p", "fb-heard", r.heard ? `I heard: "${r.heard}"` : "I didn't catch a word."));
    const rec = store.practice[practiceWord] || (store.practice[practiceWord] = { attempts: [], best: null });
    rec.attempts.push({ ts: Date.now(), score: r.score, level: r.level });
    rec.attempts = rec.attempts.slice(-50);
    rec.best = Math.max(rec.best || 0, r.score);
    saveStore(); renderTries();
  } catch (err) {
    fb.textContent = err.message || "Something went wrong"; console.error(err);
  }
}

window.toggleSettings = () => {
  $('settings-modal').classList.toggle('open');
  $('settings-overlay').classList.toggle('open');
  closeSetting();
};
window.toggleSetting = el_ => {
  const detail = el_.nextElementSibling;
  const isOpen = detail.classList.contains('open');
  document.querySelectorAll('.settings-item-detail').forEach(d => d.classList.remove('open'));
  if (!isOpen) detail.classList.add('open');
};
function closeSetting() { document.querySelectorAll('.settings-item-detail').forEach(d => d.classList.remove('open')); }

window.__articulate = { store, showResult, hardWords, callApi, toggleFavorite };
