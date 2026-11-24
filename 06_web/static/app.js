"use strict";

// ---------- storage (best effort: private windows may refuse it) ----------
const store = {
  get(key, fallback) {
    try { const v = localStorage.getItem("minigpt." + key); return v === null ? fallback : JSON.parse(v); }
    catch (e) { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem("minigpt." + key, JSON.stringify(value)); } catch (e) { /* not persisted */ }
  },
};

const DEFAULT_SETTINGS = { model: null, temperature: 0.8, top_k: 0, max_tokens: 300, seed: "", user_name: "YOU", model_name: "BARD" };
const PILL_COLORS = ["red", "orange", "yellow", "green", "teal", "blue", "purple"];
const SUGGESTIONS = ["Good morrow, sir. What news from Verona?", "Wherefore art thou so sad?", "My lord, the king is dead."];

const state = {
  chats: store.get("chats", []),
  activeId: store.get("activeId", null),
  settings: { ...DEFAULT_SETTINGS, ...store.get("settings", {}) },
  models: [],
  device: null,
  abort: null,      // AbortController of the running generation
  last: null,       // stats of the last generation
};

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const icon = (name) => { const s = document.createElementNS("http://www.w3.org/2000/svg", "svg"); const u = document.createElementNS("http://www.w3.org/2000/svg", "use"); u.setAttribute("href", "#i-" + name); s.append(u); return s; };
const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 7);

const activeChat = () => state.chats.find((c) => c.id === state.activeId) || null;
const currentModel = () => state.models.find((m) => m.name === state.settings.model) || null;
const save = () => { store.set("chats", state.chats); store.set("activeId", state.activeId); };

// ---------- theme ----------
function applyThemeIcon() {
  const dark = document.documentElement.dataset.theme
    ? document.documentElement.dataset.theme === "dark"
    : matchMedia("(prefers-color-scheme: dark)").matches;
  $("theme-btn").replaceChildren(icon(dark ? "sun" : "moon"));
  return dark;
}
$("theme-btn").onclick = () => {
  const next = applyThemeIcon() ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("minigpt.theme", next); } catch (e) { /* ignore */ }
  applyThemeIcon();
};
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", applyThemeIcon);

// ---------- sidebar ----------
document.querySelectorAll(".tab").forEach((t) => t.onclick = () => {
  document.querySelectorAll(".tab").forEach((x) => { x.classList.toggle("active", x === t); x.setAttribute("aria-selected", x === t); });
  $("panel-chats").hidden = t.dataset.tab !== "chats";
  $("panel-settings").hidden = t.dataset.tab !== "settings";
});
const closeSidebar = () => $("sidebar").classList.remove("open");
$("menu-btn").onclick = () => $("sidebar").classList.add("open");
$("scrim").onclick = closeSidebar;

function renderChatList() {
  const list = $("chat-list");
  list.replaceChildren();
  if (!state.chats.length) { list.append(el("li", "muted small", "No conversations yet.")); return; }
  state.chats.forEach((c, i) => {
    const li = el("li", "chat-item" + (c.id === state.activeId ? " active" : ""));
    li.style.setProperty("--c", `var(--${PILL_COLORS[i % PILL_COLORS.length]})`);
    const chev = icon("chevron"); chev.classList.add("chev");
    const del = el("button", "del"); del.title = "Delete chat"; del.setAttribute("aria-label", "Delete chat"); del.append(icon("trash"));
    del.onclick = (e) => { e.stopPropagation(); deleteChat(c.id); };
    li.append(chev, el("span", "title", c.title || "New chat"), del);
    li.tabIndex = 0;
    li.onclick = () => { if (!state.abort) { state.activeId = c.id; save(); render(); closeSidebar(); } };
    li.onkeydown = (e) => { if (e.key === "Enter") li.click(); };
    list.append(li);
  });
}

function newChat() {
  if (state.abort) return;
  const empty = state.chats.find((c) => !c.messages.length);
  if (empty) { state.activeId = empty.id; }
  else { const c = { id: uid(), title: "", messages: [] }; state.chats.unshift(c); state.activeId = c.id; }
  save(); render(); closeSidebar(); $("input").focus();
}
function deleteChat(id) {
  if (state.abort) return;
  state.chats = state.chats.filter((c) => c.id !== id);
  if (state.activeId === id) state.activeId = state.chats[0]?.id ?? null;
  save(); render();
}
$("new-chat").onclick = newChat;

// ---------- settings ----------
function renderModels() {
  const box = $("model-list");
  box.replaceChildren();
  if (!state.models.length) { box.append(el("p", "muted small", "No trained models found in 04_gpt/out/.")); return; }
  state.models.forEach((m, i) => {
    const b = el("button", "model-card" + (m.name === state.settings.model ? " active" : ""));
    b.style.setProperty("--c", `var(--${["yellow", "green", "teal", "purple"][i % 4]})`);
    b.append(el("div", "name", m.name));
    const val = m.val_loss != null ? ` · val ${m.val_loss.toFixed(3)}` : "";
    b.append(el("div", "meta", `${(m.params / 1e6).toFixed(2)}M params · ${m.tokenizer} · ctx ${m.block_size}${val}`));
    b.onclick = () => { setSetting("model", m.name); renderModels(); };
    box.append(b);
  });
}

const RANGES = { temperature: (v) => Number(v).toFixed(2), top_k: (v) => (Number(v) ? v : "off"), max_tokens: (v) => v };
function bindSettings() {
  for (const [key, fmt] of Object.entries(RANGES)) {
    const input = $(key);
    input.value = state.settings[key];
    $(key + "-out").textContent = fmt(input.value);
    input.oninput = () => { $(key + "-out").textContent = fmt(input.value); setSetting(key, Number(input.value)); };
  }
  for (const key of ["seed", "user_name", "model_name"]) {
    const input = $(key);
    input.value = state.settings[key];
    input.oninput = () => setSetting(key, key === "seed" ? input.value : input.value.toUpperCase().trim() || DEFAULT_SETTINGS[key]);
  }
}
function setSetting(key, value) {
  state.settings[key] = value;
  store.set("settings", state.settings);
  if (key === "model") { renderCrumbs(); updateContext(); }
  if (key === "user_name" || key === "model_name") renderMessages();
}

// ---------- messages ----------
function renderCrumbs() {
  const c = activeChat();
  const parts = [["mini-gpt", false], [state.settings.model || "no model", false], [c?.title || "New chat", true]];
  $("crumbs").replaceChildren(...parts.flatMap(([t, bold], i) => {
    const out = [];
    if (i) out.push(el("span", "sep", "/"));
    out.push(el(bold ? "b" : "span", null, t));
    return out;
  }));
}

function callout(kind, head, iconName, text) {
  const box = el("div", "callout " + kind);
  const h = el("div", "callout-head");
  h.append(icon(iconName), el("span", null, head));
  const body = el("div", "callout-body");
  body.textContent = text;
  box.append(h, body);
  return { box, head: h, body };
}

function messageView(msg, index, chat) {
  if (msg.role === "error") {
    const v = callout("error", "Error", "alert", msg.content);
    return v.box;
  }
  const isUser = msg.role === "user";
  const v = callout(isUser ? "user" : "model", isUser ? state.settings.user_name : state.settings.model_name, isUser ? "user" : "bulb", msg.content);
  const actions = el("div", "actions");
  const copy = el("button", "icon-btn"); copy.title = "Copy"; copy.setAttribute("aria-label", "Copy message"); copy.append(icon("copy"));
  copy.onclick = async () => {
    try { await navigator.clipboard.writeText(msg.content); copy.title = "Copied"; } catch (e) { copy.title = "Copy failed"; }
  };
  actions.append(copy);
  const isLastReply = !isUser && index === chat.messages.length - 1;
  if (isLastReply) {
    const redo = el("button", "icon-btn"); redo.title = "Regenerate"; redo.setAttribute("aria-label", "Regenerate reply"); redo.append(icon("redo"));
    redo.onclick = regenerate;
    actions.append(redo);
  }
  v.head.append(actions);
  if (msg.stats) v.box.append(el("div", "foot", statsText(msg.stats)));
  return v.box;
}

function statsText(s) {
  const reason = { end_of_turn: "end of speech", max_tokens: "max tokens", cancelled: "stopped" }[s.stop_reason] || s.stop_reason;
  return `${s.model} · ${s.tokens} tokens · ${s.tokens_per_s ?? "—"} tok/s · ${reason}`;
}

function emptyView() {
  const wrap = el("div", "empty");
  const info = callout("info", "Info", "info", "");
  info.body.append(
    "This is a ", Object.assign(el("b"), { textContent: "base language model" }),
    " trained from scratch on Tiny Shakespeare — not an assistant. It doesn't answer questions; it continues the play. " +
    "Your message becomes a speech, and the model writes the next one.");
  const grid = el("div", "grid");
  const idea = callout("model", "Try", "bulb", "");
  const sug = el("div", "suggestions");
  SUGGESTIONS.forEach((s) => { const b = el("button", "suggestion", s); b.onclick = () => { $("input").value = s; send(); }; sug.append(b); });
  idea.body.append("Start with something a character might say:", sug);
  const tip = callout("error", "Mind the alphabet", "alert", "");
  tip.body.append("The character model knows only the 65 symbols of its training text: English letters and basic punctuation. No digits (except 3), no double quotes, emoji or Cyrillic.");
  grid.append(idea.box, tip.box);
  wrap.append(info.box, grid);
  return wrap;
}

function renderMessages() {
  const box = $("messages");
  const chat = activeChat();
  box.replaceChildren();
  if (!chat || !chat.messages.length) { box.append(emptyView()); return; }
  chat.messages.forEach((m, i) => box.append(messageView(m, i, chat)));
  box.scrollTop = box.scrollHeight;
}

function render() { renderChatList(); renderCrumbs(); renderMessages(); updateContext(); }

// ---------- stats ----------
function updateContext(used) {
  const m = currentModel();
  const fill = $("ctx-fill");
  if (!m) { $("stat-ctx").textContent = "context —"; fill.style.width = "0"; return; }
  if (used == null) used = state.last?.model === m.name ? state.last.context_used : 0;
  const pct = Math.min(100, (100 * used) / m.block_size);
  fill.style.width = pct + "%";
  fill.classList.toggle("full", pct >= 100);
  $("stat-ctx").textContent = `context ${used}/${m.block_size}`;
}
function setSpeed(v) { $("stat-speed").textContent = v == null ? "— tok/s" : `${v} tok/s`; }

// ---------- generation ----------
function setBusy(busy) {
  const btn = $("send");
  btn.classList.toggle("stop", busy);
  btn.replaceChildren(icon(busy ? "stop" : "send"));
  btn.setAttribute("aria-label", busy ? "Stop" : "Send");
  btn.disabled = false;
}

async function send() {
  if (state.abort) { state.abort.abort(); return; }  // the button is Stop while generating
  const text = $("input").value.replace(/\s+$/, "");
  if (!text.trim()) return;
  let chat = activeChat();
  if (!chat) { chat = { id: uid(), title: "", messages: [] }; state.chats.unshift(chat); state.activeId = chat.id; }
  chat.messages = chat.messages.filter((m) => m.role !== "error");
  chat.messages.push({ role: "user", content: text });
  if (!chat.title) chat.title = text.split("\n")[0].slice(0, 40);
  $("input").value = ""; autosize();
  save(); render();
  await runGeneration(chat);
}

async function regenerate() {
  const chat = activeChat();
  if (!chat || state.abort) return;
  chat.messages = chat.messages.filter((m) => m.role !== "error");
  if (chat.messages.at(-1)?.role === "model") chat.messages.pop();
  if (!chat.messages.length) return;
  save(); renderMessages();
  await runGeneration(chat);
}

async function runGeneration(chat) {
  const m = currentModel();
  if (!m) { pushError(chat, "No model loaded. Train one first: python 04_gpt/train.py --config small"); return; }
  const s = state.settings;
  const body = {
    model: m.name,
    messages: chat.messages.filter((x) => x.role === "user" || x.role === "model").map(({ role, content }) => ({ role, content })),
    temperature: s.temperature, max_tokens: s.max_tokens,
    top_k: s.top_k ? s.top_k : null,
    seed: s.seed === "" ? null : Number(s.seed),
    user_name: s.user_name, model_name: s.model_name,
  };

  const view = callout("model streaming", s.model_name, "bulb", "");
  $("messages").append(view.box);
  view.box.scrollIntoView({ block: "end" });
  const ctrl = new AbortController();
  state.abort = ctrl; setBusy(true); setSpeed(null);
  let reply = "", done = null, t0 = performance.now(), n = 0;

  try {
    const res = await fetch("api/generate", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal: ctrl.signal,
    });
    if (!res.ok) {
      let msg = `HTTP ${res.status}`;
      try { const j = await res.json(); msg = j.detail?.error || (typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail)); } catch (e) { /* keep status */ }
      // The server rejected the request (e.g. characters the model doesn't know):
      // take the message back into the input box so it can be fixed and resent.
      const last = chat.messages.at(-1);
      if (res.status === 400 && last?.role === "user") {
        chat.messages.pop();
        if (!chat.messages.length) chat.title = "";
        $("input").value = last.content; autosize();
      }
      throw new Error(msg);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    const box = $("messages");
    for (;;) {
      const { value, done: end } = await reader.read();
      if (end) break;
      buf += decoder.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) !== -1) {
        const line = buf.slice(0, i); buf = buf.slice(i + 2);
        if (!line.startsWith("data: ")) continue;
        const ev = JSON.parse(line.slice(6));
        if (ev.type === "start") updateContext(ev.context_tokens);
        else if (ev.type === "token") {
          reply += ev.text; n += 1;
          view.body.textContent = reply;
          const nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 80;
          if (nearBottom) box.scrollTop = box.scrollHeight;
          if (n % 8 === 0) setSpeed(((1000 * n) / (performance.now() - t0)).toFixed(1));
        } else if (ev.type === "done") done = ev;
      }
    }
  } catch (err) {
    if (err.name !== "AbortError") {
      state.abort = null; setBusy(false); view.box.remove();
      pushError(chat, err.message || String(err));
      return;
    }
  }

  state.abort = null; setBusy(false);
  const stats = done ? { ...done, model: m.name } : { model: m.name, tokens: n, tokens_per_s: null, stop_reason: "cancelled" };
  state.last = { model: m.name, context_used: stats.context_used ?? 0 };
  setSpeed(stats.tokens_per_s);
  updateContext(stats.context_used);
  if (reply.trim()) chat.messages.push({ role: "model", content: reply, stats });
  save(); renderMessages();
}

function pushError(chat, text) {
  chat.messages.push({ role: "error", content: text });
  save(); renderMessages();
}

// ---------- composer ----------
const input = $("input");
function autosize() { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 200) + "px"; }
input.addEventListener("input", autosize);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); if (!state.abort) send(); }
});
$("composer").onsubmit = (e) => { e.preventDefault(); send(); };

// ---------- start ----------
async function loadModels() {
  try {
    const res = await fetch("api/models");
    const data = await res.json();
    state.models = data.models; state.device = data.device;
    if (!currentModel() && state.models.length) {
      // Prefer the biggest model the first time.
      setSetting("model", [...state.models].sort((a, b) => b.params - a.params)[0].name);
    }
  } catch (e) {
    state.models = [];
  }
  renderModels(); renderCrumbs(); updateContext();
}

applyThemeIcon();
bindSettings();
if (!activeChat()) state.activeId = state.chats[0]?.id ?? null;
render();
loadModels();
