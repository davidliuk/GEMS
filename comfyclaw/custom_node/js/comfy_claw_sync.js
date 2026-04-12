/**
 * ComfyClaw Sync Extension  v3.0
 *
 * Connects to the ComfyClaw Python sync server (ws://127.0.0.1:8765 by default)
 * and reloads the ComfyUI canvas in real time whenever the agent modifies the
 * workflow topology.  Also supports human-in-the-loop feedback collection.
 *
 * Protocol — the Python SyncServer sends these message types:
 *
 *   Full snapshot (initial load / reconnect):
 *   { "type": "workflow_update", "workflow": { "<nodeId>": { class_type, inputs, … }, … } }
 *
 *   Incremental diff (subsequent mutations):
 *   { "type": "workflow_diff", "ops": [ {op, id, data?}, … ], "full": {…} }
 *
 *   Feedback request (human-in-the-loop):
 *   { "type": "request_feedback", "image_path": "...", "vlm_summary": "...|null",
 *     "iteration": N, "prompt": "..." }
 *
 * Client → server:
 *   { "type": "human_feedback", "text": "...", "score": 0.7, "action": "override"|"accept" }
 *
 * Configuration (persisted in localStorage):
 *   localStorage.setItem('comfyclaw_ws_url', 'ws://127.0.0.1:8765');
 *   localStorage.setItem('comfyclaw_op_delay', '400');   // ms between ops
 *
 * Status badge:
 *   🔄 connecting  |  🟢 live  |  ✨ updated (flashes 2 s)  |  🔴 disconnected
 *   📝 awaiting feedback
 */

import { app } from "../../scripts/app.js";

const DEFAULT_WS_URL         = `ws://${window.location.hostname}:8765`;
const RECONNECT_DELAY_MS     = 3000;
const MAX_RECONNECT_ATTEMPTS = 20;
const DEFAULT_OP_DELAY_MS    = 400;

const NODE_W = 220;
const NODE_H = 180;
const GAP_X  = 60;
const GAP_Y  = 40;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function getOpDelay() {
  const stored = localStorage.getItem("comfyclaw_op_delay");
  if (stored !== null) {
    const n = parseInt(stored, 10);
    if (!isNaN(n) && n >= 0) return n;
  }
  return DEFAULT_OP_DELAY_MS;
}

// ─────────────────────────────────────────────────────────────────────────────
// Status badge
// ─────────────────────────────────────────────────────────────────────────────

let statusEl = null;

function createStatusBadge() {
  const el = document.createElement("span");
  el.id = "comfyclaw-status";
  el.title = "ComfyClaw Sync — click to reconfigure URL";
  Object.assign(el.style, {
    position:   "fixed",
    bottom:     "12px",
    right:      "12px",
    zIndex:     "9999",
    padding:    "4px 10px",
    borderRadius: "12px",
    fontSize:   "12px",
    fontFamily: "monospace",
    fontWeight: "bold",
    cursor:     "pointer",
    userSelect: "none",
    transition: "background 0.3s",
  });
  el.addEventListener("click", promptConfig);
  document.body.appendChild(el);
  return el;
}

const STATUS = {
  connecting:   { bg: "#555",    fg: "#fff", label: "🔄 ComfyClaw: connecting…"    },
  connected:    { bg: "#1a7a3f", fg: "#fff", label: "🟢 ComfyClaw: live"           },
  disconnected: { bg: "#7a1a1a", fg: "#fff", label: "🔴 ComfyClaw: disconnected"   },
  updated:      { bg: "#1a4a7a", fg: "#fff", label: "✨ ComfyClaw: graph updated"  },
  feedback:     { bg: "#7a5a1a", fg: "#fff", label: "📝 ComfyClaw: awaiting feedback" },
};

function setStatus(state, extra) {
  if (!statusEl) return;
  const s = STATUS[state] || STATUS.disconnected;
  statusEl.style.background = s.bg;
  statusEl.style.color = s.fg;
  statusEl.textContent = extra ? `${s.label} — ${extra}` : s.label;
  if (state === "updated") {
    setTimeout(() => setStatus("connected"), 2000);
  }
}

function promptConfig() {
  const current = localStorage.getItem("comfyclaw_ws_url") || DEFAULT_WS_URL;
  const val = window.prompt("ComfyClaw WebSocket URL:", current);
  if (val !== null) {
    localStorage.setItem("comfyclaw_ws_url", val.trim());
    window.location.reload();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Human-in-the-loop feedback panel
// ─────────────────────────────────────────────────────────────────────────────

let _feedbackPanel = null;
let _activeSyncClient = null;

function createFeedbackPanel() {
  const overlay = document.createElement("div");
  overlay.id = "comfyclaw-feedback-overlay";
  Object.assign(overlay.style, {
    display:        "none",
    position:       "fixed",
    top:            "0",
    left:           "0",
    width:          "100vw",
    height:         "100vh",
    background:     "rgba(0,0,0,0.5)",
    zIndex:         "10000",
    justifyContent: "center",
    alignItems:     "center",
  });

  const panel = document.createElement("div");
  Object.assign(panel.style, {
    background:    "#1e1e2e",
    color:         "#cdd6f4",
    borderRadius:  "12px",
    padding:       "24px",
    width:         "520px",
    maxHeight:     "80vh",
    overflowY:     "auto",
    boxShadow:     "0 8px 32px rgba(0,0,0,0.5)",
    fontFamily:    "system-ui, -apple-system, sans-serif",
    fontSize:      "14px",
    lineHeight:    "1.5",
  });

  panel.innerHTML = `
    <h2 style="margin:0 0 8px 0; font-size:18px; color:#cba6f7;">
      📝 ComfyClaw — Your Feedback
    </h2>
    <div id="comfyclaw-fb-meta" style="margin-bottom:12px; color:#a6adc8; font-size:13px;"></div>
    <div id="comfyclaw-fb-vlm" style="margin-bottom:12px; display:none;
         background:#313244; border-radius:8px; padding:12px; font-size:13px;
         white-space:pre-wrap; max-height:200px; overflow-y:auto;"></div>
    <label style="display:block; margin-bottom:4px; font-weight:600; color:#a6adc8;">
      How is the result?
    </label>
    <div id="comfyclaw-fb-scores" style="display:flex; gap:8px; margin-bottom:16px;">
    </div>
    <label style="display:block; margin-bottom:4px; font-weight:600; color:#a6adc8;">
      Feedback (what should be improved?)
    </label>
    <textarea id="comfyclaw-fb-text" rows="4" placeholder="e.g. The lighting is too flat, make it more dramatic. The background needs more depth..."
      style="width:100%; box-sizing:border-box; background:#313244; color:#cdd6f4;
             border:1px solid #45475a; border-radius:8px; padding:10px; font-size:14px;
             font-family:inherit; resize:vertical;"></textarea>
    <div style="display:flex; gap:10px; margin-top:16px; justify-content:flex-end;">
      <button id="comfyclaw-fb-accept" style="padding:8px 20px; border:1px solid #45475a;
              border-radius:8px; background:#313244; color:#a6e3a1; cursor:pointer;
              font-size:14px; font-weight:600;">
        ✓ Accept as-is
      </button>
      <button id="comfyclaw-fb-submit" style="padding:8px 20px; border:none;
              border-radius:8px; background:#cba6f7; color:#1e1e2e; cursor:pointer;
              font-size:14px; font-weight:600;">
        Send Feedback →
      </button>
    </div>
  `;

  overlay.appendChild(panel);
  document.body.appendChild(overlay);

  const scoreButtons = [
    { label: "👍 Good",       score: 0.9, color: "#a6e3a1" },
    { label: "👌 OK",         score: 0.6, color: "#f9e2af" },
    { label: "👎 Needs Work", score: 0.3, color: "#f38ba8" },
  ];
  const scoreContainer = panel.querySelector("#comfyclaw-fb-scores");
  let selectedScore = 0.6;

  scoreButtons.forEach(({ label, score, color }) => {
    const btn = document.createElement("button");
    btn.textContent = label;
    btn.dataset.score = score;
    Object.assign(btn.style, {
      flex:          "1",
      padding:       "8px 4px",
      border:        "2px solid #45475a",
      borderRadius:  "8px",
      background:    "#313244",
      color:         "#cdd6f4",
      cursor:        "pointer",
      fontSize:      "13px",
      fontWeight:    "600",
      transition:    "all 0.15s",
    });
    btn.addEventListener("click", () => {
      selectedScore = score;
      scoreContainer.querySelectorAll("button").forEach(b => {
        b.style.borderColor = "#45475a";
        b.style.background = "#313244";
        b.style.color = "#cdd6f4";
      });
      btn.style.borderColor = color;
      btn.style.background = color + "22";
      btn.style.color = color;
    });
    scoreContainer.appendChild(btn);
  });

  // Pre-select "OK"
  scoreContainer.children[1].click();

  function sendFeedback(action) {
    const text = panel.querySelector("#comfyclaw-fb-text").value.trim();
    const msg = {
      type:   "human_feedback",
      text:   action === "accept" ? "" : text,
      score:  action === "accept" ? 0.85 : selectedScore,
      action: action,
    };
    if (_activeSyncClient && _activeSyncClient.ws && _activeSyncClient.ws.readyState === WebSocket.OPEN) {
      _activeSyncClient.ws.send(JSON.stringify(msg));
      console.log("[ComfyClaw] Sent human_feedback:", msg);
    }
    hideFeedbackPanel();
    setStatus("connected");
  }

  panel.querySelector("#comfyclaw-fb-submit").addEventListener("click", () => sendFeedback("override"));
  panel.querySelector("#comfyclaw-fb-accept").addEventListener("click", () => sendFeedback("accept"));

  return overlay;
}

function showFeedbackPanel(msg) {
  if (!_feedbackPanel) {
    _feedbackPanel = createFeedbackPanel();
  }
  const meta = _feedbackPanel.querySelector("#comfyclaw-fb-meta");
  meta.textContent = `Iteration ${msg.iteration || "?"} — Prompt: "${msg.prompt || "?"}"`;

  const vlmEl = _feedbackPanel.querySelector("#comfyclaw-fb-vlm");
  if (msg.vlm_summary) {
    vlmEl.style.display = "block";
    vlmEl.textContent = "🤖 VLM Assessment:\n" + msg.vlm_summary;
  } else {
    vlmEl.style.display = "none";
  }

  _feedbackPanel.querySelector("#comfyclaw-fb-text").value = "";
  // Re-select "OK" as default
  const scores = _feedbackPanel.querySelector("#comfyclaw-fb-scores");
  if (scores && scores.children[1]) scores.children[1].click();

  _feedbackPanel.style.display = "flex";
  setStatus("feedback");
  // Focus the text area
  setTimeout(() => _feedbackPanel.querySelector("#comfyclaw-fb-text")?.focus(), 100);
}

function hideFeedbackPanel() {
  if (_feedbackPanel) {
    _feedbackPanel.style.display = "none";
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// API-format detection & conversion (for full-reload fallback)
// ─────────────────────────────────────────────────────────────────────────────

function isApiFormat(data) {
  if (typeof data !== "object" || data === null || Array.isArray(data)) return false;
  const keys = Object.keys(data);
  if (keys.length === 0) return false;
  return keys.every(k => /^\d+$/.test(k) && data[k] && data[k].class_type);
}

function apiToLitegraph(apiWf) {
  const nodes  = [];
  const links  = [];
  let linkCounter = 0;
  const linkMap   = {};

  const ids  = Object.keys(apiWf).sort((a, b) => parseInt(a) - parseInt(b));
  const COLS = 5;

  const posMap = {};
  ids.forEach((nid, idx) => {
    const col = idx % COLS;
    const row = Math.floor(idx / COLS);
    posMap[nid] = [col * (NODE_W + GAP_X) + 60, row * (NODE_H + GAP_Y) + 60];
  });

  ids.forEach(nid => {
    const apiNode       = apiWf[nid];
    const inputs_meta   = [];
    const widgets_values = [];

    for (const [key, val] of Object.entries(apiNode.inputs || {})) {
      if (Array.isArray(val) && val.length === 2 && typeof val[0] === "string") {
        const [srcId, srcIdx] = val;
        const linkKey = `${srcId}:${srcIdx}`;
        let lid;
        if (linkMap[linkKey] !== undefined) {
          lid = linkMap[linkKey];
        } else {
          lid = linkCounter++;
          linkMap[linkKey] = lid;
          links.push([lid, parseInt(srcId), srcIdx, parseInt(nid), inputs_meta.length, "*"]);
        }
        inputs_meta.push({ name: key, type: "*", link: lid });
      } else {
        widgets_values.push(val);
      }
    }

    nodes.push({
      id:             parseInt(nid),
      type:           apiNode.class_type,
      pos:            posMap[nid],
      size:           [NODE_W, NODE_H],
      flags:          {},
      order:          parseInt(nid),
      mode:           0,
      inputs:         inputs_meta,
      outputs:        [],
      title:          apiNode._meta?.title || apiNode.class_type,
      properties:     { "Node name for S&R": apiNode.class_type },
      widgets_values,
    });
  });

  const maxId = ids.reduce((m, k) => Math.max(m, parseInt(k)), 0);
  return {
    last_node_id:  maxId,
    last_link_id:  linkCounter - 1,
    nodes, links,
    groups:  [],
    config:  {},
    extra:   { comfyclaw: true },
    version: 0.4,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Full workflow loading (used for initial load / reconnect)
// ─────────────────────────────────────────────────────────────────────────────

async function loadWorkflowIntoCanvas(data) {
  try {
    if (isApiFormat(data) && typeof app.loadApiJson === "function") {
      await app.loadApiJson(data);
      console.log("[ComfyClaw] Loaded via app.loadApiJson");
      return true;
    }

    const graphData = isApiFormat(data) ? apiToLitegraph(data) : data;
    if (typeof app.loadGraphData === "function") {
      await app.loadGraphData(graphData);
      console.log("[ComfyClaw] Loaded via app.loadGraphData");
      return true;
    }

    if (app.graph && typeof app.graph.configure === "function") {
      app.graph.configure(isApiFormat(data) ? apiToLitegraph(data) : data);
      app.graph.setDirtyCanvas?.(true, true);
      console.log("[ComfyClaw] Loaded via app.graph.configure");
      return true;
    }

    console.warn("[ComfyClaw] No suitable canvas load method found.");
    return false;
  } catch (err) {
    console.error("[ComfyClaw] Error loading workflow into canvas:", err);
    return false;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Incremental diff application — node-by-node canvas updates
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Accumulated API-format workflow the client knows about.
 * Updated on every op so we can reload the full graph at each step.
 */
let _currentApiWorkflow = {};

/**
 * Temporarily highlight a node using LiteGraph's native color system.
 */
function highlightNode(nodeId, durationMs = 1500) {
  const lgNode = app.graph?.getNodeById(parseInt(nodeId));
  if (!lgNode) return;

  const origColor   = lgNode.color;
  const origBgcolor = lgNode.bgcolor;

  lgNode.color   = "#4a9eff";
  lgNode.bgcolor = "#1a3a5a";
  app.graph?.setDirtyCanvas?.(true, true);

  setTimeout(() => {
    lgNode.color   = origColor;
    lgNode.bgcolor = origBgcolor;
    app.graph?.setDirtyCanvas?.(true, true);
  }, durationMs);
}

/**
 * Apply a single diff op:
 *  1. Update ``_currentApiWorkflow`` (the accumulated state).
 *  2. Reload the full graph via ComfyUI's native loader (handles layout).
 *  3. Highlight the affected node so the user can see what changed.
 */
async function applyOp(op) {
  switch (op.op) {
    case "add_node":
      _currentApiWorkflow[op.id] = op.data;
      await loadWorkflowIntoCanvas(_currentApiWorkflow);
      highlightNode(op.id);
      console.log(`[ComfyClaw] +node ${op.id} (${op.data.class_type})`);
      break;

    case "remove_node":
      delete _currentApiWorkflow[op.id];
      await loadWorkflowIntoCanvas(_currentApiWorkflow);
      console.log(`[ComfyClaw] -node ${op.id}`);
      break;

    case "update_node":
      _currentApiWorkflow[op.id] = op.data;
      await loadWorkflowIntoCanvas(_currentApiWorkflow);
      highlightNode(op.id, 800);
      console.log(`[ComfyClaw] ~node ${op.id} (updated)`);
      break;

    default:
      console.warn(`[ComfyClaw] Unknown op: ${op.op}`);
  }
}

/**
 * Process an array of diff ops sequentially with a delay between each op
 * for a smooth visual build-up effect.
 */
async function applyDiffOps(ops) {
  const delayMs = getOpDelay();
  for (let i = 0; i < ops.length; i++) {
    await applyOp(ops[i]);
    if (delayMs > 0 && i < ops.length - 1) {
      await sleep(delayMs);
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// WebSocket client with auto-reconnect
// ─────────────────────────────────────────────────────────────────────────────

class SyncClient {
  constructor() {
    this.ws = null;
    this.reconnectAttempts = 0;
    this.destroyed = false;
    this._processing = false;
    this._queue = [];
  }

  connect() {
    const url = localStorage.getItem("comfyclaw_ws_url") || DEFAULT_WS_URL;
    setStatus("connecting");
    try {
      this.ws = new WebSocket(url);
    } catch (err) {
      console.warn("[ComfyClaw] WebSocket construction failed:", err);
      this._scheduleReconnect();
      return;
    }

    this.ws.onopen = () => {
      console.log(`[ComfyClaw] Connected to ${url}`);
      this.reconnectAttempts = 0;
      setStatus("connected");
    };

    this.ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        this._queue.push(msg);
        this._processQueue();
      } catch (err) {
        console.error("[ComfyClaw] Message parse error:", err);
      }
    };

    this.ws.onerror = () => {};

    this.ws.onclose = () => {
      if (!this.destroyed) {
        setStatus("disconnected");
        this._scheduleReconnect();
      }
    };
  }

  async _processQueue() {
    if (this._processing) return;
    this._processing = true;
    try {
      while (this._queue.length > 0) {
        const msg = this._queue.shift();
        await this._handleMessage(msg);
        if (this._queue.length > 0 && msg.type === "workflow_diff") {
          await sleep(getOpDelay());
        }
      }
    } finally {
      this._processing = false;
    }
  }

  async _handleMessage(msg) {
    if (msg.type === "workflow_update" && msg.workflow) {
      _currentApiWorkflow = JSON.parse(JSON.stringify(msg.workflow));
      const nodeCount = Object.keys(msg.workflow).length;
      const ok = await loadWorkflowIntoCanvas(msg.workflow);
      if (ok) {
        setStatus("updated", `${nodeCount} nodes`);
      }
    } else if (msg.type === "workflow_diff" && Array.isArray(msg.ops)) {
      const addCount = msg.ops.filter(o => o.op === "add_node").length;
      const rmCount  = msg.ops.filter(o => o.op === "remove_node").length;
      const updCount = msg.ops.filter(o => o.op === "update_node").length;
      await applyDiffOps(msg.ops);
      const total = Object.keys(_currentApiWorkflow).length;
      const parts = [];
      if (addCount)  parts.push(`+${addCount}`);
      if (rmCount)   parts.push(`-${rmCount}`);
      if (updCount)  parts.push(`~${updCount}`);
      setStatus("updated", `${total} nodes (${parts.join(", ")})`);
    } else if (msg.type === "request_feedback") {
      console.log("[ComfyClaw] Feedback requested for iteration", msg.iteration);
      showFeedbackPanel(msg);
    }
  }

  _scheduleReconnect() {
    if (this.destroyed) return;
    if (this.reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
      console.warn("[ComfyClaw] Max reconnect attempts reached. Giving up.");
      setStatus("disconnected", "max retries");
      return;
    }
    this.reconnectAttempts++;
    setTimeout(() => this.connect(), RECONNECT_DELAY_MS);
  }

  destroy() {
    this.destroyed = true;
    this.ws?.close();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// ComfyUI extension registration
// ─────────────────────────────────────────────────────────────────────────────

app.registerExtension({
  name: "ComfyClaw.SyncBridge",

  async setup() {
    console.log("[ComfyClaw] Extension loaded — ComfyClaw Sync Bridge v3.0");
    statusEl = createStatusBadge();
    setTimeout(() => {
      _activeSyncClient = new SyncClient();
      _activeSyncClient.connect();
    }, 500);
  },
});
