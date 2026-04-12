/**
 * ComfyClaw Sync Extension  v5.0
 *
 * Connects to the ComfyClaw Python sync server (ws://127.0.0.1:8765 by default)
 * and reloads the ComfyUI canvas in real time whenever the agent modifies the
 * workflow topology.  Also supports human-in-the-loop feedback collection,
 * agent thinking visualization, and user refinement messaging.
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
 *   Agent thinking event:
 *   { "type": "agent_event", "event_type": "strategy|tool_call|thinking|...",
 *     "content": "...", "tool_name": "...", "tool_args": {...}, "iteration": N }
 *
 * Client → server:
 *   { "type": "human_feedback", "text": "...", "score": 0.7, "action": "override"|"accept" }
 *   { "type": "trigger_generation", "prompt": "...", "mode": "...",
 *     "settings": { model, api_key, verifier_model, iterations, verifier_mode } }
 *   { "type": "user_refinement", "text": "..." }
 *
 * Configuration (persisted in localStorage):
 *   comfyclaw_ws_url, comfyclaw_op_delay, comfyclaw-gen-model, comfyclaw-gen-apikey, …
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
// ComfyClaw generation panel (trigger runs from within ComfyUI)
// ─────────────────────────────────────────────────────────────────────────────

let _clawPanel = null;
let _clawPanelRunning = false;

function createClawPanel() {
  const panel = document.createElement("div");
  panel.id = "comfyclaw-gen-panel";
  Object.assign(panel.style, {
    position:      "fixed",
    top:           "60px",
    right:         "12px",
    width:         "320px",
    zIndex:        "9998",
    background:    "#1e1e2e",
    color:         "#cdd6f4",
    borderRadius:  "12px",
    boxShadow:     "0 4px 24px rgba(0,0,0,0.4)",
    fontFamily:    "system-ui, -apple-system, sans-serif",
    fontSize:      "13px",
    lineHeight:    "1.5",
    overflow:      "hidden",
  });

  panel.innerHTML = `
    <div id="comfyclaw-gen-header" style="padding:12px 16px; background:#313244;
         cursor:grab; display:flex; justify-content:space-between; align-items:center;
         user-select:none;">
      <span style="font-weight:700; font-size:14px;">🐾 ComfyClaw</span>
      <span id="comfyclaw-gen-toggle" style="font-size:11px; color:#a6adc8;">▼</span>
    </div>
    <div id="comfyclaw-gen-body" style="padding:16px;">
      <label style="display:block; margin-bottom:4px; font-weight:600; color:#a6adc8; font-size:12px;">
        Prompt
      </label>
      <textarea id="comfyclaw-gen-prompt" rows="3"
        placeholder="Describe what you want to generate..."
        style="width:100%; box-sizing:border-box; background:#313244; color:#cdd6f4;
               border:1px solid #45475a; border-radius:8px; padding:8px; font-size:13px;
               font-family:inherit; resize:vertical; margin-bottom:12px;"></textarea>

      <label style="display:block; margin-bottom:4px; font-weight:600; color:#a6adc8; font-size:12px;">
        Mode
      </label>
      <div id="comfyclaw-gen-mode" style="display:flex; gap:6px; margin-bottom:12px;">
        <button data-mode="scratch" class="comfyclaw-mode-btn" style="flex:1; padding:6px 4px;
                border:2px solid #cba6f7; border-radius:8px; background:#cba6f722;
                color:#cba6f7; cursor:pointer; font-size:12px; font-weight:600;">
          ✨ From Scratch
        </button>
        <button data-mode="improve" class="comfyclaw-mode-btn" style="flex:1; padding:6px 4px;
                border:2px solid #45475a; border-radius:8px; background:#313244;
                color:#cdd6f4; cursor:pointer; font-size:12px; font-weight:600;">
          🔧 Improve Current
        </button>
      </div>

      <details style="margin-bottom:12px;">
        <summary style="cursor:pointer; color:#a6adc8; font-size:12px; font-weight:600;">
          ⚙ Settings
        </summary>
        <div style="padding-top:8px;">
          <div style="display:flex; gap:8px; align-items:center; margin-bottom:6px;">
            <label style="color:#a6adc8; font-size:12px; min-width:90px;">Iterations</label>
            <input id="comfyclaw-gen-iters" type="number" min="1" max="20" value="3"
              style="width:60px; background:#313244; color:#cdd6f4; border:1px solid #45475a;
                     border-radius:6px; padding:4px 8px; font-size:13px;">
          </div>
          <div style="display:flex; gap:8px; align-items:center; margin-bottom:6px;">
            <label style="color:#a6adc8; font-size:12px; min-width:90px;">Agent Model</label>
            <select id="comfyclaw-gen-model"
              style="flex:1; background:#313244; color:#cdd6f4; border:1px solid #45475a;
                     border-radius:6px; padding:4px 8px; font-size:12px;">
              <option value="">Server default</option>
              <option value="anthropic/claude-sonnet-4-5">Claude Sonnet 4.5</option>
              <option value="anthropic/claude-sonnet-4-20250514">Claude Sonnet 4</option>
              <option value="anthropic/claude-opus-4-20250514">Claude Opus 4</option>
              <option value="openai/gpt-4o">GPT-4o</option>
              <option value="openai/o3">o3</option>
              <option value="openai/o4-mini">o4-mini</option>
              <option value="gemini/gemini-2.5-pro">Gemini 2.5 Pro</option>
              <option value="gemini/gemini-2.5-flash">Gemini 2.5 Flash</option>
            </select>
          </div>
          <div style="display:flex; gap:8px; align-items:center; margin-bottom:6px;">
            <label style="color:#a6adc8; font-size:12px; min-width:90px;">Verifier</label>
            <select id="comfyclaw-gen-verifier"
              style="flex:1; background:#313244; color:#cdd6f4; border:1px solid #45475a;
                     border-radius:6px; padding:4px 8px; font-size:12px;">
              <option value="vlm">VLM (auto)</option>
              <option value="human">Human</option>
              <option value="hybrid">Hybrid</option>
            </select>
          </div>
          <div style="display:flex; gap:8px; align-items:center; margin-bottom:6px;">
            <label style="color:#a6adc8; font-size:12px; min-width:90px;">Verifier Model</label>
            <select id="comfyclaw-gen-vmodel"
              style="flex:1; background:#313244; color:#cdd6f4; border:1px solid #45475a;
                     border-radius:6px; padding:4px 8px; font-size:12px;">
              <option value="">Same as Agent</option>
              <option value="anthropic/claude-sonnet-4-5">Claude Sonnet 4.5</option>
              <option value="openai/gpt-4o">GPT-4o</option>
              <option value="gemini/gemini-2.5-flash">Gemini 2.5 Flash</option>
            </select>
          </div>
          <div style="display:flex; gap:8px; align-items:center; margin-bottom:6px;">
            <label style="color:#a6adc8; font-size:12px; min-width:90px;">API Key</label>
            <input id="comfyclaw-gen-apikey" type="password" placeholder="(use server default)"
              style="flex:1; background:#313244; color:#cdd6f4; border:1px solid #45475a;
                     border-radius:6px; padding:4px 8px; font-size:12px; font-family:monospace;">
          </div>
          <div style="display:flex; gap:8px; align-items:center;">
            <label style="color:#a6adc8; font-size:12px; min-width:90px;">Op Delay (ms)</label>
            <input id="comfyclaw-gen-opdelay" type="number" min="0" max="2000" value="400"
              style="width:70px; background:#313244; color:#cdd6f4; border:1px solid #45475a;
                     border-radius:6px; padding:4px 8px; font-size:13px;">
          </div>
        </div>
      </details>

      <button id="comfyclaw-gen-btn" style="width:100%; padding:10px; border:none;
              border-radius:8px; background:#a6e3a1; color:#1e1e2e; cursor:pointer;
              font-size:14px; font-weight:700; transition:background 0.15s;">
        ▶ Generate
      </button>
      <button id="comfyclaw-gen-stop" style="display:none; width:100%; padding:10px;
              border:none; border-radius:8px; background:#f38ba8; color:#1e1e2e;
              cursor:pointer; font-size:14px; font-weight:700; margin-top:6px;">
        ■ Stop
      </button>

      <div id="comfyclaw-gen-status" style="margin-top:12px; padding:8px 10px;
           background:#313244; border-radius:8px; font-size:12px; color:#a6adc8;
           display:none;"></div>
    </div>
  `;

  document.body.appendChild(panel);

  // Mode toggle
  let selectedMode = "scratch";
  const modeContainer = panel.querySelector("#comfyclaw-gen-mode");
  modeContainer.querySelectorAll(".comfyclaw-mode-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      selectedMode = btn.dataset.mode;
      modeContainer.querySelectorAll(".comfyclaw-mode-btn").forEach(b => {
        b.style.borderColor = "#45475a";
        b.style.background = "#313244";
        b.style.color = "#cdd6f4";
      });
      btn.style.borderColor = "#cba6f7";
      btn.style.background = "#cba6f722";
      btn.style.color = "#cba6f7";
    });
  });

  // Drag + collapse/expand
  const header = panel.querySelector("#comfyclaw-gen-header");
  const body = panel.querySelector("#comfyclaw-gen-body");
  const toggle = panel.querySelector("#comfyclaw-gen-toggle");
  let collapsed = false;

  let _dragState = null;
  const DRAG_THRESHOLD = 5;

  header.addEventListener("mousedown", (e) => {
    if (e.button !== 0) return;
    const rect = panel.getBoundingClientRect();
    _dragState = {
      startX: e.clientX,
      startY: e.clientY,
      offsetX: e.clientX - rect.left,
      offsetY: e.clientY - rect.top,
      dragging: false,
    };
    e.preventDefault();
  });

  document.addEventListener("mousemove", (e) => {
    if (!_dragState) return;
    const dx = e.clientX - _dragState.startX;
    const dy = e.clientY - _dragState.startY;
    if (!_dragState.dragging && Math.hypot(dx, dy) < DRAG_THRESHOLD) return;
    _dragState.dragging = true;
    header.style.cursor = "grabbing";
    // Switch from right-anchored to left/top positioning
    panel.style.right = "auto";
    panel.style.left = Math.max(0, e.clientX - _dragState.offsetX) + "px";
    panel.style.top = Math.max(0, e.clientY - _dragState.offsetY) + "px";
  });

  document.addEventListener("mouseup", () => {
    if (!_dragState) return;
    const wasDrag = _dragState.dragging;
    _dragState = null;
    header.style.cursor = "grab";
    if (wasDrag) {
      // Persist position
      localStorage.setItem("comfyclaw_panel_pos", JSON.stringify({
        left: panel.style.left, top: panel.style.top,
      }));
    } else {
      collapsed = !collapsed;
      body.style.display = collapsed ? "none" : "block";
      toggle.textContent = collapsed ? "▶" : "▼";
    }
  });

  // Restore saved position
  try {
    const saved = JSON.parse(localStorage.getItem("comfyclaw_panel_pos"));
    if (saved?.left && saved?.top) {
      panel.style.right = "auto";
      panel.style.left = saved.left;
      panel.style.top = saved.top;
    }
  } catch (_) { /* ignore */ }

  // Persist settings to localStorage
  const _settingsFields = [
    "comfyclaw-gen-model", "comfyclaw-gen-verifier", "comfyclaw-gen-vmodel",
    "comfyclaw-gen-apikey", "comfyclaw-gen-opdelay", "comfyclaw-gen-iters",
  ];
  _settingsFields.forEach(id => {
    const el = panel.querySelector(`#${id}`);
    if (!el) return;
    const stored = localStorage.getItem(id);
    if (stored !== null) el.value = stored;
    el.addEventListener("change", () => {
      localStorage.setItem(id, el.value);
      if (id === "comfyclaw-gen-opdelay") {
        localStorage.setItem("comfyclaw_op_delay", el.value);
      }
    });
  });

  // Generate button
  panel.querySelector("#comfyclaw-gen-btn").addEventListener("click", async () => {
    const prompt = panel.querySelector("#comfyclaw-gen-prompt").value.trim();
    if (!prompt) {
      panel.querySelector("#comfyclaw-gen-prompt").focus();
      return;
    }
    let workflow = null;
    if (selectedMode === "improve") {
      workflow = await exportCurrentWorkflow();
    }
    const settingsPayload = {
      iterations:      parseInt(panel.querySelector("#comfyclaw-gen-iters").value) || 3,
      verifier_mode:   panel.querySelector("#comfyclaw-gen-verifier").value,
      model:           panel.querySelector("#comfyclaw-gen-model").value,
      verifier_model:  panel.querySelector("#comfyclaw-gen-vmodel").value,
      api_key:         panel.querySelector("#comfyclaw-gen-apikey").value.trim(),
    };
    const msg = {
      type:     "trigger_generation",
      prompt:   prompt,
      mode:     selectedMode,
      workflow: workflow,
      settings: settingsPayload,
    };
    if (_activeSyncClient?.ws?.readyState === WebSocket.OPEN) {
      _activeSyncClient.ws.send(JSON.stringify(msg));
      console.log("[ComfyClaw] Sent trigger_generation:", msg.mode, msg.prompt.slice(0, 60));
      setGenRunning(true);
      setGenStatus("running", "Waiting for agent...");
      clearAgentLog();
    }
  });

  // Stop button
  panel.querySelector("#comfyclaw-gen-stop").addEventListener("click", () => {
    if (_activeSyncClient?.ws?.readyState === WebSocket.OPEN) {
      _activeSyncClient.ws.send(JSON.stringify({ type: "cancel_generation" }));
    }
    setGenRunning(false);
    setGenStatus("idle", "Cancelled.");
  });

  return panel;
}

async function exportCurrentWorkflow() {
  try {
    if (typeof app.graphToPrompt === "function") {
      const result = await app.graphToPrompt();
      return result?.output || result?.workflow || null;
    }
  } catch (err) {
    console.warn("[ComfyClaw] Failed to export workflow:", err);
  }
  if (Object.keys(_currentApiWorkflow).length > 0) {
    return JSON.parse(JSON.stringify(_currentApiWorkflow));
  }
  return null;
}

function setGenRunning(running) {
  _clawPanelRunning = running;
  if (!_clawPanel) return;
  const genBtn  = _clawPanel.querySelector("#comfyclaw-gen-btn");
  const stopBtn = _clawPanel.querySelector("#comfyclaw-gen-stop");
  genBtn.style.display  = running ? "none" : "block";
  stopBtn.style.display = running ? "block" : "none";
}

function setGenStatus(state, text) {
  if (!_clawPanel) return;
  const el = _clawPanel.querySelector("#comfyclaw-gen-status");
  el.style.display = text ? "block" : "none";

  const colors = {
    running:   "#89b4fa",
    verifying: "#f9e2af",
    repairing: "#fab387",
    complete:  "#a6e3a1",
    error:     "#f38ba8",
    idle:      "#a6adc8",
  };
  el.style.color = colors[state] || "#a6adc8";
  el.textContent = text;
}

// ─────────────────────────────────────────────────────────────────────────────
// Agent Thinking Log panel
// ─────────────────────────────────────────────────────────────────────────────

let _thinkingPanel = null;
let _thinkingEntries = [];
const MAX_LOG_ENTRIES = 200;

const EVENT_STYLES = {
  strategy:    { icon: "🧠", color: "#cba6f7", label: "Strategy"   },
  tool_call:   { icon: "🔧", color: "#89b4fa", label: "Tool Call"  },
  tool_result: { icon: "📋", color: "#a6adc8", label: "Result"     },
  thinking:    { icon: "💭", color: "#f9e2af", label: "Thinking"   },
  validation:  { icon: "✓",  color: "#a6e3a1", label: "Validation" },
  error:       { icon: "❌", color: "#f38ba8", label: "Error"      },
  info:        { icon: "ℹ",  color: "#74c7ec", label: "Info"       },
  user:        { icon: "👤", color: "#fab387", label: "You"        },
};

function createThinkingPanel() {
  const panel = document.createElement("div");
  panel.id = "comfyclaw-thinking-panel";
  Object.assign(panel.style, {
    position:      "fixed",
    top:           "60px",
    left:          "12px",
    width:         "380px",
    maxHeight:     "70vh",
    zIndex:        "9997",
    background:    "#1e1e2e",
    color:         "#cdd6f4",
    borderRadius:  "12px",
    boxShadow:     "0 4px 24px rgba(0,0,0,0.4)",
    fontFamily:    "system-ui, -apple-system, sans-serif",
    fontSize:      "12px",
    lineHeight:    "1.4",
    display:       "flex",
    flexDirection: "column",
    overflow:      "hidden",
  });

  panel.innerHTML = `
    <div id="comfyclaw-think-header" style="padding:10px 14px; background:#313244;
         cursor:grab; display:flex; justify-content:space-between; align-items:center;
         user-select:none; flex-shrink:0;">
      <span style="font-weight:700; font-size:13px;">🧠 Agent Thinking</span>
      <div style="display:flex; gap:6px; align-items:center;">
        <span id="comfyclaw-think-count" style="font-size:11px; color:#a6adc8;"></span>
        <span id="comfyclaw-think-toggle" style="font-size:11px; color:#a6adc8; cursor:pointer;">▼</span>
      </div>
    </div>
    <div id="comfyclaw-think-body" style="display:flex; flex-direction:column; flex:1; min-height:0;">
      <div id="comfyclaw-think-log" style="flex:1; overflow-y:auto; padding:8px 10px;
           max-height:calc(70vh - 110px); scroll-behavior:smooth;"></div>
      <div id="comfyclaw-think-input-area" style="padding:8px 10px; border-top:1px solid #45475a;
           flex-shrink:0;">
        <div style="display:flex; gap:6px;">
          <input id="comfyclaw-think-input" type="text"
            placeholder="Tell the agent what to change…"
            style="flex:1; background:#313244; color:#cdd6f4; border:1px solid #45475a;
                   border-radius:8px; padding:6px 10px; font-size:12px; font-family:inherit;">
          <button id="comfyclaw-think-send" style="padding:6px 12px; border:none;
                  border-radius:8px; background:#cba6f7; color:#1e1e2e; cursor:pointer;
                  font-size:12px; font-weight:700; white-space:nowrap;">
            Send ↵
          </button>
        </div>
      </div>
    </div>
  `;

  document.body.appendChild(panel);

  // Drag logic (same pattern as gen panel)
  const header = panel.querySelector("#comfyclaw-think-header");
  const body = panel.querySelector("#comfyclaw-think-body");
  const toggle = panel.querySelector("#comfyclaw-think-toggle");
  let collapsed = false;
  let _dragState = null;

  header.addEventListener("mousedown", (e) => {
    if (e.target.id === "comfyclaw-think-toggle") return;
    if (e.button !== 0) return;
    const rect = panel.getBoundingClientRect();
    _dragState = {
      startX: e.clientX, startY: e.clientY,
      offsetX: e.clientX - rect.left, offsetY: e.clientY - rect.top,
      dragging: false,
    };
    e.preventDefault();
  });

  document.addEventListener("mousemove", (e) => {
    if (!_dragState) return;
    const dx = e.clientX - _dragState.startX;
    const dy = e.clientY - _dragState.startY;
    if (!_dragState.dragging && Math.hypot(dx, dy) < 5) return;
    _dragState.dragging = true;
    header.style.cursor = "grabbing";
    panel.style.left = Math.max(0, e.clientX - _dragState.offsetX) + "px";
    panel.style.top = Math.max(0, e.clientY - _dragState.offsetY) + "px";
  });

  document.addEventListener("mouseup", () => {
    if (!_dragState) return;
    const wasDrag = _dragState.dragging;
    _dragState = null;
    header.style.cursor = "grab";
    if (wasDrag) {
      localStorage.setItem("comfyclaw_think_pos", JSON.stringify({
        left: panel.style.left, top: panel.style.top,
      }));
    }
  });

  toggle.addEventListener("click", () => {
    collapsed = !collapsed;
    body.style.display = collapsed ? "none" : "flex";
    toggle.textContent = collapsed ? "▶" : "▼";
  });

  // Restore position
  try {
    const saved = JSON.parse(localStorage.getItem("comfyclaw_think_pos"));
    if (saved?.left && saved?.top) {
      panel.style.left = saved.left;
      panel.style.top = saved.top;
    }
  } catch (_) {}

  // Send refinement
  const input = panel.querySelector("#comfyclaw-think-input");
  const sendBtn = panel.querySelector("#comfyclaw-think-send");

  function sendRefinement() {
    const text = input.value.trim();
    if (!text) return;
    if (_activeSyncClient?.ws?.readyState === WebSocket.OPEN) {
      _activeSyncClient.ws.send(JSON.stringify({
        type: "user_refinement",
        text: text,
      }));
      appendAgentLog({ event_type: "user", content: text, timestamp: Date.now() / 1000 });
      input.value = "";
    }
  }

  sendBtn.addEventListener("click", sendRefinement);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendRefinement();
    }
  });

  return panel;
}

function appendAgentLog(event) {
  if (!_thinkingPanel) return;
  const logEl = _thinkingPanel.querySelector("#comfyclaw-think-log");
  if (!logEl) return;

  _thinkingEntries.push(event);
  if (_thinkingEntries.length > MAX_LOG_ENTRIES) {
    _thinkingEntries.shift();
    if (logEl.firstChild) logEl.removeChild(logEl.firstChild);
  }

  const style = EVENT_STYLES[event.event_type] || EVENT_STYLES.info;
  const entry = document.createElement("div");
  Object.assign(entry.style, {
    marginBottom:  "4px",
    padding:       "4px 8px",
    borderRadius:  "6px",
    background:    "#313244",
    borderLeft:    `3px solid ${style.color}`,
    fontSize:      "12px",
    lineHeight:    "1.4",
    wordBreak:     "break-word",
  });

  const time = event.timestamp
    ? new Date(event.timestamp * 1000).toLocaleTimeString()
    : "";
  const iterBadge = event.iteration
    ? `<span style="color:#585b70; margin-left:4px;">[iter ${event.iteration}]</span>`
    : "";

  let body = escapeHtml(event.content || "");

  // For tool_call, show tool name and abbreviated args
  if (event.event_type === "tool_call" && event.tool_name) {
    const argsStr = event.tool_args
      ? Object.entries(event.tool_args).map(([k, v]) =>
          `<span style="color:#a6adc8;">${escapeHtml(k)}</span>=<span style="color:#f9e2af;">${escapeHtml(String(v).slice(0, 80))}</span>`
        ).join(", ")
      : "";
    body = `<span style="color:${style.color}; font-weight:600;">${escapeHtml(event.tool_name)}</span>`
      + (argsStr ? `<br><span style="font-size:11px;">${argsStr}</span>` : "");
  }

  // For strategy, render as formatted block
  if (event.event_type === "strategy") {
    body = body.replace(/\*\*(.*?)\*\*/g, '<strong style="color:#cba6f7;">$1</strong>');
    body = body.replace(/\n/g, "<br>");
  }

  // For tool_result, dim it
  if (event.event_type === "tool_result") {
    entry.style.opacity = "0.7";
    entry.style.fontSize = "11px";
  }

  entry.innerHTML = `
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:2px;">
      <span>${style.icon} <span style="color:${style.color}; font-weight:600; font-size:11px;">${style.label}</span>${iterBadge}</span>
      <span style="color:#585b70; font-size:10px;">${time}</span>
    </div>
    <div>${body}</div>
  `;

  logEl.appendChild(entry);

  // Auto-scroll to bottom
  logEl.scrollTop = logEl.scrollHeight;

  // Update count badge
  const countEl = _thinkingPanel.querySelector("#comfyclaw-think-count");
  if (countEl) countEl.textContent = `${_thinkingEntries.length}`;
}

function clearAgentLog() {
  _thinkingEntries = [];
  if (!_thinkingPanel) return;
  const logEl = _thinkingPanel.querySelector("#comfyclaw-think-log");
  if (logEl) logEl.innerHTML = "";
  const countEl = _thinkingPanel.querySelector("#comfyclaw-think-count");
  if (countEl) countEl.textContent = "";
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
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
    if (msg.type === "workflow_update") {
      const wf = msg.workflow || {};
      _currentApiWorkflow = JSON.parse(JSON.stringify(wf));
      const nodeCount = Object.keys(wf).length;
      if (nodeCount === 0) {
        // "From scratch" — reset internal state; canvas stays untouched
        // until the first new node arrives via workflow_diff.
        console.log("[ComfyClaw] State reset (from-scratch); waiting for new nodes…");
      } else {
        const ok = await loadWorkflowIntoCanvas(wf);
        if (ok) {
          setStatus("updated", `${nodeCount} nodes`);
        }
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
    } else if (msg.type === "generation_status") {
      const detail = msg.detail || msg.state;
      const iter = msg.iteration ? ` (iter ${msg.iteration})` : "";
      setGenStatus(msg.state, `${detail}${iter}`);
      console.log(`[ComfyClaw] Generation status: ${msg.state}${iter}`);
    } else if (msg.type === "generation_complete") {
      setGenRunning(false);
      setGenStatus("complete",
        `Done! Score: ${(msg.score ?? 0).toFixed(2)} in ${msg.iterations_used} iteration(s)`);
      console.log("[ComfyClaw] Generation complete:", msg);
    } else if (msg.type === "generation_error") {
      setGenRunning(false);
      setGenStatus("error", `Error: ${msg.error}`);
      appendAgentLog({ event_type: "error", content: msg.error, timestamp: Date.now() / 1000 });
      console.error("[ComfyClaw] Generation error:", msg.error);
    } else if (msg.type === "agent_event") {
      appendAgentLog(msg);
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
    console.log("[ComfyClaw] Extension loaded — ComfyClaw Sync Bridge v5.0");
    statusEl = createStatusBadge();
    _clawPanel = createClawPanel();
    _thinkingPanel = createThinkingPanel();
    setTimeout(() => {
      _activeSyncClient = new SyncClient();
      _activeSyncClient.connect();
    }, 500);
  },
});
