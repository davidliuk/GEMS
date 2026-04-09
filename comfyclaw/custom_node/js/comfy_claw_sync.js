/**
 * ComfyClaw Sync Extension  v1.1
 *
 * Connects to the ComfyClaw Python sync server (ws://127.0.0.1:8765 by default)
 * and reloads the ComfyUI canvas in real time whenever the agent modifies the
 * workflow topology.
 *
 * Protocol — the Python SyncServer sends:
 *   { "type": "workflow_update", "workflow": { "<nodeId>": { class_type, inputs, ... }, ... } }
 *
 * Configuration (persisted in localStorage):
 *   localStorage.setItem('comfyclaw_ws_url', 'ws://127.0.0.1:8765');
 *
 * Status badge:
 *   🔄 connecting  |  🟢 live  |  ✨ updated (flashes 2 s)  |  🔴 disconnected
 */

import { app } from "../../scripts/app.js";

const DEFAULT_WS_URL       = `ws://${window.location.hostname}:8765`;
const RECONNECT_DELAY_MS   = 3000;
const MAX_RECONNECT_ATTEMPTS = 20;

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
// Workflow loading
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Returns true if ``data`` looks like a ComfyUI API-format workflow
 * (flat object keyed by numeric string node IDs, each with a class_type).
 */
function isApiFormat(data) {
  if (typeof data !== "object" || data === null || Array.isArray(data)) return false;
  const keys = Object.keys(data);
  if (keys.length === 0) return false;
  return keys.every(k => /^\d+$/.test(k) && data[k] && data[k].class_type);
}

/**
 * Convert API-format workflow → minimal LiteGraph graph object.
 * Node positions are arranged left-to-right in columns of 5.
 */
function apiToLitegraph(apiWf) {
  const nodes  = [];
  const links  = [];
  let linkCounter = 0;
  const linkMap   = {};   // "srcId:srcIdx" → linkId

  const ids   = Object.keys(apiWf).sort((a, b) => parseInt(a) - parseInt(b));
  const COLS  = 5;
  const NODE_W = 220, NODE_H = 180, GAP_X = 60, GAP_Y = 40;

  // Assign grid positions
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
        // Link reference: [srcNodeId, srcOutputIdx]
        const [srcId, srcIdx] = val;
        const linkKey = `${srcId}:${srcIdx}`;
        let lid;
        if (linkMap[linkKey] !== undefined) {
          lid = linkMap[linkKey];
        } else {
          lid = linkCounter++;
          linkMap[linkKey] = lid;
          // LiteGraph link: [link_id, src_node_id, src_slot, dst_node_id, dst_slot, type]
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

/**
 * Load ``data`` (API or LiteGraph format) into the ComfyUI canvas.
 * Tries three methods in order for broad version compatibility.
 */
async function loadWorkflowIntoCanvas(data) {
  try {
    // Method 1: ComfyUI ≥ 0.2 — loadApiJson accepts API format natively
    if (isApiFormat(data) && typeof app.loadApiJson === "function") {
      await app.loadApiJson(data);
      console.log("[ComfyClaw] Loaded via app.loadApiJson");
      return true;
    }

    // Method 2: loadGraphData — expects LiteGraph / UI format
    const graphData = isApiFormat(data) ? apiToLitegraph(data) : data;
    if (typeof app.loadGraphData === "function") {
      await app.loadGraphData(graphData);
      console.log("[ComfyClaw] Loaded via app.loadGraphData");
      return true;
    }

    // Method 3: low-level LiteGraph configure
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
// WebSocket client with auto-reconnect
// ─────────────────────────────────────────────────────────────────────────────

class SyncClient {
  constructor() {
    this.ws = null;
    this.reconnectAttempts = 0;
    this.destroyed = false;
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

    this.ws.onmessage = async (event) => {
      try {
        const msg = JSON.parse(event.data);
        // Python SyncServer sends: { type: "workflow_update", workflow: {...} }
        if (msg.type === "workflow_update" && msg.workflow) {
          const nodeCount = Object.keys(msg.workflow).length;
          const ok = await loadWorkflowIntoCanvas(msg.workflow);
          if (ok) {
            setStatus("updated", `${nodeCount} nodes`);
          }
        }
      } catch (err) {
        console.error("[ComfyClaw] Message parse error:", err);
      }
    };

    this.ws.onerror = () => {
      // Suppress noisy browser error — reconnect logic handles it
    };

    this.ws.onclose = () => {
      if (!this.destroyed) {
        setStatus("disconnected");
        this._scheduleReconnect();
      }
    };
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
    console.log("[ComfyClaw] Extension loaded — ComfyClaw Sync Bridge v1.1");
    statusEl = createStatusBadge();
    // Small delay so the rest of ComfyUI finishes initialising
    setTimeout(() => new SyncClient().connect(), 500);
  },
});
