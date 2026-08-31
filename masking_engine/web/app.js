const promptInput = document.getElementById("prompt-input");
const analyzeBtn = document.getElementById("analyze-btn");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const errorBanner = document.getElementById("error-banner");
const demoButtonsEl = document.getElementById("demo-buttons");

const RISK_CLASS = {
  CRITICAL: "critical",
  HIGH: "high",
  MEDIUM: "medium",
  LOW: "low",
};

function riskClass(level) {
  if (!level) return "";
  return RISK_CLASS[level.toUpperCase()] || "low";
}

function actionBadgeClass(confidenceLevel) {
  const key = (confidenceLevel || "").toLowerCase().replace(/\s+/g, "-");
  if (key === "very-high" || key === "high") return "badge-high";
  if (key === "medium") return "badge-medium";
  if (key === "low") return "badge-low";
  return "badge-very-low";
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

async function loadDemoCases() {
  try {
    const res = await fetch("/api/demo-cases");
    const data = await res.json();
    demoButtonsEl.innerHTML = "";
    (data.cases || []).forEach((c) => {
      const btn = document.createElement("button");
      btn.textContent = c.label;
      btn.title = c.text;
      btn.addEventListener("click", () => {
        promptInput.value = c.text;
        analyze();
      });
      demoButtonsEl.appendChild(btn);
    });
  } catch (err) {
    // Non-fatal — quick-fill is a convenience, not required for the tool to work.
    demoButtonsEl.textContent = "(couldn't load demo cases)";
  }
}

function renderEntity(entity) {
  const cls = riskClass(entity.sensitivity);
  const bd = entity.score_breakdown || {};
  const scorePct = Math.round((entity.score || 0) * 100);

  return `
    <div class="entity-card ${cls}">
      <div class="entity-header">
        <span class="entity-type">${escapeHtml(entity.entity_type)}</span>
        <span class="action-pill ${actionBadgeClass(entity.confidence_level)}">${escapeHtml(entity.action)}</span>
      </div>
      <div class="entity-meta">
        category ${escapeHtml(entity.category)} &middot; sensitivity ${escapeHtml(entity.sensitivity)} &middot; chars [${entity.start}, ${entity.end})
      </div>
      <div class="entity-value">${escapeHtml(entity.value)}</div>
      <div class="pattern-row">${escapeHtml(entity.matched_pattern)}</div>
      <div class="score-row">
        <div class="score-bar-track">
          <div class="score-bar-fill" style="width:${scorePct}%; background:currentColor" ></div>
        </div>
        <strong>${entity.score.toFixed(2)}</strong>
        <span>${escapeHtml(entity.confidence_level)}</span>
      </div>
      <div class="factor-breakdown">
        <span>pattern <strong>${(bd.pattern_strength ?? 0).toFixed(2)}</strong></span>
        <span>keyword <strong>${(bd.keyword_proximity ?? 0).toFixed(2)}</strong></span>
        <span>co-occur <strong>${(bd.co_occurrence ?? 0).toFixed(2)}</strong></span>
        <span>format <strong>${(bd.format_validity ?? 0).toFixed(2)}</strong></span>
        ${bd.elevated_by_cooccur && bd.elevated_by_cooccur !== "—" ? `<span>&#9888; elevated to <strong>${escapeHtml(bd.elevated_by_cooccur)}</strong></span>` : ""}
      </div>
    </div>
  `;
}

function renderResults(data) {
  errorBanner.classList.add("hidden");
  errorBanner.innerHTML = "";

  document.getElementById("masked-text").textContent = data.masked_text;

  const riskBadge = document.getElementById("risk-badge");
  const cls = riskClass(data.overall_risk);
  riskBadge.textContent = `Overall risk: ${data.overall_risk || "NONE"}`;
  riskBadge.className = "risk-badge badge-" + (actionBadgeClass(data.overall_risk).replace("badge-", ""));

  const transformBox = document.getElementById("transformations-box");
  if (data.transformations && data.transformations.length) {
    transformBox.classList.remove("hidden");
    let text = data.transformations.join(", ");
    if (data.normalized) text += ` → normalized: "${data.normalized}"`;
    if (data.despaced) text += ` → despaced: "${data.despaced}"`;
    document.getElementById("transformations-text").textContent = text;
  } else {
    transformBox.classList.add("hidden");
  }

  const mlBox = document.getElementById("ml-safety-net-box");
  const mlNet = data.ml_safety_net;
  if (mlNet && mlNet.flagged && mlNet.masked) {
    mlBox.classList.remove("hidden");
    mlBox.classList.add("flagged");
    document.getElementById("ml-safety-net-text").textContent =
      `no rule-based pattern matched, but the trained classifier scored this prompt ${Math.round(mlNet.score * 100)}% likely sensitive by structure/entropy — the suspicious token(s) below were masked as <ML_FLAGGED_...> (see engine/ml_anomaly.py).`;
  } else if (mlNet && mlNet.flagged) {
    mlBox.classList.remove("hidden");
    mlBox.classList.add("flagged");
    document.getElementById("ml-safety-net-text").textContent =
      `no rule-based pattern matched, but the trained classifier scored this prompt ${Math.round(mlNet.score * 100)}% likely sensitive by structure/entropy — no specific span could be located, so it's flagged for human review instead of masked (see engine/ml_anomaly.py).`;
  } else if (mlNet && mlNet.available) {
    mlBox.classList.remove("hidden");
    mlBox.classList.remove("flagged");
    document.getElementById("ml-safety-net-text").textContent =
      "checked — no anomaly beyond what Layer 1 already found.";
  } else {
    mlBox.classList.add("hidden");
  }

  const listEl = document.getElementById("entities-list");
  if (!data.entities || data.entities.length === 0) {
    listEl.innerHTML = '<p class="no-entities">No sensitive entities detected.</p>';
  } else {
    listEl.innerHTML = data.entities.map(renderEntity).join("");
  }

  resultsEl.classList.remove("hidden");
}

function renderError(message) {
  errorBanner.classList.remove("hidden");
  errorBanner.innerHTML = `<div class="box">${escapeHtml(message)}</div>`;
  resultsEl.classList.add("hidden");
}

async function analyze() {
  const text = promptInput.value;
  if (!text || !text.trim()) {
    renderError("Enter some text first.");
    return;
  }

  analyzeBtn.disabled = true;
  statusEl.textContent = "Analyzing…";

  try {
    const res = await fetch("/api/detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    if (!res.ok) {
      renderError(data.error || `Request failed (${res.status})`);
    } else {
      renderResults(data);
    }
  } catch (err) {
    renderError(`Could not reach the server: ${err.message}`);
  } finally {
    analyzeBtn.disabled = false;
    statusEl.textContent = "";
  }
}

analyzeBtn.addEventListener("click", analyze);
promptInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    analyze();
  }
});

loadDemoCases();
