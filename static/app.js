const state = {
  imageData: "",
  image: new Image(),
  result: null,
  mode: "outgoing",
  busy: false,
};

const els = {
  canvas: document.querySelector("#imageCanvas"),
  fileInput: document.querySelector("#fileInput"),
  status: document.querySelector("#status"),
  modelMode: document.querySelector("#modelMode"),
  patchSize: document.querySelector("#patchSize"),
  layer: document.querySelector("#layer"),
  head: document.querySelector("#head"),
  token: document.querySelector("#token"),
  layerValue: document.querySelector("#layerValue"),
  headValue: document.querySelector("#headValue"),
  tokenValue: document.querySelector("#tokenValue"),
  modelInfo: document.querySelector("#modelInfo"),
  tokenCount: document.querySelector("#tokenCount"),
  gridInfo: document.querySelector("#gridInfo"),
  maxWeight: document.querySelector("#maxWeight"),
  layers: document.querySelector("#layers"),
  tokens: document.querySelector("#tokens"),
  modeButtons: document.querySelectorAll("[data-mode]"),
};

const ctx = els.canvas.getContext("2d");

function sampleImage() {
  const c = document.createElement("canvas");
  c.width = 224;
  c.height = 224;
  const g = c.getContext("2d");
  const sky = g.createLinearGradient(0, 0, 224, 224);
  sky.addColorStop(0, "#1e3a5f");
  sky.addColorStop(0.55, "#477e88");
  sky.addColorStop(1, "#f2c14e");
  g.fillStyle = sky;
  g.fillRect(0, 0, 224, 224);
  g.fillStyle = "#f7f1df";
  g.beginPath();
  g.arc(164, 54, 24, 0, Math.PI * 2);
  g.fill();
  g.fillStyle = "#21342f";
  g.beginPath();
  g.moveTo(0, 210);
  g.lineTo(84, 92);
  g.lineTo(154, 210);
  g.closePath();
  g.fill();
  g.fillStyle = "#315047";
  g.beginPath();
  g.moveTo(62, 210);
  g.lineTo(148, 118);
  g.lineTo(224, 210);
  g.closePath();
  g.fill();
  g.fillStyle = "#ef6f6c";
  g.fillRect(64, 132, 48, 38);
  g.fillStyle = "#101316";
  g.fillRect(76, 143, 12, 12);
  g.fillRect(94, 143, 12, 12);
  return c.toDataURL("image/png");
}

function setImage(dataUrl) {
  state.imageData = dataUrl;
  state.image.onload = () => {
    draw();
    analyze();
  };
  state.image.src = dataUrl;
}

async function analyze() {
  if (!state.imageData || state.busy) return;
  state.busy = true;
  setStatus("Считаю attention...");
  syncLabels();

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        imageData: state.imageData,
        modelMode: els.modelMode.value,
        patchSize: Number(els.patchSize.value),
        layer: Number(els.layer.value),
        head: Number(els.head.value),
        token: Number(els.token.value),
      }),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || "Ошибка анализа");
    }
    state.result = await response.json();
    applyResultLimits();
    els.token.max = state.result.tokenCount - 1;
    if (Number(els.token.value) > state.result.tokenCount - 1) els.token.value = 0;
    renderResult();
    setStatus("Click to choose single patch");
  } catch (error) {
    setStatus(error.message);
  } finally {
    state.busy = false;
  }
}

function draw() {
  const size = els.canvas.width;
  ctx.clearRect(0, 0, size, size);
  ctx.drawImage(state.image, 0, 0, size, size);
  if (!state.result) return;

  const weights = currentWeights();
  const grid = state.result.grid;
  const cell = size / grid;
  const max = Math.max(...weights, 1e-6);

  weights.forEach((weight, index) => {
    const x = index % grid;
    const y = Math.floor(index / grid);
    const t = Math.min(weight / max, 1);
    ctx.fillStyle = `rgba(${Math.round(239 * t + 93 * (1 - t))}, ${Math.round(111 * t + 212 * (1 - t))}, ${Math.round(108 * t + 198 * (1 - t))}, ${0.08 + t * 0.62})`;
    ctx.fillRect(x * cell, y * cell, cell, cell);
  });

  ctx.strokeStyle = "rgba(255,255,255,0.32)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= grid; i += 1) {
    ctx.beginPath();
    ctx.moveTo(i * cell, 0);
    ctx.lineTo(i * cell, size);
    ctx.moveTo(0, i * cell);
    ctx.lineTo(size, i * cell);
    ctx.stroke();
  }

  const token = Number(els.token.value);
  if (token > 0) {
    const patch = token - 1;
    const x = patch % grid;
    const y = Math.floor(patch / grid);
    ctx.strokeStyle = "#5dd4c6";
    ctx.lineWidth = 4;
    ctx.strokeRect(x * cell + 2, y * cell + 2, cell - 4, cell - 4);
  }
}

function renderResult() {
  if (!state.result) return;
  const weights = currentWeights();
  const max = Math.max(...weights, 0);
  els.tokenCount.textContent = state.result.tokenCount;
  els.gridInfo.textContent = `${state.result.grid} x ${state.result.grid}`;
  els.modelInfo.textContent = modelShortName(state.result.modelMode);
  els.maxWeight.textContent = max.toFixed(3);
  renderLayers();
  renderTokens();
  syncLabels();
  draw();
}

function renderLayers() {
  const rows = state.result.layerSummaries.map((values, index) => {
    const average = values.reduce((a, b) => a + b, 0) / values.length;
    const width = Math.max(2, average * state.result.patchCount * 100);
    return `<div class="layerRow"><span>Layer ${index}</span><div class="bar"><i style="width:${Math.min(width, 100)}%"></i></div></div>`;
  });
  els.layers.innerHTML = rows.join("");
}

function renderTokens() {
  const selected = Number(els.token.value);
  els.tokens.innerHTML = state.result.featurePreview
    .slice(0, 96)
    .map((item) => {
      const rgb = item.rgb.map((value) => Math.round(value * 255)).join(", ");
      return `<button class="tokenCard ${selected === item.token ? "active" : ""}" data-token="${item.token}">
        <strong>#${item.token} (${item.x}, ${item.y})</strong>
        <span>RGB ${rgb}</span>
        <span>edge ${item.edge}</span>
      </button>`;
    })
    .join("");
}

function currentWeights() {
  if (!state.result) return [];
  if (state.mode === "incoming") return state.result.incomingAttention;
  return state.result.attention;
}

function syncLabels() {
  els.layerValue.textContent = els.layer.value;
  els.headValue.textContent = els.head.value;
  els.tokenValue.textContent = Number(els.token.value) === 0 ? "CLS" : `#${els.token.value}`;
}

function setStatus(message) {
  els.status.textContent = message;
}

function debounce(fn, delay = 180) {
  let timer;
  return (...args) => {
    window.clearTimeout(timer);
    timer = window.setTimeout(() => fn(...args), delay);
  };
}

const delayedAnalyze = debounce(analyze);

[els.modelMode, els.patchSize, els.layer, els.head, els.token].forEach((el) => {
  el.addEventListener("input", () => {
    applyModeControls();
    syncLabels();
    delayedAnalyze();
  });
});

function applyModeControls() {
  const mode = els.modelMode.value;
  const lockedPatch = mode === "vit_b_16" || mode === "dinov2_vits14";
  if (mode === "vit_b_16") els.patchSize.value = "16";
  if (mode === "dinov2_vits14") els.patchSize.value = "14";
  els.patchSize.disabled = lockedPatch;
  els.layer.max = mode === "simulation" ? 7 : 11;
  els.head.max = mode === "vit_b_16" ? 11 : 5;
  if (Number(els.layer.value) > Number(els.layer.max)) els.layer.value = els.layer.max;
  if (Number(els.head.value) > Number(els.head.max)) els.head.value = els.head.max;
}

function applyResultLimits() {
  if (!state.result) return;
  els.layer.max = state.result.layerCount - 1;
  els.head.max = state.result.headCount - 1;
  els.layer.value = state.result.layer;
  els.head.value = state.result.head;
}

function modelShortName(mode) {
  if (mode === "vit_b_16") return "ViT-B/16";
  if (mode === "dinov2_vits14") return "DINOv2";
  return "Sim";
}

els.fileInput.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => setImage(reader.result);
  reader.readAsDataURL(file);
});

els.canvas.addEventListener("click", (event) => {
  if (!state.result) return;
  const rect = els.canvas.getBoundingClientRect();
  const x = ((event.clientX - rect.left) / rect.width) * state.result.grid;
  const y = ((event.clientY - rect.top) / rect.height) * state.result.grid;
  const patch = Math.floor(y) * state.result.grid + Math.floor(x);
  els.token.value = patch + 1;
  analyze();
});

els.tokens.addEventListener("click", (event) => {
  const card = event.target.closest("[data-token]");
  if (!card) return;
  els.token.value = card.dataset.token;
  analyze();
});

els.modeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    els.modeButtons.forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    state.mode = button.dataset.mode;
    renderResult();
  });
});

setImage(sampleImage());
