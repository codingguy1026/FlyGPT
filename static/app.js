const chatMessages = document.getElementById('chatMessages');
const chatForm = document.getElementById('chatForm');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');

const brainCanvas = document.getElementById('brainCanvas');
const brainStage = document.getElementById('brainStage');
const brainStatus = document.getElementById('brainStatus');
const brainEmpty = document.getElementById('brainEmpty');
const brainTooltip = document.getElementById('brainTooltip');
const stageChip = document.getElementById('stageChip');
const brainNodeCount = document.getElementById('brainNodeCount');
const brainEdgeCount = document.getElementById('brainEdgeCount');
const brainStepCount = document.getElementById('brainStepCount');
const routeName = document.getElementById('routeName');
const routeConfidence = document.getElementById('routeConfidence');
const routeBars = document.getElementById('routeBars');

let graphData = null;
let nodeLayout = [];
let currentActivations = [];
let traceTimer = null;
let traceGeneration = 0;
let hoveredNode = null;

function appendMessage(sender, content) {
  const msgDiv = document.createElement('div');
  msgDiv.className = `message ${sender}`;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = sender === 'assistant' ? '🪰' : '👤';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = content;

  msgDiv.appendChild(avatar);
  msgDiv.appendChild(bubble);
  chatMessages.appendChild(msgDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function createLoadingMessage() {
  const id = `loading-${Date.now()}`;

  const wrapper = document.createElement('div');
  wrapper.className = 'message assistant';
  wrapper.id = id;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = '🪰';

  const bubble = document.createElement('div');
  bubble.className = 'bubble loading';
  bubble.textContent = 'FlyGPT가 요청을 처리 중입니다...';

  wrapper.appendChild(avatar);
  wrapper.appendChild(bubble);
  chatMessages.appendChild(wrapper);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  return id;
}

function setBrainStatus(text, state = 'idle') {
  brainStatus.textContent = text;
  brainStatus.dataset.state = state;
}

function buildNodeLayout() {
  if (!graphData) return;

  const count = graphData.nodes.length;
  const leftCount = Math.ceil(count / 2);
  const rightCount = Math.floor(count / 2);
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));

  nodeLayout = graphData.nodes.map((node, index) => {
    const isRight = index % 2 === 1;
    const rank = Math.floor(index / 2);
    const lobeCount = isRight ? rightCount : leftCount;
    const progress = (rank + 0.5) / Math.max(lobeCount, 1);
    const angle = rank * goldenAngle + (isRight ? 0.45 : -0.45);
    const radius = Math.sqrt(progress);

    return {
      ...node,
      x: (isRight ? 0.655 : 0.345) + Math.cos(angle) * 0.245 * radius,
      y: 0.5 + Math.sin(angle) * 0.42 * radius,
    };
  });

  currentActivations = new Array(count).fill(0);
}

function canvasMetrics() {
  const rect = brainCanvas.getBoundingClientRect();
  return {
    width: Math.max(1, rect.width),
    height: Math.max(1, rect.height),
  };
}

function resizeCanvas() {
  const { width, height } = canvasMetrics();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);

  brainCanvas.width = Math.round(width * dpr);
  brainCanvas.height = Math.round(height * dpr);

  const ctx = brainCanvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  drawBrain();
}

function nodeScreenPosition(node, width, height) {
  return {
    x: node.x * width,
    y: node.y * height,
  };
}

function drawBrain() {
  const ctx = brainCanvas.getContext('2d');
  const { width, height } = canvasMetrics();

  ctx.clearRect(0, 0, width, height);

  const glow = ctx.createRadialGradient(
    width * 0.5,
    height * 0.5,
    0,
    width * 0.5,
    height * 0.5,
    Math.max(width, height) * 0.6,
  );
  glow.addColorStop(0, 'rgba(95, 216, 255, 0.08)');
  glow.addColorStop(1, 'rgba(95, 216, 255, 0)');
  ctx.fillStyle = glow;
  ctx.fillRect(0, 0, width, height);

  if (!graphData || !nodeLayout.length) return;

  const maxEdgeWeight = graphData.edges.reduce(
    (max, edge) => Math.max(max, edge.weight || 0),
    0.0001,
  );

  ctx.lineCap = 'round';

  for (const edge of graphData.edges) {
    const src = nodeLayout[edge.src];
    const dst = nodeLayout[edge.dst];
    if (!src || !dst) continue;

    const a = currentActivations[edge.src] || 0;
    const b = currentActivations[edge.dst] || 0;
    const activity = Math.max(a, b);
    const normalizedWeight = Math.min(1, (edge.weight || 0) / maxEdgeWeight);

    const p1 = nodeScreenPosition(src, width, height);
    const p2 = nodeScreenPosition(dst, width, height);

    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.lineWidth = 0.45 + normalizedWeight * 1.1 + activity * 0.8;
    ctx.strokeStyle = `rgba(95, 216, 255, ${0.055 + activity * 0.33})`;
    ctx.stroke();
  }

  nodeLayout.forEach((node, index) => {
    const activation = Math.max(0, Math.min(1, currentActivations[index] || 0));
    const { x, y } = nodeScreenPosition(node, width, height);
    const radius = 1.8 + activation * 5.8;

    if (activation > 0.08) {
      const halo = ctx.createRadialGradient(x, y, 0, x, y, radius * 3.6);
      halo.addColorStop(0, `rgba(255, 221, 87, ${0.48 * activation})`);
      halo.addColorStop(1, 'rgba(255, 221, 87, 0)');
      ctx.fillStyle = halo;
      ctx.beginPath();
      ctx.arc(x, y, radius * 3.6, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fillStyle = activation > 0.08
      ? `rgba(255, 221, 87, ${0.55 + activation * 0.45})`
      : 'rgba(95, 216, 255, 0.72)';
    ctx.fill();

    if (hoveredNode === index) {
      ctx.beginPath();
      ctx.arc(x, y, radius + 4, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
      ctx.lineWidth = 1.2;
      ctx.stroke();
    }
  });
}

function renderRoute(router) {
  if (!router) return;

  routeName.textContent = router.route;
  routeConfidence.textContent = `${(router.confidence * 100).toFixed(1)}%`;
  routeBars.innerHTML = '';

  (router.top_routes || []).forEach((item) => {
    const row = document.createElement('div');
    row.className = 'route-row';

    const head = document.createElement('div');
    head.className = 'route-row-head';

    const label = document.createElement('span');
    label.textContent = item.route;

    const value = document.createElement('span');
    value.textContent = `${(item.confidence * 100).toFixed(1)}%`;

    head.appendChild(label);
    head.appendChild(value);

    const track = document.createElement('div');
    track.className = 'route-track';

    const fill = document.createElement('div');
    fill.className = 'route-fill';
    fill.style.width = `${Math.max(2, item.confidence * 100)}%`;

    track.appendChild(fill);
    row.appendChild(head);
    row.appendChild(track);
    routeBars.appendChild(row);
  });
}

function applyTraceFrame(frame) {
  if (!frame || !Array.isArray(frame.activations)) return;

  currentActivations = frame.activations;
  stageChip.textContent = frame.stage.replace('_', ' ');
  drawBrain();
}

function animateTrace(trace) {
  if (!Array.isArray(trace) || trace.length === 0) return;

  traceGeneration += 1;
  const generation = traceGeneration;
  clearTimeout(traceTimer);

  let index = 0;
  setBrainStatus('THINKING', 'thinking');

  const next = () => {
    if (generation !== traceGeneration) return;

    applyTraceFrame(trace[index]);
    index += 1;

    if (index < trace.length) {
      traceTimer = setTimeout(next, 520);
    } else {
      setBrainStatus('ACTIVE', 'active');
      traceTimer = setTimeout(() => {
        if (generation === traceGeneration) {
          setBrainStatus('READY', 'ready');
        }
      }, 900);
    }
  };

  next();
}

function updateTooltip(event) {
  if (!graphData || !nodeLayout.length) return;

  const rect = brainCanvas.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;
  const px = event.clientX - rect.left;
  const py = event.clientY - rect.top;

  let bestIndex = -1;
  let bestDistance = 14;

  nodeLayout.forEach((node, index) => {
    const p = nodeScreenPosition(node, width, height);
    const distance = Math.hypot(px - p.x, py - p.y);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index;
    }
  });

  hoveredNode = bestIndex >= 0 ? bestIndex : null;
  drawBrain();

  if (bestIndex < 0) {
    brainTooltip.hidden = true;
    return;
  }

  const node = nodeLayout[bestIndex];
  const activation = currentActivations[bestIndex] || 0;
  const rootId = node.root_id || 'not mapped';

  brainTooltip.innerHTML =
    `<strong>Node ${bestIndex}</strong><br>` +
    `FlyWire ID: ${rootId}<br>` +
    `Model activation: ${(activation * 100).toFixed(1)}%`;
  brainTooltip.style.left = `${Math.min(px + 14, width - 205)}px`;
  brainTooltip.style.top = `${Math.max(8, py - 34)}px`;
  brainTooltip.hidden = false;
}

async function loadBrainGraph() {
  try {
    const response = await fetch('/api/router/graph');
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || `HTTP ${response.status}`);
    }

    graphData = data;
    buildNodeLayout();

    brainNodeCount.textContent = `${data.n_nodes} nodes`;
    brainEdgeCount.textContent = `${data.n_edges} edges`;
    brainStepCount.textContent = `${data.steps} steps`;
    brainEmpty.hidden = true;
    setBrainStatus('READY', 'ready');

    resizeCanvas();
  } catch (error) {
    setBrainStatus('OFFLINE', 'error');
    brainEmpty.hidden = false;
    brainEmpty.textContent =
      `뉴런 지도를 불러오지 못했습니다.\n${error.message}`;
  }
}

brainCanvas.addEventListener('pointermove', updateTooltip);
brainCanvas.addEventListener('pointerdown', updateTooltip);
brainCanvas.addEventListener('pointerleave', () => {
  hoveredNode = null;
  brainTooltip.hidden = true;
  drawBrain();
});

const resizeObserver = new ResizeObserver(() => resizeCanvas());
resizeObserver.observe(brainStage);

chatForm.addEventListener('submit', async (event) => {
  event.preventDefault();

  const text = userInput.value.trim();
  if (!text) return;

  userInput.value = '';
  userInput.disabled = true;
  sendBtn.disabled = true;

  appendMessage('user', text);
  const loadingId = createLoadingMessage();
  setBrainStatus('THINKING', 'thinking');

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message: text,
      }),
    });

    let data;

    try {
      data = await response.json();
    } catch {
      throw new Error(`HTTP ${response.status}`);
    }

    document.getElementById(loadingId)?.remove();

    if (!response.ok) {
      appendMessage(
        'assistant',
        `⚠️ 오류 발생\n${data.detail || '알 수 없는 오류가 발생했습니다.'}`,
      );
      setBrainStatus('ERROR', 'error');
      return;
    }

    appendMessage('assistant', data.answer);

    if (data.router) {
      renderRoute(data.router);
      animateTrace(data.router.trace);
    } else {
      setBrainStatus('READY', 'ready');
    }
  } catch (error) {
    document.getElementById(loadingId)?.remove();
    appendMessage(
      'assistant',
      `⚠️ 통신 오류\n서버와 연결할 수 없습니다.\n${error.message}`,
    );
    setBrainStatus('OFFLINE', 'error');
  } finally {
    userInput.disabled = false;
    sendBtn.disabled = false;
    userInput.focus();
  }
});

loadBrainGraph();
