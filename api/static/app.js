/* SuSchedule-r frontend — app.js */

// ── State ──────────────────────────────────────────────────────
let sessionId   = null;
let isLoading   = false;
let courseNames = {};   // code → title (filled from plan reasoning)
let currentMode = 'pipeline';   // 'pipeline' | 'react'

// ── DOM refs ───────────────────────────────────────────────────
const dropZone    = document.getElementById('drop-zone');
const dropLabel   = document.getElementById('drop-label');
const fileInput   = document.getElementById('file-input');
const studentInfo = document.getElementById('student-info');
const planList    = document.getElementById('plan-list');
const planCredits = document.getElementById('plan-credits');
const msgInput    = document.getElementById('msg-input');
const sendBtn     = document.getElementById('send-btn');
const chatMsgs    = document.getElementById('chat-messages');
const statsBody   = document.getElementById('stats-body');

// ── Mode toggle ────────────────────────────────────────────────
async function setMode(mode) {
  if (mode === currentMode) return;
  currentMode = mode;

  // Update button styles
  document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.mode === mode);
  });

  // If a session already exists, tear it down so the new mode takes effect
  // on the next transcript upload.
  if (sessionId) {
    await fetch(`/session/${sessionId}`, { method: 'DELETE' }).catch(() => {});
    sessionId = null;
    resetUI();
    addSystemMsg(`🔄 Switched to **${mode === 'react' ? 'ReAct' : 'Pipeline'}** mode. Please re-upload your transcript to start a new session.`);
  }
}

function resetUI() {
  document.getElementById('chat-messages').innerHTML = '';
  document.getElementById('plan-list').innerHTML = '<div class="plan-empty">No plan yet — chat to get started.</div>';
  document.getElementById('plan-credits').classList.add('hidden');
  document.getElementById('student-info').classList.add('hidden');
  document.getElementById('drop-zone').classList.remove('hidden');
  document.getElementById('drop-label').innerHTML = 'Drop transcript here<br/><small>JSON or PDF</small>';
  document.getElementById('drop-zone').style.pointerEvents = '';
  document.getElementById('stats-body').textContent = '—';
  courseNames = {};
}

// ── Session init ───────────────────────────────────────────────
async function initSession() {
  const res = await fetch('/session', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      term: '202601',
      min_credits: 12,
      max_credits: 21,
      target_credits: 17,
      mode: currentMode,
    }),
  });
  const data = await res.json();
  sessionId = data.session_id;
  updateStats({ target_term: data.term, term_label: data.term_label, mode: currentMode });
}

// ── File upload ────────────────────────────────────────────────
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) uploadTranscript(file);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) uploadTranscript(fileInput.files[0]);
});

async function uploadTranscript(file) {
  if (!sessionId) await initSession();

  dropLabel.textContent = '⏳ Uploading…';
  dropZone.style.pointerEvents = 'none';

  const form = new FormData();
  form.append('file', file);

  try {
    const res = await fetch(`/session/${sessionId}/transcript`, {
      method: 'POST',
      body: form,
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Upload failed');
    }
    const data = await res.json();
    showStudentInfo(data);
    addSystemMsg(`✅ Transcript loaded for **${data.name}** (${data.program}).
Required courses still needed: ${data.required_left.length > 0 ? data.required_left.join(', ') : 'none — almost done!'}.
What would you like to take this semester?`);

  } catch (err) {
    dropLabel.innerHTML = `❌ ${err.message}<br/><small>Drop to retry</small>`;
    dropZone.style.pointerEvents = '';
  }
}

function showStudentInfo(data) {
  dropZone.classList.add('hidden');
  studentInfo.classList.remove('hidden');
  studentInfo.innerHTML = `
    <div class="name">${data.name} <span class="tag">${data.program}</span></div>
    Semester ${data.semester} &nbsp;·&nbsp; CGPA ${data.cgpa.toFixed(2)}<br/>
    ${data.completed_count} completed &nbsp;·&nbsp;
    ${data.in_progress.length} in progress<br/>
    <span style="color:var(--accent2)">Required left: ${data.required_left.length} courses</span>
  `;
}

// ── Chat ───────────────────────────────────────────────────────
msgInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

// Auto-resize textarea
msgInput.addEventListener('input', () => {
  msgInput.style.height = 'auto';
  msgInput.style.height = Math.min(msgInput.scrollHeight, 140) + 'px';
});

function sendSuggestion(el) {
  msgInput.value = el.textContent;
  msgInput.dispatchEvent(new Event('input'));
  sendMessage();
}

async function sendMessage() {
  const text = msgInput.value.trim();
  if (!text || isLoading) return;
  if (!sessionId) await initSession();

  addUserMsg(text);
  msgInput.value = '';
  msgInput.style.height = 'auto';

  const typingEl = addTypingIndicator();
  setLoading(true);

  try {
    const res = await fetch(`/session/${sessionId}/turn`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Server error');
    }
    const data = await res.json();

    typingEl.remove();
    addAgentMsg(data.response, data.intent);

    if (data.plan) updatePlan(data.plan, data.total_credits, data.validation_ok, data.warnings);
    refreshStats(data.token_usage);

  } catch (err) {
    typingEl.remove();
    addAgentMsg(`⚠️ Error: ${err.message}`, 'error');
  } finally {
    setLoading(false);
  }
}

// ── Message rendering ──────────────────────────────────────────
function addUserMsg(text) {
  const div = document.createElement('div');
  div.className = 'msg user';
  div.innerHTML = `
    <div class="avatar">🎓</div>
    <div class="bubble">${escapeHtml(text)}</div>
  `;
  chatMsgs.appendChild(div);
  scrollBottom();
}

function addAgentMsg(text, intent) {
  const div = document.createElement('div');
  div.className = 'msg agent';
  let badge = '';
  if (intent && intent !== 'error') {
    const cls = intent === 'react' ? 'intent-badge react' : 'intent-badge';
    badge = `<span class="${cls}">${intent}</span>`;
  }
  div.innerHTML = `
    <div class="avatar">🤖</div>
    <div class="bubble">${badge}${marked.parse(text)}</div>
  `;
  chatMsgs.appendChild(div);
  scrollBottom();
}

function addSystemMsg(text) {
  const div = document.createElement('div');
  div.className = 'msg agent';
  div.innerHTML = `
    <div class="avatar">📋</div>
    <div class="bubble">${marked.parse(text)}</div>
  `;
  chatMsgs.appendChild(div);
  scrollBottom();
}

function addTypingIndicator() {
  const div = document.createElement('div');
  div.className = 'msg agent';
  div.innerHTML = `
    <div class="avatar">🤖</div>
    <div class="bubble typing"><span></span><span></span><span></span></div>
  `;
  chatMsgs.appendChild(div);
  scrollBottom();
  return div;
}

// ── Plan sidebar ───────────────────────────────────────────────
function updatePlan(plan, credits, validationOk, warnings) {
  if (!plan || plan.length === 0) {
    planList.innerHTML = '<div class="plan-empty">No plan yet.</div>';
    planCredits.classList.add('hidden');
    return;
  }

  planList.innerHTML = plan.map(code => {
    const title = courseNames[code] || '';
    return `
      <div class="plan-item">
        <span class="code">${code}</span>
        <span class="title-text">${title}</span>
      </div>
    `;
  }).join('');

  planCredits.classList.remove('hidden');
  const statusClass = validationOk ? 'ok' : 'warn';
  const statusIcon  = validationOk ? '✓' : '⚠';
  planCredits.innerHTML = `
    <span class="${statusClass}">${statusIcon} ${credits?.toFixed(1)} credits</span>
  `;

  if (warnings && warnings.length) {
    planCredits.innerHTML += `<br/><small style="color:var(--text-faint)">${warnings[0]}</small>`;
  }
}

// Extract course names from reasoning markdown for the plan sidebar
function extractCourseNames(responseText) {
  // e.g. "• **CS 306** — Required course..."
  const re = /\*\*([A-Z]{2,5}\s+\d+[A-Z]?)\*\*(?:\s*—\s*(.+?))?(?:\n|$)/g;
  let m;
  while ((m = re.exec(responseText)) !== null) {
    // try to get title from the session state endpoint instead
    courseNames[m[1]] = '';
  }
}

// ── Stats ──────────────────────────────────────────────────────
function updateStats(data) {
  const rows = [];
  if (data.term_label) rows.push(['Term', data.term_label]);
  const modeLabel = (data.mode || currentMode) === 'react' ? '🔮 ReAct' : '⚙️ Pipeline';
  rows.push(['Mode', modeLabel]);
  if (data.eligible_pool_size != null) rows.push(['Eligible pool', data.eligible_pool_size]);
  if (data.history_turns != null) rows.push(['Turns', data.history_turns]);
  if (data.total_tokens != null) rows.push(['Tokens used', data.total_tokens.toLocaleString()]);
  if (data.this_turn != null) rows.push(['This turn', data.this_turn.toLocaleString()]);

  if (rows.length === 0) { statsBody.textContent = '—'; return; }
  statsBody.innerHTML = rows.map(([k, v]) =>
    `<div class="stat-row"><span>${k}</span><span class="stat-val">${v}</span></div>`
  ).join('');
}

async function refreshStats(tokenUsage) {
  if (!sessionId) return;
  try {
    const res = await fetch(`/session/${sessionId}/state`);
    const data = await res.json();
    updateStats({
      term_label: data.term_label,
      eligible_pool_size: data.eligible_pool_size,
      history_turns: data.history_turns,
      total_tokens: tokenUsage?.total_tokens,
      this_turn: tokenUsage?.this_turn,
      mode: currentMode,
    });
    // Enrich plan sidebar with title info from reasoning
    if (data.reasoning) {
      Object.entries(data.reasoning).forEach(([code, reason]) => {
        const m = reason.match(/^(.{0,40})/);
        courseNames[code] = '';   // title lookup via reasoning snippets
      });
    }
  } catch (_) {}
}

// ── Helpers ────────────────────────────────────────────────────
function setLoading(val) {
  isLoading = val;
  sendBtn.disabled = val;
  sendBtn.textContent = val ? '…' : 'Send';
}

function scrollBottom() {
  chatMsgs.scrollTop = chatMsgs.scrollHeight;
}

function escapeHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── Boot ───────────────────────────────────────────────────────
(async () => {
  await initSession();
})();
