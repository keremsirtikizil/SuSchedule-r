/* SuSchedule-r frontend */

let sessionId = null;
let isLoading = false;
let courseNames = {};
const currentMode = 'react';

const dropZone = document.getElementById('drop-zone');
const dropLabel = document.getElementById('drop-label');
const fileInput = document.getElementById('file-input');
const studentInfo = document.getElementById('student-info');
const planList = document.getElementById('plan-list');
const planCredits = document.getElementById('plan-credits');
const msgInput = document.getElementById('msg-input');
const sendBtn = document.getElementById('send-btn');
const chatMsgs = document.getElementById('chat-messages');
const statsBody = document.getElementById('stats-body');
const traceBody = document.getElementById('trace-body');

function resetUI() {
  chatMsgs.innerHTML = '';
  planList.innerHTML = '<div class="plan-empty">No plan yet - chat to get started.</div>';
  planCredits.classList.add('hidden');
  studentInfo.classList.add('hidden');
  dropZone.classList.remove('hidden');
  dropLabel.innerHTML = 'Optional: drop transcript<br/><small>JSON, PDF, or degree-evaluation HTML for exact eligibility</small>';
  dropZone.style.pointerEvents = '';
  statsBody.textContent = '-';
  traceBody.textContent = 'No trace yet.';
  courseNames = {};
}

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

  if (!res.ok) {
    throw new Error('Could not create session');
  }

  const data = await res.json();
  sessionId = data.session_id;
  updateStats({ target_term: data.term, term_label: data.term_label });
}

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
  dropLabel.textContent = 'Uploading...';
  dropZone.style.pointerEvents = 'none';

  const form = new FormData();
  form.append('file', file);

  try {
    const res = await fetch(`/session/${sessionId}/transcript`, { method: 'POST', body: form });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Upload failed');
    }

    const data = await res.json();
    showStudentInfo(data);

    const requiredLeft = Array.isArray(data.required_left) ? data.required_left : [];
    addSystemMsg(`Transcript loaded for **${escapeHtml(data.name)}** (${escapeHtml(data.program)}).
Required courses still needed: ${requiredLeft.length > 0 ? requiredLeft.map(escapeHtml).join(', ') : 'none - almost done'}.
What would you like to take this semester?`);
  } catch (err) {
    dropLabel.innerHTML = `${escapeHtml(err.message)}<br/><small>Drop to retry</small>`;
    dropZone.style.pointerEvents = '';
  }
}

function showStudentInfo(data) {
  const inProgress = Array.isArray(data.in_progress) ? data.in_progress : [];
  const requiredLeft = Array.isArray(data.required_left) ? data.required_left : [];

  dropZone.classList.add('hidden');
  studentInfo.classList.remove('hidden');
  studentInfo.innerHTML = `
    <div class="name">${escapeHtml(data.name)} <span class="tag">${escapeHtml(data.program)}</span></div>
    Semester ${escapeHtml(data.semester ?? 0)} &nbsp;-&nbsp; CGPA ${Number(data.cgpa || 0).toFixed(2)}<br/>
    ${escapeHtml(data.completed_count ?? 0)} completed &nbsp;-&nbsp;
    ${inProgress.length} in progress<br/>
    <span style="color:var(--accent2)">Required left: ${requiredLeft.length} courses</span>
  `;
}

msgInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

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
    renderTrace(data.trace || []);
    if (data.plan) updatePlan(data.plan, data.total_credits, data.validation_ok, data.warnings);
    refreshStats(data.token_usage);
  } catch (err) {
    typingEl.remove();
    addAgentMsg(`Error: ${escapeHtml(err.message)}`, 'error');
  } finally {
    setLoading(false);
  }
}

function addUserMsg(text) {
  const div = document.createElement('div');
  div.className = 'msg user';
  div.innerHTML = `<div class="avatar">You</div><div class="bubble">${escapeHtml(text)}</div>`;
  chatMsgs.appendChild(div);
  scrollBottom();
}

function addAgentMsg(text, intent) {
  const div = document.createElement('div');
  div.className = 'msg agent';
  let badge = '';
  if (intent && intent !== 'error') {
    const cls = intent === 'react' ? 'intent-badge react' : 'intent-badge';
    badge = `<span class="${cls}">${escapeHtml(intent)}</span>`;
  }
  div.innerHTML = `<div class="avatar">AI</div><div class="bubble">${badge}${marked.parse(text || '')}</div>`;
  chatMsgs.appendChild(div);
  scrollBottom();
}

function addSystemMsg(text) {
  const div = document.createElement('div');
  div.className = 'msg agent';
  div.innerHTML = `<div class="avatar">Info</div><div class="bubble">${marked.parse(text || '')}</div>`;
  chatMsgs.appendChild(div);
  scrollBottom();
}

function addTypingIndicator() {
  const div = document.createElement('div');
  div.className = 'msg agent';
  div.innerHTML = '<div class="avatar">AI</div><div class="bubble typing"><span></span><span></span><span></span></div>';
  chatMsgs.appendChild(div);
  scrollBottom();
  return div;
}

function updatePlan(plan, credits, validationOk, warnings) {
  if (!plan || plan.length === 0) {
    planList.innerHTML = '<div class="plan-empty">No plan yet.</div>';
    planCredits.classList.add('hidden');
    return;
  }

  planList.innerHTML = plan.map(code => `
    <div class="plan-item">
      <span class="code">${escapeHtml(code)}</span>
      <span class="title-text">${escapeHtml(courseNames[code] || '')}</span>
    </div>
  `).join('');

  planCredits.classList.remove('hidden');
  const statusClass = validationOk ? 'ok' : 'warn';
  const statusIcon = validationOk ? 'OK' : 'WARN';
  planCredits.innerHTML = `<span class="${statusClass}">${statusIcon} ${Number(credits || 0).toFixed(1)} credits</span>`;
  if (warnings && warnings.length) {
    planCredits.innerHTML += `<br/><small style="color:var(--text-faint)">${escapeHtml(warnings[0])}</small>`;
  }
}

function updateStats(data) {
  const rows = [];
  if (data.term_label) rows.push(['Term', data.term_label]);
  rows.push(['Advisor', 'RAG + tools']);
  if (data.eligible_pool_size != null) rows.push(['Eligible pool', data.eligible_pool_size]);
  if (data.history_turns != null) rows.push(['Turns', data.history_turns]);
  if (data.total_tokens != null) rows.push(['Tokens used', Number(data.total_tokens).toLocaleString()]);
  if (data.this_turn != null) rows.push(['This turn', Number(data.this_turn).toLocaleString()]);

  if (rows.length === 0) {
    statsBody.textContent = '-';
    return;
  }
  statsBody.innerHTML = rows.map(([k, v]) =>
    `<div class="stat-row"><span>${escapeHtml(k)}</span><span class="stat-val">${escapeHtml(v)}</span></div>`
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
    });
    if (data.last_trace) renderTrace(data.last_trace);
    if (data.reasoning) {
      Object.keys(data.reasoning).forEach(code => { courseNames[code] = ''; });
    }
  } catch (_) {
    // Session stats are supplemental; keep chat usable if this refresh fails.
  }
}

function renderTrace(trace) {
  if (!trace || trace.length === 0) {
    traceBody.textContent = 'No trace yet.';
    return;
  }

  const rows = [];
  trace.forEach(ev => {
    if (ev.type === 'turn_start') {
      rows.push(`<div class="trace-item"><div class="trace-title">Turn start</div><div class="trace-meta">${escapeHtml(ev.user_message || '')}</div></div>`);
    } else if (ev.type === 'assistant_step') {
      const calls = (ev.tool_calls || []).map(tc => {
        let args = tc.arguments || '{}';
        try { args = JSON.stringify(JSON.parse(args), null, 2); } catch (_) {}
        return `<div class="trace-call">${escapeHtml(tc.name || '')}<pre>${escapeHtml(args)}</pre></div>`;
      }).join('');
      rows.push(`<div class="trace-item"><div class="trace-title">Step ${escapeHtml(ev.step)}: tool decision</div>${calls || '<div class="trace-meta">Final answer step</div>'}</div>`);
    } else if (ev.type === 'tool_result' && ev.name === 'rewrite_retrieval_queries') {
      const queries = ev.result?.queries || [];
      rows.push(`<div class="trace-item"><div class="trace-title">Rewritten retrieval queries</div><div class="trace-meta">${queries.map(escapeHtml).join(' | ')}</div></div>`);
    } else if (ev.type === 'selected_graphs') {
      const counts = ev.section_counts || {};
      const countText = Object.entries(counts).map(([section, count]) => `${escapeHtml(section)}: ${escapeHtml(count)}`).join(' | ');
      rows.push(`
        <div class="trace-item">
          <div class="trace-title">Selected degree graphs</div>
          <div class="trace-meta">program: ${escapeHtml(ev.program)} | cohort: ${escapeHtml(ev.cohort_term)}</div>
          <div class="trace-meta">${countText}</div>
        </div>
      `);
    } else if (ev.type === 'requirements') {
      const required = (ev.required_left || []).slice(0, 12).map(escapeHtml).join(', ');
      rows.push(`
        <div class="trace-item">
          <div class="trace-title">Requirement state</div>
          <div class="trace-meta">program: ${escapeHtml(ev.program)} | cohort: ${escapeHtml(ev.cohort_term)} | required credits left: ${escapeHtml(ev.required_credits_left ?? '-')}</div>
          <div class="trace-meta">required left (${escapeHtml(ev.required_left_count ?? 0)}): ${required || 'none'}</div>
          <div class="trace-meta">core left: ${escapeHtml(ev.core_left_count ?? 0)} | area left: ${escapeHtml(ev.area_left_count ?? 0)} | free left: ${escapeHtml(ev.free_left_count ?? 0)}</div>
        </div>
      `);
    } else if (ev.type === 'retrieval') {
      const filters = ev.filters || {};
      const hits = (ev.merged_results || []).slice(0, 10).map(h =>
        `<li><span>${escapeHtml(h.code || '')}</span> ${escapeHtml(h.title || '')}<small>${h.rerank_score != null ? `rerank ${escapeHtml(h.rerank_score)}` : `score ${Number(h.score || 0).toFixed(2)}`}</small></li>`
      ).join('');
      rows.push(`
        <div class="trace-item trace-retrieval">
          <div class="trace-title">Retrieval</div>
          <div class="trace-meta">queries: ${(ev.queries || []).map(escapeHtml).join(' | ')}</div>
          <div class="trace-meta">section: ${escapeHtml(filters.section || 'auto')} | pool: ${escapeHtml(filters.candidate_pool_size ?? 'full')} | subj: ${escapeHtml((filters.subj || ['all']).join(', '))}</div>
          <ol>${hits}</ol>
        </div>
      `);
    } else if (ev.type === 'section_courses') {
      const hits = (ev.courses || []).slice(0, 10).map(h =>
        `<li><span>${escapeHtml(h.code || '')}</span> ${escapeHtml(h.title || '')}</li>`
      ).join('');
      rows.push(`
        <div class="trace-item">
          <div class="trace-title">Graph section lookup</div>
          <div class="trace-meta">${escapeHtml((ev.sections || []).join(', '))} | ${escapeHtml(ev.retrieval_query || ev.query || 'no topic filter')}</div>
          <ol>${hits}</ol>
        </div>
      `);
    } else if (ev.type === 'tool_result' && ['get_student_profile', 'select_degree_graphs', 'get_requirement_state'].includes(ev.name)) {
      const result = ev.result || {};
      let summary = '';
      if (ev.name === 'get_student_profile') {
        summary = `program: ${escapeHtml(result.program || '-')} | admit: ${escapeHtml(result.admit_term || '-')} | transcript: ${escapeHtml(result.transcript_loaded)}`;
      } else if (ev.name === 'select_degree_graphs') {
        const counts = result.section_counts || {};
        summary = `program: ${escapeHtml(result.program || '-')} | cohort: ${escapeHtml(result.cohort_term || '-')} | ${Object.entries(counts).map(([section, count]) => `${escapeHtml(section)}: ${escapeHtml(count)}`).join(' | ')}`;
      } else {
        summary = `program: ${escapeHtml(result.program || '-')} | cohort: ${escapeHtml(result.cohort_term || '-')} | required left: ${escapeHtml((result.required_left || []).length)}`;
      }
      rows.push(`<div class="trace-item"><div class="trace-title">${escapeHtml(ev.name)} result</div><div class="trace-meta">${summary}</div></div>`);
    } else if (ev.type === 'tool_result' && ev.result?.error) {
      rows.push(`<div class="trace-item trace-error"><div class="trace-title">${escapeHtml(ev.name || 'tool')} error</div><div class="trace-meta">${escapeHtml(ev.result.error)}</div></div>`);
    } else if (ev.type === 'final_response') {
      rows.push(`<div class="trace-item"><div class="trace-title">Final response</div><div class="trace-meta">${escapeHtml((ev.content || '').slice(0, 220))}</div></div>`);
    }
  });

  traceBody.innerHTML = rows.join('') + `
    <details class="trace-raw">
      <summary>Raw JSON</summary>
      <pre>${escapeHtml(JSON.stringify(trace, null, 2))}</pre>
    </details>
  `;
}

function setLoading(val) {
  isLoading = val;
  sendBtn.disabled = val;
  sendBtn.textContent = val ? '...' : 'Send';
}

function scrollBottom() {
  chatMsgs.scrollTop = chatMsgs.scrollHeight;
}

function escapeHtml(str) {
  str = String(str ?? '');
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

(async () => {
  try {
    await initSession();
  } catch (err) {
    addAgentMsg(`Error: ${escapeHtml(err.message)}`, 'error');
  }
})();
