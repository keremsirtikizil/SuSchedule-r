/* SuSchedule-r frontend — chat (left) + schedule builder (right) */

let sessionId = null;
let isLoading = false;
let courseNames = {};
const currentMode = 'react';

// Client-side schedule the student is assembling in the builder. Each pick:
// { code, title, su_credit, crn, section, meetings:[...], when, sections:[...],
//   offered, is_corequisite?, corequisite_for? }
let schedule = [];

// ── Chat-pane elements ──────────────────────────────────────────
const uploadBtn = document.getElementById('upload-btn');
const fileInput = document.getElementById('file-input');
const uploadStatus = document.getElementById('upload-status');
const studentInfo = document.getElementById('student-info');
const chatMsgs = document.getElementById('chat-messages');
const msgInput = document.getElementById('msg-input');
const sendBtn = document.getElementById('send-btn');
const planList = document.getElementById('plan-list');
const planCredits = document.getElementById('plan-credits');
const statsBody = document.getElementById('stats-body');
const traceBody = document.getElementById('trace-body');

// ── Builder-pane elements ───────────────────────────────────────
const builderTerm = document.getElementById('builder-term');
const courseInput = document.getElementById('course-input');
const addCourseBtn = document.getElementById('add-course-btn');
const builderNote = document.getElementById('builder-note');
const scheduleGrid = document.getElementById('schedule-grid');
const courseList = document.getElementById('course-list');
const builderSummary = document.getElementById('builder-summary');
const checkScheduleBtn = document.getElementById('check-schedule-btn');

// ════════════════════════ Session ════════════════════════
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
  if (!res.ok) throw new Error('Could not create session');

  const data = await res.json();
  sessionId = data.session_id;
  if (builderTerm) builderTerm.textContent = data.term_label || '';
  updateStats({ target_term: data.term, term_label: data.term_label });
}

// ════════════════════════ Transcript upload ════════════════════════
uploadBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) uploadTranscript(fileInput.files[0]);
});

async function uploadTranscript(file) {
  if (!sessionId) await initSession();
  uploadStatus.classList.remove('hidden', 'err');
  uploadStatus.textContent = `Uploading ${file.name}…`;
  uploadBtn.disabled = true;

  const form = new FormData();
  form.append('file', file);

  try {
    const res = await fetch(`/session/${sessionId}/transcript`, { method: 'POST', body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Upload failed');
    }

    const data = await res.json();
    uploadStatus.classList.add('hidden');
    showStudentInfo(data);

    const requiredLeft = Array.isArray(data.required_left) ? data.required_left : [];
    addSystemMsg(`Transcript loaded for **${escapeHtml(data.name)}** (${escapeHtml(data.program)}).
Required courses still needed: ${requiredLeft.length ? requiredLeft.map(escapeHtml).join(', ') : 'none — almost done'}.
What would you like to take this semester?`);
  } catch (err) {
    uploadStatus.classList.remove('hidden');
    uploadStatus.classList.add('err');
    uploadStatus.textContent = err.message;
  } finally {
    uploadBtn.disabled = false;
  }
}

function showStudentInfo(data) {
  const inProgress = Array.isArray(data.in_progress) ? data.in_progress : [];
  const requiredLeft = Array.isArray(data.required_left) ? data.required_left : [];

  studentInfo.classList.remove('hidden');
  studentInfo.innerHTML = `
    <div class="name">${escapeHtml(data.name)} <span class="tag">${escapeHtml(data.program)}</span></div>
    Semester ${escapeHtml(data.semester ?? 0)} · CGPA ${Number(data.cgpa || 0).toFixed(2)}<br/>
    ${escapeHtml(data.completed_count ?? 0)} completed · ${inProgress.length} in progress<br/>
    <span style="color:var(--accent2)">Required left: ${requiredLeft.length} courses</span>
  `;
  uploadBtn.textContent = 'Replace transcript';
}

// ════════════════════════ Chat ════════════════════════
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
      const err = await res.json().catch(() => ({}));
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
    planList.innerHTML = '<div class="plan-empty">No committed plan yet.</div>';
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

// ════════════════════════ Schedule builder ════════════════════════
const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'];
const PX_PER_MIN = 0.82;
const TOP_PAD = 10; // px of breathing room above/below the timeline
const PALETTE = ['#1c47a8', '#2f7fb8', '#1f8f6a', '#6a5cb0', '#b8588c', '#c08a3a', '#3aa0a0', '#5a6bd0'];

function courseColor(code) {
  let h = 0;
  for (let i = 0; i < code.length; i++) h = (h * 31 + code.charCodeAt(i)) >>> 0;
  return PALETTE[h % PALETTE.length];
}

// "cs412" / "cs 412" / "CS412" → "CS 412"
function normalizeCode(raw) {
  let s = (raw || '').toUpperCase().trim().replace(/\s+/g, ' ');
  const m = s.match(/^([A-Z]+)\s?(\d+[A-Z]*)$/);
  return m ? `${m[1]} ${m[2]}` : s;
}

function fmtHour(mins) {
  return `${String(Math.floor(mins / 60)).padStart(2, '0')}:00`;
}

addCourseBtn.addEventListener('click', addCourse);
courseInput.addEventListener('keydown', e => {
  if (e.key === 'Enter') {
    e.preventDefault();
    addCourse();
  }
});

async function addCourse() {
  const code = normalizeCode(courseInput.value);
  if (!code) return;
  if (!sessionId) await initSession();

  if (schedule.some(p => p.code === code)) {
    showBuilderNote(`${code} is already in your schedule.`, true);
    return;
  }

  addCourseBtn.disabled = true;
  addCourseBtn.textContent = '…';
  try {
    const res = await fetch(`/session/${sessionId}/course_sections?code=${encodeURIComponent(code)}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Lookup failed');
    }
    const data = await res.json();

    if (!data.sections || !data.sections.length) {
      const label = data.title ? `${code} — ${data.title}` : code;
      showBuilderNote(`${label} isn't offered in the schedule term, so it has no time slots to place. ${data.note || ''}`.trim(), true);
      return;
    }

    const first = data.sections[0];
    schedule.push({
      code: data.code,
      title: data.title || first.title || '',
      su_credit: data.su_credit,
      crn: first.crn,
      section: first.section,
      meetings: first.meetings || [],
      when: first.when || '',
      sections: data.sections,
      offered: true,
    });
    courseNames[data.code] = data.title || '';

    const addedCoreqs = [];
    (data.corequisites || []).forEach(coreq => {
      if (!coreq.code || schedule.some(p => p.code === coreq.code)) return;
      const coreqFirst = (coreq.sections || [])[0];
      if (!coreqFirst) {
        addedCoreqs.push(`${coreq.code} (not offered / no time slot)`);
        return;
      }
      schedule.push({
        code: coreq.code,
        title: coreq.title || coreqFirst.title || '',
        su_credit: coreq.su_credit || 0,
        crn: coreqFirst.crn,
        section: coreqFirst.section,
        meetings: coreqFirst.meetings || [],
        when: coreqFirst.when || '',
        sections: coreq.sections || [],
        offered: true,
        is_corequisite: true,
        corequisite_for: data.code,
      });
      courseNames[coreq.code] = coreq.title || coreqFirst.title || '';
      addedCoreqs.push(coreq.code);
    });
    courseInput.value = '';

    const notes = [];
    if (addedCoreqs.length) notes.push(`Also added required recitation/lab: ${addedCoreqs.join(', ')}.`);
    if (data.proxy_term_used && data.note) notes.push(data.note);
    if (notes.length) showBuilderNote(notes.join(' '), false);
    else hideBuilderNote();

    renderBuilder();
    persistSchedule();
  } catch (err) {
    showBuilderNote(err.message, true);
  } finally {
    addCourseBtn.disabled = false;
    addCourseBtn.textContent = 'Add';
  }
}

function changeSection(idx, crn) {
  const p = schedule[idx];
  if (!p) return;
  const sec = (p.sections || []).find(s => String(s.crn) === String(crn));
  if (!sec) return;
  p.crn = sec.crn;
  p.section = sec.section;
  p.meetings = sec.meetings || [];
  p.when = sec.when || '';
  renderBuilder();
  persistSchedule();
}

function removeCourse(idx) {
  const removed = schedule[idx];
  if (!removed) return;
  if (removed.is_corequisite && schedule.some(p => p.code === removed.corequisite_for)) {
    showBuilderNote(`${removed.code} is a required recitation/lab for ${removed.corequisite_for}. Remove ${removed.corequisite_for} first.`, true);
    return;
  }
  schedule = schedule.filter((p, i) => i !== idx && p.corequisite_for !== removed.code);
  renderBuilder();
  persistSchedule();
}

// Detect overlapping meetings across different courses (Mon–Fri).
function computeConflicts() {
  const conflictRows = new Set();
  const conflictKeys = new Set(); // `${idx}|${day}|${start}`
  const intervals = [];
  schedule.forEach((p, idx) => {
    (p.meetings || []).forEach(m => {
      if (m.start == null || m.end == null) return;
      (m.days || []).forEach(d => {
        if (d >= 0 && d <= 4) intervals.push({ idx, day: d, start: m.start, end: m.end });
      });
    });
  });
  for (let i = 0; i < intervals.length; i++) {
    for (let j = i + 1; j < intervals.length; j++) {
      const a = intervals[i], b = intervals[j];
      if (a.idx === b.idx || a.day !== b.day) continue;
      if (a.start < b.end && b.start < a.end) {
        conflictRows.add(a.idx);
        conflictRows.add(b.idx);
        conflictKeys.add(`${a.idx}|${a.day}|${a.start}`);
        conflictKeys.add(`${b.idx}|${b.day}|${b.start}`);
      }
    }
  }
  return { conflictRows, conflictKeys };
}

function renderGrid(conflictKeys) {
  let minStart = 9 * 60, maxEnd = 17 * 60;
  schedule.forEach(p => (p.meetings || []).forEach(m => {
    if (m.start != null) minStart = Math.min(minStart, m.start);
    if (m.end != null) maxEnd = Math.max(maxEnd, m.end);
  }));
  const rangeStart = Math.floor(minStart / 60) * 60;
  const rangeEnd = Math.ceil(maxEnd / 60) * 60;
  const height = (rangeEnd - rangeStart) * PX_PER_MIN + TOP_PAD * 2;
  const yOf = mins => TOP_PAD + (mins - rangeStart) * PX_PER_MIN;

  let html = '<div class="grid-col-head corner"></div>';
  DAYS.forEach(d => { html += `<div class="grid-col-head">${d}</div>`; });

  // Hour axis (labels aligned to each hour line).
  html += `<div class="grid-axis" style="height:${height}px">`;
  for (let h = rangeStart; h <= rangeEnd; h += 60) {
    html += `<div class="hour-label" style="top:${yOf(h)}px">${fmtHour(h)}</div>`;
  }
  html += '</div>';

  DAYS.forEach((dname, d) => {
    let col = `<div class="grid-day" style="height:${height}px">`;
    for (let h = rangeStart; h <= rangeEnd; h += 30) {
      const cls = h % 60 === 0 ? 'hour-line' : 'hour-line half';
      col += `<div class="${cls}" style="top:${yOf(h)}px"></div>`;
    }

    // Collect this day's meeting instances.
    const items = [];
    schedule.forEach((p, idx) => {
      (p.meetings || []).forEach(m => {
        if (m.start == null || m.end == null || !(m.days || []).includes(d)) return;
        items.push({ p, m, start: m.start, end: m.end, bad: conflictKeys.has(`${idx}|${d}|${m.start}`) });
      });
    });
    items.sort((a, b) => a.start - b.start || a.end - b.end);

    // Calendar-style layout: split overlapping blocks into side-by-side lanes.
    let i = 0;
    while (i < items.length) {
      let clusterEnd = items[i].end, j = i + 1;
      while (j < items.length && items[j].start < clusterEnd) {
        clusterEnd = Math.max(clusterEnd, items[j].end);
        j++;
      }
      const cluster = items.slice(i, j);
      const laneEnds = [];
      cluster.forEach(it => {
        let lane = laneEnds.findIndex(end => end <= it.start);
        if (lane === -1) { lane = laneEnds.length; laneEnds.push(it.end); }
        else laneEnds[lane] = it.end;
        it.lane = lane;
      });
      cluster.forEach(it => { it.lanes = laneEnds.length; });
      i = j;
    }

    items.forEach(it => {
      const { p, m, bad, lane = 0, lanes = 1 } = it;
      const color = courseColor(p.code);
      const top = yOf(it.start);
      const bh = Math.max((it.end - it.start) * PX_PER_MIN, 17);
      const leftPct = (lane / lanes) * 100;
      const widthPct = 100 / lanes;
      const tip = `${p.code} · ${p.title || ''}\n${m.start_label || ''}–${m.end_label || ''}` +
        (m.where ? `\n${m.where}` : '') + (m.instructor ? `\n${m.instructor}` : '');
      col += `<div class="grid-block${bad ? ' conflict' : ''}" style="top:${top}px;height:${bh}px;left:calc(${leftPct}% + 3px);width:calc(${widthPct}% - 5px);--blk:${color}" title="${escapeHtml(tip)}">
        <span class="blk-code">${escapeHtml(p.code)}</span>
        <span class="blk-time">${escapeHtml(m.start_label || '')}–${escapeHtml(m.end_label || '')}</span>
        ${bh >= 46 && m.where ? `<span class="blk-where">${escapeHtml(m.where)}</span>` : ''}
      </div>`;
    });

    col += '</div>';
    html += col;
  });

  scheduleGrid.innerHTML = html;
}

function renderCourseList(conflictRows) {
  if (!schedule.length) {
    courseList.innerHTML = '<div class="course-list-empty">No courses yet — add one above to build your weekly schedule.</div>';
    return;
  }
  courseList.innerHTML = schedule.map((p, idx) => {
    const color = courseColor(p.code);
    const conflicted = conflictRows.has(idx);

    let sectionEl = '';
    if (p.sections && p.sections.length > 1) {
      const opts = p.sections.map(s =>
        `<option value="${escapeHtml(s.crn)}"${String(s.crn) === String(p.crn) ? ' selected' : ''}>§${escapeHtml(s.section || '?')}${s.when ? ' · ' + escapeHtml(s.when) : ''}</option>`
      ).join('');
      sectionEl = `<select class="cr-section" onchange="changeSection(${idx}, this.value)">${opts}</select>`;
    } else if (p.section) {
      sectionEl = `<span class="cr-credit">§${escapeHtml(p.section)}</span>`;
    }

    const credit = p.su_credit != null ? `${Number(p.su_credit).toFixed(1)} cr` : '';
    const coreqNote = p.is_corequisite
      ? `<div class="cr-warn">required recitation/lab for ${escapeHtml(p.corequisite_for || '')}</div>`
      : '';
    const when = p.when
      ? `<div class="cr-when">${escapeHtml(p.when)}</div>`
      : '<div class="cr-warn">No set meeting time (TBA)</div>';

    return `
      <div class="course-row${conflicted ? ' conflict' : ''}" style="border-left-color:${color}">
        <span class="cr-code">${escapeHtml(p.code)}</span>
        <div class="cr-mid">
          <div class="cr-title">${escapeHtml(p.title || '')}</div>
          ${coreqNote}
          ${when}
          ${conflicted ? '<div class="cr-warn">⚠ time conflict</div>' : ''}
        </div>
        ${sectionEl}
        <span class="cr-credit">${escapeHtml(credit)}</span>
        <button class="cr-remove" onclick="removeCourse(${idx})" title="Remove">×</button>
      </div>`;
  }).join('');
}

function renderSummary(conflictRows) {
  const totalCredits = schedule.reduce((s, p) => s + (Number(p.su_credit) || 0), 0);
  let txt = `${schedule.length} course${schedule.length === 1 ? '' : 's'} · ${totalCredits.toFixed(1)} SU credits`;
  if (conflictRows.size) {
    txt += ` · <span class="warn">${conflictRows.size} with conflicts</span>`;
  }
  builderSummary.innerHTML = txt;
  checkScheduleBtn.disabled = schedule.length === 0 || isLoading;
}

function renderBuilder() {
  const { conflictRows, conflictKeys } = computeConflicts();
  if (!schedule.length) {
    scheduleGrid.innerHTML = '<div class="grid-empty"><span class="ge-icon">▦</span>Your weekly grid is empty.<br/>Add a course above to see its time slots.</div>';
  } else {
    renderGrid(conflictKeys);
  }
  renderCourseList(conflictRows);
  renderSummary(conflictRows);
}

function showBuilderNote(msg, isErr) {
  builderNote.textContent = msg;
  builderNote.classList.remove('hidden');
  builderNote.classList.toggle('err', !!isErr);
}
function hideBuilderNote() {
  builderNote.classList.add('hidden');
}

// Keep the server's copy of the schedule in sync (best-effort, no LLM call).
async function persistSchedule() {
  if (!sessionId) return;
  try {
    await fetch(`/session/${sessionId}/schedule`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ picks: schedule.map(p => ({ code: p.code, crn: p.crn })) }),
    });
  } catch (_) {
    // Sync is supplemental; check_schedule re-sends the picks anyway.
  }
}

async function checkSchedule() {
  if (isLoading) return;
  if (!schedule.length) {
    showBuilderNote('Add at least one course before checking.', true);
    return;
  }
  if (!sessionId) await initSession();

  addUserMsg('Check my schedule');
  const typingEl = addTypingIndicator();
  setLoading(true);

  try {
    const res = await fetch(`/session/${sessionId}/check_schedule`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ picks: schedule.map(p => ({ code: p.code, crn: p.crn })) }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
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

// ════════════════════════ Session stats + trace ════════════════════════
function updateStats(data) {
  const rows = [];
  if (data.term_label) rows.push(['Term', data.term_label]);
  rows.push(['Advisor', 'RAG + tools']);
  if (data.eligible_pool_size != null) rows.push(['Eligible pool', data.eligible_pool_size]);
  if (data.history_turns != null) rows.push(['Turns', data.history_turns]);
  if (data.total_tokens != null) rows.push(['Tokens used', Number(data.total_tokens).toLocaleString()]);
  if (data.this_turn != null) rows.push(['This turn', Number(data.this_turn).toLocaleString()]);

  if (rows.length === 0) {
    statsBody.textContent = '—';
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
      Object.keys(data.reasoning).forEach(code => { courseNames[code] = courseNames[code] || ''; });
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
    } else if (ev.type === 'prefetched_recommendation_context') {
      rows.push(`
        <div class="trace-item">
          <div class="trace-title">Prefetched recommendation context</div>
          <pre>${escapeHtml(ev.content || '')}</pre>
        </div>
      `);
    } else if (ev.type === 'profile_reask_repair') {
      rows.push(`
        <div class="trace-item trace-error">
          <div class="trace-title">Profile re-ask repaired</div>
          <div class="trace-meta">${escapeHtml(ev.message || '')}</div>
        </div>
      `);
    } else if (ev.type === 'catalog_title_repair') {
      const fixes = (ev.mismatches || []).map(m =>
        `<li><span>${escapeHtml(m.code || '')}</span> ${escapeHtml(m.stated || '')} -> ${escapeHtml(m.expected || '')}</li>`
      ).join('');
      rows.push(`
        <div class="trace-item trace-error">
          <div class="trace-title">Catalog title repair</div>
          <ol>${fixes}</ol>
        </div>
      `);
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
    } else if (ev.type === 'tool_result' && ev.name === 'get_current_schedule') {
      const r = ev.result || {};
      const conflicts = (r.conflicts || []).map(c => `${escapeHtml((c.between || []).join(' vs '))} (${escapeHtml(c.day || '')} ${escapeHtml(c.times || '')})`).join('<br/>');
      rows.push(`
        <div class="trace-item">
          <div class="trace-title">Current builder schedule</div>
          <div class="trace-meta">${escapeHtml(r.course_count ?? 0)} courses · ${escapeHtml(r.total_su_credits ?? 0)} SU credits · conflicts: ${escapeHtml(r.has_conflicts ? 'yes' : 'no')}</div>
          ${conflicts ? `<div class="trace-meta">${conflicts}</div>` : ''}
        </div>
      `);
    } else if (ev.type === 'tool_result' && ev.name === 'check_courses_against_current_schedule') {
      const r = ev.result || {};
      const checked = (r.candidate_codes_checked || []).join(', ') || (r.requested_codes || []).join(', ');
      const blockers = (r.blocking_conflicts || []).slice(0, 6).map(c =>
        `${escapeHtml(c.candidate || '')}: ${escapeHtml((c.between || []).join(' vs '))} (${escapeHtml(c.day || '')} ${escapeHtml(c.times || '')})`
      ).join('<br/>');
      rows.push(`
        <div class="trace-item">
          <div class="trace-title">Candidate schedule check</div>
          <div class="trace-meta">checked: ${escapeHtml(checked || '-')} | fits: ${escapeHtml(r.candidate_fits_current_schedule)} | overall clean: ${escapeHtml(r.overall_schedule_conflict_free)}</div>
          ${r.auto_added_coreqs?.length ? `<div class="trace-meta">auto-added coreqs: ${escapeHtml(r.auto_added_coreqs.join(', '))}</div>` : ''}
          ${blockers ? `<div class="trace-meta">${blockers}</div>` : ''}
        </div>
      `);
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
        `<li><span>${escapeHtml(h.code || '')}</span> ${escapeHtml(h.title || '')}<small>${h.rerank_score != null ? `rerank ${escapeHtml(h.rerank_score)}` : `score ${Number(h.score || 0).toFixed(2)}`} | eligible ${escapeHtml(h.eligible)} | offered ${escapeHtml(h.likely_offered)}</small></li>`
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
  sendBtn.textContent = val ? '…' : 'Send';
  checkScheduleBtn.disabled = val || schedule.length === 0;
}

function scrollBottom() {
  chatMsgs.scrollTop = chatMsgs.scrollHeight;
}

function escapeHtml(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

// ════════════════════════ Light / dark theme toggle ════════════════════════
const ICON_MOON = '<svg viewBox="0 0 24 24" width="13" height="13" fill="currentColor" aria-hidden="true"><path d="M13 2.5a8 8 0 1 0 8.5 11.2A6.5 6.5 0 0 1 13 2.5z"/></svg>';
const ICON_SUN  = '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4.2" fill="currentColor" stroke="none"/><path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.2 5.2l1.4 1.4M17.4 17.4l1.4 1.4M18.8 5.2l-1.4 1.4M6.6 17.4l-1.4 1.4"/></svg>';
function applyThemeLabel() {
  const btn = document.getElementById('theme-toggle'); if (!btn) return;
  const dark = document.documentElement.getAttribute('data-theme') === 'dark';
  btn.querySelector('.label').textContent = dark ? 'Light' : 'Dark';
  btn.querySelector('.ico').innerHTML = dark ? ICON_SUN : ICON_MOON;
}
(function initTheme() {
  try { const s = localStorage.getItem('suschedule-theme'); if (s) document.documentElement.setAttribute('data-theme', s); } catch (_) {}
  const btn = document.getElementById('theme-toggle');
  if (btn) btn.addEventListener('click', () => {
    const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem('suschedule-theme', next); } catch (_) {}
    applyThemeLabel();
  });
  applyThemeLabel();
})();

// ════════════════════════ Boot ════════════════════════
(async () => {
  renderBuilder();
  try {
    await initSession();
  } catch (err) {
    addAgentMsg(`Error: ${escapeHtml(err.message)}`, 'error');
  }
})();
