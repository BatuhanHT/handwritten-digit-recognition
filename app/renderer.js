// Arayüz: çizim, terminal paneli, ara görüntüler ve sonuç kartı.
// Python'un çıktısı olduğu gibi gösterilir; yalnızca okunabilsin diye satırlar aralıklı açılır.
const $ = (sel) => document.querySelector(sel);
const canvas = $('#pad');
const ctx = canvas.getContext('2d');
const term = $('#term');
const hint = $('#hint');
const btnPredict = $('#predict');
const btnClear = $('#clear');
const resultEl = $('#result');

const LINE_DELAY = 70;   // ms, sıradan satırlar arası
const STEP_DELAY = 300;  // ms, [n/4] adım başlıklarından önce
const RESULT_DELAY = 250;
const TIMEOUT_MS = 10000; // Python bu sürede yanıt vermezse istek hata sayılır

const state = { ready: false, busy: false, drawn: false, reqId: 0, buffer: [], drawing: false, last: null, timer: null };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
// %100,0 gibi yanıltıcı yuvarlamalar yerine %99,9+ (predict.py'deki pct ile aynı kural)
const pct = (p) => (p >= 0.9995 ? '%99,9+' : `%${(p * 100).toFixed(1).replace('.', ',')}`);

/* ---------- Çizim ---------- */
function resetPad() {
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 20;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = '#000';
  ctx.fillStyle = '#000';
  state.drawn = false;
  hint.hidden = false;
}

function pos(e) {
  const r = canvas.getBoundingClientRect();
  return { x: ((e.clientX - r.left) * canvas.width) / r.width, y: ((e.clientY - r.top) * canvas.height) / r.height };
}

canvas.addEventListener('pointerdown', (e) => {
  if (state.busy) return;
  state.drawing = true;
  canvas.setPointerCapture(e.pointerId);
  state.last = pos(e);
  ctx.beginPath();
  ctx.arc(state.last.x, state.last.y, ctx.lineWidth / 2, 0, Math.PI * 2);
  ctx.fill();
  state.drawn = true;
  hint.hidden = true;
  setButtons();
});
canvas.addEventListener('pointermove', (e) => {
  if (!state.drawing) return;
  const events = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
  for (const ev of events) {
    const p = pos(ev);
    ctx.beginPath();
    ctx.moveTo(state.last.x, state.last.y);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
    state.last = p;
  }
});
for (const type of ['pointerup', 'pointercancel']) {
  canvas.addEventListener(type, () => { state.drawing = false; });
}

/* ---------- Terminal ---------- */
function addLine(text, kind) {
  const div = document.createElement('div');
  div.className = `ln ${kind || classify(text)}`;
  const step = text.match(/^(\[\d\/4\] [^:]+:)(.*)$/);
  const final = text.match(/^(Tahmin: )(\d)(.*)$/);
  if (step) {
    const b = document.createElement('b');
    b.textContent = step[1];
    div.append(b, step[2]);
  } else if (final) {
    const b = document.createElement('b');
    b.textContent = final[2];
    div.append(final[1], b, final[3]);
  } else {
    div.textContent = text;
  }
  term.appendChild(div);
  term.scrollTop = term.scrollHeight;
}

function classify(text) {
  if (/^\[\d\/4\]/.test(text)) return 'step';
  if (text.startsWith('Tahmin:')) return 'final';
  if (text.startsWith('Hata')) return 'err';
  if (/^\s+\d\s/.test(text)) return text.includes('◀') ? 'bar top' : 'bar';
  return '';
}

/* ---------- Aşamalar ve sonuç ---------- */
function setStage(n, src) {
  const st = $(`#st${n}`);
  st.classList.add('on');
  if (n === 4) {
    const box = st.querySelector('.maps');
    box.replaceChildren(...src.map((b64) => Object.assign(new Image(), { src: `data:image/png;base64,${b64}`, alt: '' })));
  } else {
    st.querySelector('img').src = src.startsWith('data:') ? src : `data:image/png;base64,${src}`;
  }
}

function resetStages() {
  for (let n = 1; n <= 4; n++) {
    const st = $(`#st${n}`);
    st.classList.remove('on');
    const img = st.querySelector('img');
    if (img && n !== 4) img.removeAttribute('src');
    if (n === 4) st.querySelector('.maps').replaceChildren();
  }
}

function setResult(stateName, digit = '?', conf = 'Çizim bekleniyor', probs = null) {
  resultEl.dataset.state = stateName;
  $('#digit').textContent = digit;
  $('#conf').textContent = conf;
  const list = $('#top3');
  list.replaceChildren();
  if (!probs) return;
  const top = probs.map((p, d) => [d, p]).sort((a, b) => b[1] - a[1]).slice(0, 3);
  for (const [d, p] of top) {
    const li = document.createElement('li');
    const bar = document.createElement('span');
    bar.className = 'b';
    bar.style.width = `${Math.max(2, p * 72)}px`;
    li.append(String(d), bar, pct(p));
    list.appendChild(li);
  }
}

/* ---------- Akış ---------- */
function setButtons() {
  btnPredict.disabled = !state.ready || state.busy;
  btnClear.disabled = state.busy;
}

function setStatus(kind, text) {
  $('#status').dataset.state = kind;
  $('#status-text').textContent = text;
}

function predict() {
  if (!state.ready || state.busy) return;
  if (!state.drawn) {
    hint.classList.remove('nudge');
    void hint.offsetWidth;
    hint.classList.add('nudge');
    return;
  }
  state.busy = true;
  state.reqId += 1;
  state.buffer = [];
  setButtons();
  const png = canvas.toDataURL('image/png');
  resetStages();
  setStage(1, png);
  setResult('busy', '…', 'Hesaplanıyor');
  term.replaceChildren();
  addLine(`$ tahmin #${state.reqId}  →  çizim Python sürecine gönderildi`, 'cmd');
  const id = state.reqId;
  state.timer = setTimeout(() => {
    if (state.busy && state.reqId === id) finishWithError('Zaman aşımı: Python yanıt vermedi');
  }, TIMEOUT_MS);
  window.demo.predict(id, png.split(',')[1]);
}

// fromPython: Python hatayı zaten "Hata: ..." diye yazdı; terminalde ikinci kez gösterme
function finishWithError(message, { fromPython = false } = {}) {
  clearTimeout(state.timer);
  const lines = state.buffer;
  state.buffer = [];
  playback(lines, null, message, !fromPython);
}

async function playback(lines, result, error, showErrorLine = false) {
  for (const { stream, text } of lines) {
    await sleep(/^\[\d\/4\]/.test(text) ? STEP_DELAY : LINE_DELAY);
    addLine(text, stream === 'stderr' ? 'err' : undefined);
    if (result && text.startsWith('[2/4]')) setStage(2, result.images.crop);
    if (result && text.startsWith('[3/4]')) setStage(3, result.images.img28);
    if (result && text.startsWith('[4/4]')) setStage(4, result.images.maps);
  }
  if (result) {
    await sleep(RESULT_DELAY);
    setResult('done', String(result.digit), `${pct(result.probs[result.digit])} güven`, result.probs);
  } else {
    if (showErrorLine) addLine(error || 'Hata', 'err');
    setResult('idle', '!', error || 'Hata');
  }
  state.busy = false;
  setButtons();
}

function clearAll() {
  if (state.busy) return;
  resetPad();
  resetStages();
  setResult('idle');
  term.replaceChildren();
  addLine('Yeni çizim bekleniyor…', 'muted');
  setButtons();
}

function handleTag(tag, data) {
  if (tag === 'READY') {
    state.ready = true;
    setStatus('ok', 'Model hazır · CPU');
    addLine('Bir rakam çizin ve Tahmin Et’e basın (Enter).', 'muted');
  } else if (tag === 'FATAL') {
    setStatus('err', data.message);
  } else if (tag === 'RESULT' && data.id === state.reqId && state.busy) {
    clearTimeout(state.timer);
    const lines = state.buffer;
    state.buffer = [];
    playback(lines, data, null);
  } else if (tag === 'ERROR' && data.id === state.reqId && state.busy) {
    finishWithError(data.message, { fromPython: true });
  }
  setButtons();
}

window.demo.onLine(({ stream, text }) => {
  if (text.startsWith('@@')) {
    const space = text.indexOf(' ');
    let data = {};
    try {
      data = JSON.parse(text.slice(space + 1));
    } catch {
      addLine(`Okunamayan çıktı: ${text.slice(0, 80)}`, 'err');
      return;
    }
    handleTag(text.slice(2, space), data);
    return;
  }
  if (state.busy) state.buffer.push({ stream, text });
  else addLine(text, stream === 'stderr' ? 'err' : stream === 'cmd' ? 'cmd' : undefined);
});

window.demo.onExit(({ code, signal }) => {
  state.ready = false;
  if (state.busy) finishWithError('Python süreci kapandı');
  // FATAL mesajı (ör. model dosyası yok) daha açıklayıcı; onu ezme
  if ($('#status').dataset.state !== 'err') setStatus('err', `Python süreci kapandı (${signal || `kod ${code}`})`);
  addLine('Python süreci kapandı. Yeniden başlatmak için Cmd+R.', 'err');
  setButtons();
});

btnPredict.addEventListener('click', predict);
btnClear.addEventListener('click', clearAll);
window.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') { e.preventDefault(); predict(); }
  if (e.key === 'Escape' || e.key === 'Backspace' || e.key === 'Delete') { e.preventDefault(); clearAll(); }
});

resetPad();
setResult('idle');
setButtons();
