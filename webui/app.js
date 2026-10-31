/* NInfer Studio — front-end. No framework, no build step, no cloud.
   Owns: tab routing, language (EN/EL), theme, telemetry polling, streaming chat,
   engine start/stop/suspend/resume, artifact listing. */
'use strict';

/* ─────────────────────────── i18n ─────────────────────────── */
const I18N = {
  en: {
    'a11y.skip': 'Skip to content',
    'a11y.theme': 'Toggle theme',
    'a11y.lang': 'Switch language',

    'brand.sub': 'local inference · RTX 3090',
    'tab.chat': 'Chat', 'tab.live': 'Live', 'tab.engine': 'Engine', 'tab.model': 'Model',

    'status.connecting': 'Connecting…',
    'status.ready': 'ready',
    'status.busy': 'generating',
    'status.suspended': 'suspended',
    'status.offline': 'offline',

    'common.retry': 'Retry',
    'common.loading': 'loading…',
    'common.none': '—',
    'common.unknown': 'unknown',

    'chat.title': 'Chat',
    'chat.sub': 'Streaming from the local engine',
    'chat.new': 'New chat',
    'chat.write': 'Write',
    'chat.ready': 'ready',
    'chat.busy': 'generating',
    'chat.thinking': 'thinking',
    'chat.maxTokens': 'max tokens',
    'chat.temp': 'temp',
    'chat.off': 'off',
    'chat.emptyTitle': 'Ready for your first question',
    'chat.emptyBody': 'Text or image (drag & drop). History stays in this browser tab only.',
    'chat.attach': 'Attach image',
    'chat.placeholder': 'Ask something… (Enter = send, Shift+Enter = new line)',
    'chat.send': 'Send',
    'chat.stop': 'Stop',
    'chat.turns': '{n} messages',
    'chat.turnsOne': '{n} message',
    'chat.you': 'You',
    'chat.ai': 'AI',
    'chat.thinkingTokens': 'thinking · {n} tokens',
    'chat.thinkingOpen': 'thinking',
    'stat.gen': 'gen',
    'stat.decode': 'decode',
    'stat.prefill': 'prefill',
    'stat.accept': 'draft',
    'stat.ttft': 'ttft',
    'stat.total': 'total',
    'stat.genTitle': 'Tokens generated in this reply',
    'stat.decodeTitle': 'Generation speed: output tokens per second, measured from the first token to the last, so prompt reading is excluded.',
    'stat.prefillTitle': 'Prompt reading speed: prompt tokens divided by the time until the first output token',
    'stat.ttftTitle': 'Time to first token: how long the engine spent reading the prompt before answering',
    'therm.title': 'Temperatures & fans',
    'therm.core': 'GPU core',
    'therm.hotspot': 'hot spot',
    'therm.memory': 'VRAM junction',
    'therm.board': 'board / VRM',
    'therm.fans': 'fans',
    'therm.naDriver': 'not reported by this card/driver (NVML: not supported)',
    'therm.naApi': 'not exposed by NVML on GeForce cards',
    'therm.slowdown': 'slowdown {n} °C',
    'therm.maxop': 'max {n} °C',
    'therm.shutdown': 'shutdown {n} °C',
    'status.loading': 'loading model',
    'caps.noSuspend': 'This server was not started with model suspend enabled',
    'caps.noMetrics': 'This engine build does not serve /metrics, so throughput, VRAM and power tiles stay empty. Chat works normally.',
    'caps.noMetricsAuth': '/metrics requires the API key (NINFER_API_KEY), so the metric tiles stay empty.',
    'caps.noVision': 'This model was not loaded with image input',
    'caps.startDisabled': 'The engine is already serving — nothing to start',
    'cfg.title': 'Configuration',
    'cfg.hint': 'studio.config.json / environment',
    'cfg.engine': 'engine',
    'cfg.ui': 'web UI',
    'cfg.cmd': 'start command',
    'cfg.file': 'config file',
    'cfg.none': 'not set',
    'cfg.defaults': 'built-in defaults',
    'therm.fanCount': '{n} fans',
    'm.hot': 'hot',
    'm.warm': 'warm',
    'm.ok': 'ok',
    'stat.totalTitle': 'Total wall-clock time of this request',
    'stat.acceptTitle': 'Speculative decoding: share of drafted tokens the model accepted. Higher = fewer wasted drafts.',
    'chat.copied': 'Copied',

    'm.decode': 'decode',
    'm.decodeTitle': 'Decode',
    'm.tprTitle': 'Tok / round',
    'm.tprFoot': 'speculative verification',
    'm.prefill': 'prefill',
    'm.tempTitle': 'GPU temperature',
    'm.powerTitle': 'Power',
    'm.utilTitle': 'GPU util',
    'm.kvTitle': 'KV cache',

    'rail.waiting': 'waiting for a sample',
    'rail.lastReq': 'last request · {n} tok',
    'rail.now': 'Now',
    'rail.tokRound': 'tok / round',
    'rail.generated': 'generated',
    'rail.kvPages': 'KV pages',
    'rail.model': 'model',
    'rail.context': 'context',
    'rail.vram': 'VRAM',
    'rail.state': 'state',
    'rail.suspend': 'Suspend',
    'rail.resume': 'Resume',

    'live.title': 'Live',
    'live.sub': 'Telemetry from',
    'live.sub2': 'sample every',
    'live.refresh': 'Refresh',
    'live.spec': 'spec',
    'live.uptime': 'uptime',

    'ctx.title': 'Context space',
    'ctx.hint': 'host pinned',
    'ctx.used': 'in use',
    'ctx.total': 'total',
    'ctx.peak': 'peak',
    'ctx.kvTokens': 'KV tokens',

    'q.title': 'Queues',

    'eng.title': 'Engine',
    'eng.sub': 'Control plus facts the server reports about itself — no estimates.',
    'eng.start': 'Start',
    'eng.resume': 'Resume',
    'eng.stop': 'Stop',
    'eng.hint': 'Suspend snapshots the model state to host RAM and releases its device memory. Resume reads the weights back from disk. Stop kills the process.',
    'eng.process': 'Process',
    'eng.modelId': 'model id',
    'eng.weights': 'weights path',
    'eng.spec': 'speculation',
    'eng.decoded': 'decode tokens',
    'eng.sleeping': 'sleeping',
    'eng.pid': 'pid',
    'eng.caps': 'Capabilities',
    'eng.maxCtx': 'max context',
    'eng.build': 'Build',
    'eng.version': 'version',
    'eng.log': 'Engine log',

    'state.idle': 'Idle',
    'state.generating': 'Generating',
    'state.prefill': 'Prefill',
    'state.suspended': 'Suspended',
    'state.offline': 'Engine down',
    'state.loading': 'Loading…',

    'model.title': 'Model',
    'model.sub': 'The loaded artifact and the files sitting on disk.',
    'model.loaded': 'Loaded',
    'model.alias': 'alias',
    'model.path': 'path',
    'model.modalities': 'modalities',
    'model.vramNow': 'VRAM now',
    'model.artifacts': 'Artifacts on disk',
    'model.file': 'file',
    'model.size': 'size',
    'model.license': 'License note',
    'model.licenseBody': 'The base model is Apache 2.0. The Swift 1.5 artifact is free for individuals and organizations up to US$1,000,000 annual revenue; above that a Swift Enterprise License is required.',
    'model.notLoaded': 'not loaded',
    'model.suspended': 'suspended',
    'model.loadedTag': 'loaded',
    'model.noFiles': 'no .ninfer files found under models/',
    'model.freeDisk': '{free} free of {total}',

    'toast.suspended': 'Model suspended — device memory released.',
    'toast.resumed': 'Model resumed.',
    'toast.busy': 'Engine is busy — suspend refused (409).',
    'toast.suspendDisabled': 'This engine was started without --model-suspend.',
    'toast.engineUp': 'Engine started.',
    'toast.engineDown': 'Engine stopped.',
    'toast.alreadyUp': 'Engine is already running.',
    'toast.noEngine': 'Engine binary or artifact not found.',
    'toast.failed': 'Request failed: {msg}',
    'toast.stopped': 'Generation cancelled.',

    'err.offline': 'Engine is not reachable at {url}. Start it from the Engine tab.',
    'err.suspend': 'suspend needs --model-suspend'
  },

  el: {
    'a11y.skip': 'Μετάβαση στο περιεχόμενο',
    'a11y.theme': 'Αλλαγή θέματος',
    'a11y.lang': 'Αλλαγή γλώσσας',

    'brand.sub': 'τοπικό inference · RTX 3090',
    'tab.chat': 'Συνομιλία', 'tab.live': 'Live', 'tab.engine': 'Μηχανή', 'tab.model': 'Μοντέλο',

    'status.connecting': 'Σύνδεση…',
    'status.ready': 'έτοιμο',
    'status.busy': 'παράγει',
    'status.suspended': 'σε αναστολή',
    'status.offline': 'εκτός',

    'common.retry': 'Ξανά',
    'common.loading': 'φόρτωση…',
    'common.none': '—',
    'common.unknown': 'άγνωστο',

    'chat.title': 'Συνομιλία',
    'chat.sub': 'Streaming από το τοπικό engine',
    'chat.new': 'Νέα συνομιλία',
    'chat.write': 'Γράψε',
    'chat.ready': 'έτοιμο',
    'chat.busy': 'παράγει',
    'chat.thinking': 'thinking',
    'chat.maxTokens': 'max tokens',
    'chat.temp': 'θερμοκρασία',
    'chat.off': 'κλειστό',
    'chat.emptyTitle': 'Έτοιμο για την πρώτη ερώτηση',
    'chat.emptyBody': 'Κείμενο ή εικόνα (drag & drop). Το ιστορικό μένει μόνο σε αυτή την καρτέλα.',
    'chat.attach': 'Επισύναψη εικόνας',
    'chat.placeholder': 'Ρώτα κάτι… (Enter = αποστολή, Shift+Enter = νέα γραμμή)',
    'chat.send': 'Αποστολή',
    'chat.stop': 'Διακοπή',
    'chat.turns': '{n} μηνύματα',
    'chat.turnsOne': '{n} μήνυμα',
    'chat.you': 'Εσύ',
    'chat.ai': 'AI',
    'chat.thinkingTokens': 'thinking · {n} tokens',
    'chat.thinkingOpen': 'thinking',
    'stat.gen': 'παραγωγή',
    'stat.decode': 'decode',
    'stat.prefill': 'prefill',
    'stat.accept': 'draft',
    'stat.ttft': 'ttft',
    'stat.total': 'σύνολο',
    'stat.genTitle': 'Tokens που παράχθηκαν σε αυτή την απάντηση',
    'stat.decodeTitle': 'Ταχύτητα παραγωγής: tokens ανά δευτερόλεπτο από το πρώτο ως το τελευταίο token — χωρίς το διάβασμα του prompt.',
    'stat.prefillTitle': 'Ταχύτητα ανάγνωσης prompt: tokens prompt διά τον χρόνο μέχρι το πρώτο token',
    'stat.ttftTitle': 'Χρόνος μέχρι το πρώτο token: πόσο έκανε ο engine να διαβάσει το prompt',
    'therm.title': 'Θερμοκρασίες & ανεμιστήρες',
    'therm.core': 'GPU core',
    'therm.hotspot': 'hot spot',
    'therm.memory': 'VRAM junction',
    'therm.board': 'board / VRM',
    'therm.fans': 'ανεμιστήρες',
    'therm.naDriver': 'δεν το αναφέρει η κάρτα/driver (NVML: not supported)',
    'therm.naApi': 'δεν εκτίθεται από NVML στις GeForce',
    'therm.slowdown': 'slowdown {n} °C',
    'therm.maxop': 'max {n} °C',
    'therm.shutdown': 'shutdown {n} °C',
    'status.loading': 'φορτώνει το μοντέλο',
    'caps.noSuspend': 'Ο server δεν ξεκίνησε με ενεργό suspend μοντέλου',
    'caps.noMetrics': 'Αυτή η έκδοση του engine δεν σερβίρει /metrics, οπότε τα tiles απόδοσης, VRAM και ισχύος μένουν κενά. Το chat δουλεύει κανονικά.',
    'caps.noMetricsAuth': 'Το /metrics απαιτεί το API key (NINFER_API_KEY), οπότε τα metric tiles μένουν κενά.',
    'caps.noVision': 'Το μοντέλο δεν φορτώθηκε με υποστήριξη εικόνας',
    'caps.startDisabled': 'Η μηχανή εξυπηρετεί ήδη — δεν υπάρχει κάτι να ξεκινήσει',
    'cfg.title': 'Ρυθμίσεις',
    'cfg.hint': 'studio.config.json / environment',
    'cfg.engine': 'engine',
    'cfg.ui': 'web UI',
    'cfg.cmd': 'εντολή εκκίνησης',
    'cfg.file': 'αρχείο ρυθμίσεων',
    'cfg.none': 'δεν έχει οριστεί',
    'cfg.defaults': 'προεπιλογές κώδικα',
    'therm.fanCount': '{n} ανεμιστήρες',
    'm.hot': 'ζεστό',
    'm.warm': 'χλιαρό',
    'm.ok': 'ok',
    'stat.totalTitle': 'Συνολικός πραγματικός χρόνος του αιτήματος',
    'stat.acceptTitle': 'Speculative decoding: ποσοστό drafted tokens που δέχτηκε το μοντέλο. Πιο ψηλό = λιγότερα χαμένα drafts.',
    'chat.copied': 'Αντιγράφηκε',

    'm.decode': 'decode',
    'm.decodeTitle': 'Decode',
    'm.tprTitle': 'Tok / round',
    'm.tprFoot': 'speculative verification',
    'm.prefill': 'prefill',
    'm.tempTitle': 'Θερμοκρασία GPU',
    'm.powerTitle': 'Ισχύς',
    'm.utilTitle': 'GPU util',
    'm.kvTitle': 'KV cache',

    'rail.waiting': 'αναμονή δείγματος',
    'rail.lastReq': 'τελευταίο αίτημα · {n} tok',
    'rail.now': 'Τώρα',
    'rail.tokRound': 'tok / round',
    'rail.generated': 'παράχθηκαν',
    'rail.kvPages': 'KV pages',
    'rail.model': 'μοντέλο',
    'rail.context': 'context',
    'rail.vram': 'VRAM',
    'rail.state': 'κατάσταση',
    'rail.suspend': 'Αναστολή',
    'rail.resume': 'Επαναφορά',

    'live.title': 'Live',
    'live.sub': 'Telemetry από',
    'live.sub2': 'δείγμα κάθε',
    'live.refresh': 'Ανανέωση',
    'live.spec': 'spec',
    'live.uptime': 'uptime',

    'ctx.title': 'Χώρος context',
    'ctx.hint': 'host pinned',
    'ctx.used': 'σε χρήση',
    'ctx.total': 'σύνολο',
    'ctx.peak': 'κορυφή',
    'ctx.kvTokens': 'KV tokens',

    'q.title': 'Ουρές',

    'eng.title': 'Μηχανή',
    'eng.sub': 'Έλεγχος και πραγματικά στοιχεία που δίνει ο server — χωρίς εκτιμήσεις.',
    'eng.start': 'Έναρξη',
    'eng.resume': 'Επαναφορά',
    'eng.stop': 'Τερματισμός',
    'eng.hint': 'Η αναστολή κρατά snapshot της κατάστασης του μοντέλου στη RAM και ελευθερώνει τη μνήμη της κάρτας. Η επαναφορά ξαναδιαβάζει τα weights από τον δίσκο. Ο τερματισμός σκοτώνει τη διεργασία.',
    'eng.process': 'Διεργασία',
    'eng.modelId': 'model id',
    'eng.weights': 'διαδρομή weights',
    'eng.spec': 'speculation',
    'eng.decoded': 'decode tokens',
    'eng.sleeping': 'sleeping',
    'eng.pid': 'pid',
    'eng.caps': 'Ικανότητες',
    'eng.maxCtx': 'μέγιστο context',
    'eng.build': 'Build',
    'eng.version': 'έκδοση',
    'eng.log': 'Log μηχανής',

    'state.idle': 'Αδρανές',
    'state.generating': 'Παράγει',
    'state.prefill': 'Prefill',
    'state.suspended': 'Σε αναστολή',
    'state.offline': 'Μηχανή εκτός',
    'state.loading': 'Φορτώνει…',

    'model.title': 'Μοντέλο',
    'model.sub': 'Το φορτωμένο artifact και τα αρχεία στον δίσκο.',
    'model.loaded': 'Φορτωμένο',
    'model.alias': 'alias',
    'model.path': 'διαδρομή',
    'model.modalities': 'modalities',
    'model.vramNow': 'VRAM τώρα',
    'model.artifacts': 'Artifacts στον δίσκο',
    'model.file': 'αρχείο',
    'model.size': 'μέγεθος',
    'model.license': 'Σημείωση άδειας',
    'model.licenseBody': 'Το base μοντέλο είναι Apache 2.0. Το Swift 1.5 artifact είναι δωρεάν για άτομα και οργανισμούς έως US$1.000.000 ετήσιο τζίρο· πάνω από αυτό απαιτείται Swift Enterprise License.',
    'model.notLoaded': 'δεν είναι φορτωμένο',
    'model.suspended': 'σε αναστολή',
    'model.loadedTag': 'φορτωμένο',
    'model.noFiles': 'δεν βρέθηκαν αρχεία .ninfer στο models/',
    'model.freeDisk': '{free} ελεύθερα από {total}',

    'toast.suspended': 'Το μοντέλο μπήκε σε αναστολή — η μνήμη επιστράφηκε.',
    'toast.resumed': 'Το μοντέλο επανήλθε.',
    'toast.busy': 'Η μηχανή είναι απασχολημένη — η αναστολή απορρίφθηκε (409).',
    'toast.suspendDisabled': 'Το engine ξεκίνησε χωρίς --model-suspend.',
    'toast.engineUp': 'Η μηχανή ξεκίνησε.',
    'toast.engineDown': 'Η μηχανή τερματίστηκε.',
    'toast.alreadyUp': 'Η μηχανή τρέχει ήδη.',
    'toast.noEngine': 'Δεν βρέθηκε binary ή artifact.',
    'toast.failed': 'Αποτυχία: {msg}',
    'toast.stopped': 'Η παραγωγή ακυρώθηκε.',

    'err.offline': 'Η μηχανή δεν αποκρίνεται στο {url}. Ξεκίνα τη από την καρτέλα Μηχανή.',
    'err.suspend': 'η αναστολή θέλει --model-suspend'
  }
};

const LS = { lang: 'ninfer.lang', theme: 'ninfer.theme' };
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

let lang = localStorage.getItem(LS.lang) || 'en';
const t = (key, vars) => {
  const table = I18N[lang] || I18N.en;
  let s = table[key] ?? I18N.en[key] ?? key;
  if (vars) for (const k in vars) s = s.replace('{' + k + '}', vars[k]);
  return s;
};

function applyLang() {
  document.documentElement.lang = lang;
  $('#langLabel').textContent = lang === 'en' ? 'EN' : 'ΕΛ';
  $('#langBtn').title = t('a11y.lang');
  $$('[data-i18n]').forEach((el) => { el.textContent = t(el.dataset.i18n); });
  $$('[data-i18n-title]').forEach((el) => { el.title = t(el.dataset.i18nTitle); });
  $$('[data-i18n-placeholder]').forEach((el) => { el.placeholder = t(el.dataset.i18nPlaceholder); });
  renderTurns();
  if (state) renderStatus(state);
}

$('#langBtn').addEventListener('click', () => {
  lang = lang === 'en' ? 'el' : 'en';
  localStorage.setItem(LS.lang, lang);
  applyLang();
});

/* ─────────────────────────── theme ─────────────────────────── */
const theme = localStorage.getItem(LS.theme) || 'dark';
document.documentElement.dataset.theme = theme;
$('#themeBtn').addEventListener('click', () => {
  const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  localStorage.setItem(LS.theme, next);
});

/* ─────────────────────────── tabs ─────────────────────────── */
const VIEWS = ['chat', 'live', 'engine', 'model'];
function showView(name) {
  if (!VIEWS.includes(name)) name = 'chat';
  $$('.view').forEach((v) => { v.hidden = v.id !== 'view-' + name; });
  $$('.tabs button').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.view === name)));
  if (location.hash !== '#' + name) history.replaceState(null, '', '#' + name);
  if (name === 'model') loadArtifacts();
  if (name === 'engine') loadLog();   // fresh tail every time the tab is opened
}
$$('.tabs button').forEach((b) => b.addEventListener('click', () => showView(b.dataset.view)));
window.addEventListener('hashchange', () => showView(location.hash.slice(1)));

/* ─────────────────────────── helpers ─────────────────────────── */
const fmtInt = (n) => (n == null || !isFinite(n) ? '—' : Math.round(n).toLocaleString(lang === 'el' ? 'el-GR' : 'en-US'));
const fmt1 = (n) => (n == null || !isFinite(n) ? '—' : (Math.round(n * 10) / 10).toFixed(1));
const gib = (bytes) => (bytes == null ? null : bytes / 1073741824);
const pct = (a, b) => (a == null || !b ? null : Math.max(0, Math.min(100, (a / b) * 100)));
const bytesText = (b) => {
  if (b == null) return '—';
  const g = b / 1073741824;
  return g >= 1 ? fmt1(g) + ' GiB' : fmt1(b / 1048576) + ' MiB';
};
const dur = (s) => {
  if (!isFinite(s) || s < 0) return '—';
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60);
  return h ? `${h}h ${m}m` : m ? `${m}m ${sec}s` : `${fmt1(s)}s`;
};
const num = (v) => (typeof v === 'number' && isFinite(v) ? v : null);

function toast(msg, kind) {
  const el = document.createElement('div');
  el.className = 'toast' + (kind ? ' ' + kind : '');
  el.textContent = msg;
  $('#toasts').appendChild(el);
  setTimeout(() => el.remove(), kind === 'err' ? 7000 : 3800);
}

async function api(path, opts) {
  const res = await fetch(path, { ...opts, headers: { ...opts?.headers, 'X-Ninfer-Studio': '1' } });
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch (_) { data = { raw: text }; }
  if (!res.ok) {
    const msg = (data && (data.error?.message || data.error?.raw || data.error)) || res.status;
    throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg));
  }
  return data;
}

/* ─────────────────────────── sparklines ─────────────────────────── */
const HIST = { tps: [], tpr: [], pre: [], vram: [], temp: [], power: [], util: [], kv: [] };
const HIST_MAX = 60;
function push(key, value) {
  if (value == null || !isFinite(value)) return;
  const arr = HIST[key];
  arr.push(value);
  if (arr.length > HIST_MAX) arr.shift();
}
function sparkPath(values, w = 100, h = 28) {
  if (!values.length) return '';
  if (values.length === 1) return `M0 ${(h / 2).toFixed(2)} L${w} ${(h / 2).toFixed(2)}`;
  const min = Math.min(...values), max = Math.max(...values);
  const span = max - min || Math.max(1, Math.abs(max) * 0.1);
  const step = values.length > 1 ? w / (values.length - 1) : w;
  return values.map((v, i) => {
    const x = (i * step).toFixed(2);
    const y = (h - ((v - min) / span) * (h - 2) - 1).toFixed(2);
    return (i ? 'L' : 'M') + x + ' ' + y;
  }).join(' ');
}
function paintSparks() {
  for (const key of Object.keys(HIST)) {
    const d = sparkPath(HIST[key]);
    $$(`[data-spark="${key}"]`).forEach((p) => { p.setAttribute('d', d); });
  }
  const rail = $('#sparkPath');
  if (rail) rail.setAttribute('d', sparkPath(HIST.tps));
}

/* ─────────────────────────── telemetry ─────────────────────────── */
let state = null;
let firstUpAt = null;
let peakCtx = 0;
let failStreak = 0;   // consecutive /api/state misses; 1 blip must not flash a warning
let prevCounters = null;

function rate(cur, prev, dt, curKey, secKey) {
  if (!cur || !prev || dt <= 0) return null;
  const dGen = num(cur[curKey]) != null && num(prev[curKey]) != null ? cur[curKey] - prev[curKey] : null;
  const dSec = num(cur[secKey]) != null && num(prev[secKey]) != null ? cur[secKey] - prev[secKey] : null;
  if (dSec != null && dSec > 0) return dGen / dSec;
  if (dGen != null) return dGen / dt;
  return null;
}

function engineStateIcon(st) {
  const badge = $('#esBadge');
  if (!st.up) {
    badge.dataset.state = 'offline';
    $('#esState').textContent = t('state.offline');
    return;
  }
  const sleeping = st.props?.is_sleeping || st.residency?.state === 'suspended';
  if (sleeping) {
    badge.dataset.state = 'suspended';
    $('#esState').textContent = t('state.suspended');
    return;
  }
  const running = num(st.load?.running) ?? num(st.load?.requests?.running) ?? 0;
  const prefill = num(st.load?.prefilling) ?? num(st.load?.requests?.prefilling) ?? 0;
  if (running || prefill) {
    badge.dataset.state = prefill && !running ? 'prefill' : 'generating';
    $('#esState').textContent = badge.dataset.state === 'prefill' ? t('state.prefill') : t('state.generating');
  } else {
    badge.dataset.state = 'idle';
    $('#esState').textContent = t('state.idle');
  }
}

function renderStatus(st) {
  const pill = $('#statusPill'), text = $('#statusPillText');
  if (!st) { pill.dataset.state = 'unknown'; text.textContent = t('status.connecting'); return; }
  if (!st.up) {
    // 503 while the model loads is not "offline": the process is up and working.
    const loading = !!st.caps?.health?.loading;
    pill.dataset.state = loading ? 'warn' : 'off';
    text.textContent = loading ? t('status.loading') : t('status.offline');
    return;
  }
  const sleeping = st.props?.is_sleeping || st.residency?.state === 'suspended';
  if (sleeping) { pill.dataset.state = 'warn'; text.textContent = t('status.suspended'); return; }
  const busy = busyChat || (num(st.load?.running) || 0) > 0;
  pill.dataset.state = busy ? 'busy' : 'ok';
  text.textContent = busy ? t('status.busy') : t('status.ready');
  $('#pillAction').title = sleeping ? t('rail.resume') : t('rail.suspend');
}

// Per-sensor availability: a card that does not expose a sensor (or a driver
// that hides it) shows "n/a" with the reason, never a borrowed or estimated value.
const NA_REASON = {
  'not supported': 'therm.naDriver',
  'not exposed by NVML': 'therm.naApi',
  'n/a': 'therm.naApi',
};

function renderThermal(st) {
  const list = $('#thermList');
  if (!list) return;
  const g = st && st.gpu;
  const src = $('#thermSrc');
  const limits = $('#thermLimits');
  if (!g || g.source === 'unavailable' || g.source === 'error') {
    list.innerHTML = `<div class="fact"><dt>${t('therm.title')}</dt>`
      + `<dd><span class="na" title="${(g && (g.error || g.reason?.core_c)) || 'NVML unavailable'}">n/a</span></dd></div>`;
    if (src) src.textContent = 'NVML unavailable';
    if (limits) limits.textContent = '';
    return;
  }
  if (src) src.textContent = 'NVML';
  const reasonText = (why) => {
    const key = NA_REASON[why];
    return key ? t(key) : (why || t('therm.naApi'));
  };
  const row = (label, value, unit, key) => {
    const ok = !!(g.available && g.available[key]) && value != null;
    const why = reasonText(g.reason && g.reason[key]);
    const shown = ok
      ? `<b>${unit === '°C' ? fmt1(value) : fmtInt(value)}</b> ${unit}`
      : `<span class="na" title="${why}">n/a</span>`;
    return `<div class="fact"><dt>${label}</dt><dd>${shown}</dd></div>`;
  };
  list.innerHTML = [
    row(t('therm.core'), g.core_c, '°C', 'core_c'),
    row(t('therm.hotspot'), g.hotspot_c, '°C', 'hotspot_c'),
    row(t('therm.memory'), g.memory_c, '°C', 'memory_c'),
    row(t('therm.board'), g.board_c, '°C', 'board_c'),
    row(t('therm.fans'), g.fan_pct, '%', 'fan_pct'),
  ].join('');
  const L = g.limits || {};
  const parts = [];
  if (L.slowdown != null) parts.push(t('therm.slowdown', { n: L.slowdown }));
  if (L.max_operating != null) parts.push(t('therm.maxop', { n: L.max_operating }));
  if (L.shutdown != null) parts.push(t('therm.shutdown', { n: L.shutdown }));
  if (g.fan_count) parts.push(t('therm.fanCount', { n: g.fan_count }));
  if (limits) limits.textContent = parts.join(' · ');
}

// The engine is an external service with an optional capability set. Anything
// the running build does not answer for is disabled or labelled here, so the
// dashboard cannot show a control that would fail or a number it never got.
function applyCaps(caps, up, cfg, engineOwned) {
  caps = caps || {};
  // suspend needs both: the endpoint present and the profile started with it
  const canSuspend = up && !!(caps.suspend ? caps.suspend.ok : (caps.residency && caps.residency.ok));
  ['#engSuspend', '#engResume', '#railSuspend', '#railResume'].forEach(sel => {
    const b = $(sel);
    if (!b) return;
    b.disabled = !canSuspend;
    b.title = canSuspend ? '' : t('caps.noSuspend');
  });

  // Start only makes sense when nothing is serving; Stop only when it is.
  const startBtn = $('#engStart');
  if (startBtn) {
    startBtn.disabled = up;
    startBtn.title = up ? t('caps.startDisabled') : '';
  }
  const stopBtn = $('#engStop');
  if (stopBtn) stopBtn.disabled = !up || !engineOwned;

  const metricsOk = !!(caps.metrics && caps.metrics.ok);
  const note = $('#metricsNote');
  if (note) {
    note.hidden = metricsOk;
    note.textContent = metricsOk ? ''
      : (caps.metrics && caps.metrics.reason === 'auth' ? t('caps.noMetricsAuth') : t('caps.noMetrics'));
  }

  const vision = caps.vision === undefined ? true : !!caps.vision;
  const img = $('#btnImage');
  if (img) {
    img.disabled = !vision;
    img.title = vision ? t('chat.attach') : t('caps.noVision');
  }

  if (cfg) {
    if ($('#cfgEngine')) $('#cfgEngine').textContent = cfg.engine_url || '—';
    if ($('#cfgUi')) $('#cfgUi').textContent = ':' + (cfg.ui_port ?? '—');
    if ($('#cfgCmd')) $('#cfgCmd').textContent = (cfg.start_command || []).join(' ') || t('cfg.none');
    if ($('#cfgFile')) $('#cfgFile').textContent = cfg.config_file || t('cfg.defaults');
  }
}

function renderState(st) {
  state = st;
  const m = st.metrics || {};
  const up = !!st.up;
  applyCaps(st.caps, up, st.config, st.engine_owned);
  if (up && !firstUpAt) firstUpAt = Date.now();
  if (!up) firstUpAt = null;

  const decodeTok = num(m.decode_tokens);
  const decodeSec = num(m.decode_seconds);
  const preTok = num(m.prefill_tokens);
  const preSec = num(m.prefill_seconds);
  const rounds = num(m.rounds);

  const dt = state.ts ? 2 : 2;
  const tpsDelta = up ? rate(m, prevCounters, dt, 'decode_tokens', 'decode_seconds') : null;
  const lastReq = st.load?.last_request || null;
  const lastTps = lastReq && lastReq.generation_wall_seconds > 0
    ? lastReq.completion_tokens / lastReq.generation_wall_seconds : null;
  const generating = (tpsDelta != null && tpsDelta > 0);
  const tps = tpsDelta ?? (decodeTok && decodeSec ? decodeTok / decodeSec : null);
  const tpsShown = generating ? tpsDelta : (lastTps ?? tps);
  const preDelta = up ? rate(m, prevCounters, dt, 'prefill_tokens', 'prefill_seconds') : null;
  const preAvg = preTok && preSec ? preTok / preSec : null;
  const tpr = decodeTok && rounds ? decodeTok / rounds : null;

  const vramUsed = num(m.vram_used), vramTotal = num(m.vram_total);
  const vramPct = pct(vramUsed, vramTotal);
  const kvUsed = num(m.kv_pages), kvTotal = num(m.kv_pages_total);
  const kvPct = pct(kvUsed, kvTotal);
  const ctxUsed = num(m.ctx_used), ctxTotal = num(m.ctx_total);

  if (up) {
    push('tps', tpsShown); push('tpr', tpr); push('pre', preDelta ?? preAvg);
    push('vram', vramUsed ? gib(vramUsed) : null); push('temp', num(m.temp));
    push('power', num(m.power)); push('util', num(m.gpu_util)); push('kv', kvUsed);
  }
  prevCounters = m;
  paintSparks();

  // status pill / chat header
  renderStatus(st);
  const chatSleeping = up && ((st.residency || {}).state === 'suspended' || !!(st.props && st.props.is_sleeping));
  $('#chatState').className = 'tag ' + (up ? ((busyChat || chatSleeping) ? 'warn' : 'good') : 'bad');
  $('#chatState').querySelector('span:last-child').textContent =
    !up ? t('status.offline') : (chatSleeping ? t('status.suspended') : t('chat.ready'));
  $('#chatModel').textContent = st.model?.id || '—';

  // chat rail
  $('#mTps').textContent = up ? fmt1(tpsShown) : '—';
  $('#mTpsFoot').textContent = up
    ? (generating ? (tpr ? `≈ ${fmt1(tpr)} tok / round` : t('rail.waiting'))
                  : (lastReq ? t('rail.lastReq', { n: fmtInt(lastReq.completion_tokens) }) : t('rail.waiting')))
    : t('rail.waiting');
  $('#mTpr').textContent = fmt1(tpr);
  $('#mPrefill').textContent = up ? fmt1((preDelta != null && preDelta > 0) ? preDelta : preAvg) + ' tok/s' : '—';
  $('#mGen').textContent = up ? fmtInt(decodeTok) : '—';
  $('#mKv').textContent = up ? `${fmtInt(kvUsed)} / ${fmtInt(kvTotal)}` : '—';
  $('#railModel').textContent = st.model?.id || '—';
  $('#railCtx').textContent = st.model?.max_model_len ? fmtInt(st.model.max_model_len) + ' tok' : (st.props?.n_ctx ? fmtInt(st.props.n_ctx) + ' tok' : '—');
  $('#railVram').textContent = vramUsed ? fmt1(gib(vramUsed)) + ' GiB' : '—';
  $('#railState').textContent = st.props?.is_sleeping ? t('state.suspended') : up ? t('state.idle') : t('state.offline');
  $('#railSpec').textContent = m.spec_backend ? `${m.spec_backend.toUpperCase()} K=${fmtInt(m.draft_window)}` : '—';

  // engine strip
  $('#esEngine').textContent = st.props?.build_info || 'ninfer-serve';
  const modelPath = st.props?.model_path || st.model_path || '';
  // ninfer does not publish the artifact path, so do not leave a dangling separator
  $('#esModel').textContent = st.model?.id ? (modelPath ? `${st.model.id} · ${modelPath}` : st.model.id) : '—';
  const sleeping = st.props?.is_sleeping || st.residency?.state === 'suspended';
  $('#esSpec').textContent = m.spec_backend ? `${m.spec_backend.toUpperCase()} · K=${fmtInt(m.draft_window)}` : '—';
  $('#esContext').textContent = st.model?.max_model_len ? fmtInt(st.model.max_model_len) + ' tok' : '—';
  $('#esKv').textContent = kvTotal ? `${fmtInt(kvUsed)} / ${fmtInt(kvTotal)} pages` : '—';
  $('#esUptime').textContent = num(st.load?.uptime_seconds) != null ? dur(st.load.uptime_seconds) : (firstUpAt && up ? dur((Date.now() - firstUpAt) / 1000) : '—');
  engineStateIcon(st);

  // metric grid
  $('#mgTps').textContent = up ? fmt1(tpsShown) : '—';
  $('#mgTpsFoot').textContent = generating ? (tpr ? `≈ ${fmt1(tpr)} tok / round` : '')
    : (lastReq ? t('rail.lastReq', { n: fmtInt(lastReq.completion_tokens) }) : '');
  $('#mgTpr').textContent = fmt1(tpr);
  $('#mgPre').textContent = up ? fmt1((preDelta != null && preDelta > 0) ? preDelta : preAvg) : '—';
  $('#mgPreFoot').textContent = preTok ? `${fmtInt(preTok)} tokens total` : '';
  $('#mgVram').textContent = vramUsed ? fmt1(gib(vramUsed)) : '—';
  $('#mgVramFoot').textContent = vramTotal ? `${fmt1(gib(vramUsed))} / ${fmt1(gib(vramTotal))} GiB` : '';
  const vramBar = $('#mgVramBar');
  vramBar.style.width = (vramPct ?? 0) + '%';
  vramBar.className = vramPct > 92 ? 'bad' : vramPct > 80 ? 'warn' : '';
  const coreTemp = num(st.gpu?.core_c) ?? num(m.temp);
  $('#mgTemp').textContent = coreTemp != null ? fmt1(coreTemp) : '—';
  $('#mgTempFoot').textContent = coreTemp != null ? (coreTemp > 83 ? t('m.hot') : coreTemp > 75 ? t('m.warm') : t('m.ok')) : '';
  renderThermal(st);
  $('#mgPower').textContent = num(m.power) != null ? fmtInt(m.power) : '—';
  $('#mgPowerFoot').textContent = num(m.power) != null ? `${Math.round((m.power / 350) * 100)}% of 350 W` : '';
  $('#mgUtil').textContent = num(m.gpu_util) != null ? fmtInt(m.gpu_util) : '—';
  $('#mgUtilFoot').textContent = num(m.gpu_util) != null ? (m.gpu_util > 90 ? 'saturated' : 'idle slots') : '';
  $('#mgKv').textContent = kvTotal ? `${fmtInt(kvUsed)}/${fmtInt(kvTotal)}` : '—';
  $('#mgKvFoot').textContent = kvTotal ? `${Math.round(kvPct)}% of the KV pool` : '';
  const kvBar = $('#mgKvBar');
  kvBar.style.width = (kvPct ?? 0) + '%';
  kvBar.className = kvPct > 90 ? 'warn' : '';

  // context + queues
  $('#ctxUsed').textContent = ctxUsed ? bytesText(ctxUsed) : '—';
  $('#ctxTotal').textContent = ctxTotal ? bytesText(ctxTotal) : bytesText(num(st.load?.capacity?.host_context_bytes));
  if (ctxUsed) peakCtx = Math.max(peakCtx, ctxUsed);
  $('#ctxPeak').textContent = bytesText(num(st.residency?.peak_host_context_bytes) ?? (peakCtx || null));
  const kvTokens = num(st.load?.occupancy?.device_main_kv_tokens);
  $('#ctxKvTok').textContent = kvTokens != null ? fmtInt(kvTokens) + ' tokens' : '—';
  $('#ctxBar').style.width = (pct(ctxUsed, ctxTotal) ?? 0) + '%';

  const load = st.load || {};
  const pick = (...names) => {
    for (const path of names) {
      const parts = path.split('.');
      let v = load;
      for (const p of parts) { v = v == null ? null : v[p]; }
      if (num(v) != null) return v;
    }
    return null;
  };
  $('#qRunning').textContent = fmtInt(pick('running', 'requests.running', 'occupancy.running'));
  $('#qWaiting').textContent = fmtInt(pick('waiting', 'requests.waiting', 'deferred'));
  $('#qPrefill').textContent = fmtInt(pick('prefilling', 'requests.prefilling'));
  $('#qReady').textContent = fmtInt(pick('decode_ready', 'requests.decode_ready'));
  $('#qHost').textContent = num(pick('host.active_seconds')) != null ? fmt1(pick('host.active_seconds')) + ' s' : '—';
  $('#qDev').textContent = num(pick('host.device_wait_seconds')) != null ? fmt1(pick('host.device_wait_seconds')) + ' s' : '—';

  // engine tab
  $('#engPill').textContent = up ? (sleeping ? t('state.suspended') : t('state.idle')) : t('state.offline');
  $('#engPill').className = 'tag ' + (up ? (sleeping ? 'warn' : 'good') : 'bad');
  $('#eUp').textContent = st.upstream || '—';
  $('#eId').textContent = st.model?.id || '—';
  $('#ePath').textContent = st.props?.model_path || st.model_path || '—';
  $('#eCtx').textContent = st.model?.max_model_len ? fmtInt(st.model.max_model_len) + ' tok' : (st.props?.n_ctx ? fmtInt(st.props.n_ctx) : '—');
  $('#eSpec').textContent = m.spec_backend ? `${m.spec_backend} · draft window ${fmtInt(m.draft_window)}` : '—';
  $('#eKv').textContent = kvTotal ? `${fmtInt(kvUsed)} / ${fmtInt(kvTotal)}` : '—';
  $('#eDecoded').textContent = fmtInt(decodeTok);
  $('#eSleep').textContent = st.props?.is_sleeping == null ? '—' : (st.props.is_sleeping ? 'true' : 'false');
  $('#ePid').textContent = (st.pids && st.pids.length) ? st.pids.join(', ') : '—';
  $('#cIn').textContent = (st.model?.architecture?.input_modalities || []).join(', ') || '—';
  $('#cOut').textContent = (st.model?.architecture?.output_modalities || []).join(', ') || '—';
  $('#cCtx').textContent = st.model?.context_length ? fmtInt(st.model.context_length) + ' tok' : '—';
  $('#bVer').textContent = (st.props?.build_info || '—').split(' ').slice(0, 2).join(' ') || '—';
  $('#bBuild').textContent = st.props?.build_info || '—';

  // model tab
  $('#mdLoaded').textContent = up ? (sleeping ? t('model.suspended') : t('model.loadedTag')) : t('model.notLoaded');
  $('#mdLoaded').className = 'tag ' + (up ? (sleeping ? 'warn' : 'good') : 'bad');
  $('#mdId').textContent = st.model?.id || '—';
  $('#mdPath').textContent = st.props?.model_path || st.model_path || '—';
  $('#mdCtx').textContent = st.model?.max_model_len ? fmtInt(st.model.max_model_len) + ' tok' : '—';
  $('#mdMod').textContent = (st.model?.architecture?.input_modalities || []).join(', ') || '—';
  $('#mdVram').textContent = vramUsed ? fmt1(gib(vramUsed)) + ' GiB' : '—';

  // banner: one missed poll is normal while the engine restarts or loads a
  // model, so the warning only appears on the second consecutive miss.
  const banner = $('#connBanner');
  failStreak = up ? 0 : failStreak + 1;
  banner.hidden = up || failStreak < 2;
  if (!banner.hidden) {
    // the server reports its own engine URL; never assume a port here
    const url = st.upstream || 'the engine';
    const msg = t('err.offline', { url });
    // never render an empty strip: fall back to a literal if a translation is missing
    $('#connBannerText').textContent = (msg && msg !== 'err.offline') ? msg
      : 'Engine is not reachable at ' + url + '. Start it from the Engine tab.';
  }
}

let pollTimer = null;
async function poll(silent) {
  try {
    const st = await api('/api/state');
    renderState(st);
  } catch (e) {
    renderStatus(null);
    if (!silent) toast(t('toast.failed', { msg: e.message }), 'err');
  }
}
function startPolling() {
  clearInterval(pollTimer);
  // keep polling even when the tab is in the background: the engine is local,
  // a 2 s /api/state call costs nothing, and returning to the tab should show
  // an already-warm history instead of an empty sparkline.
  pollTimer = setInterval(() => poll(true), 2000);
}
document.addEventListener('visibilitychange', () => { if (!document.hidden) poll(true); });

/* ─────────────────────────── engine control ─────────────────────────── */
async function engineAction(path, okKey, opts) {
  const btns = $$('#engStart,#engStop,#engSuspend,#engResume,#railSuspend,#railResume,#pillAction');
  btns.forEach((b) => { b.disabled = true; });
  try {
    const data = await api(path, { method: 'POST' });
    if (data && data.ok === false) {
      const status = data.status;
      if (status === 409) toast(t('toast.busy'), 'err');
      else if (status === 400 && JSON.stringify(data).includes('suspend')) toast(t('toast.suspendDisabled'), 'err');
      else toast(t('toast.failed', { msg: JSON.stringify(data.error || data).slice(0, 160) }), 'err');
    } else if (okKey) {
      toast(t(okKey), 'ok');
    }
    setTimeout(() => poll(true), 400);
  } catch (e) {
    toast(t('toast.failed', { msg: e.message }), 'err');
  } finally {
    btns.forEach((b) => { b.disabled = false; });
    // re-apply the capability rules so a finished action cannot leave Start
    // clickable while the engine is up or suspend clickable when it is off
    if (state) applyCaps(state.caps, !!state.up, state.config, state.engine_owned);
  }
}

$('#engStart').addEventListener('click', async () => {
  toast(t('state.loading'), 'ok');
  await engineAction('/api/engine/start', 'toast.engineUp');
  // engine needs ~11 s to load the artifact; poll until healthy
  for (let i = 0; i < 30; i++) {
    await new Promise((r) => setTimeout(r, 1000));
    try { const st = await api('/api/state'); renderState(st); if (st.up) break; } catch (_) {}
  }
});
$('#engStop').addEventListener('click', () => engineAction('/api/engine/stop', 'toast.engineDown'));
$('#engSuspend').addEventListener('click', () => engineAction('/api/model/suspend', 'toast.suspended'));
$('#engResume').addEventListener('click', () => engineAction('/api/model/resume', 'toast.resumed'));
$('#railSuspend').addEventListener('click', () => engineAction('/api/model/suspend', 'toast.suspended'));
$('#railResume').addEventListener('click', () => engineAction('/api/model/resume', 'toast.resumed'));
$('#pillAction').addEventListener('click', () => {
  const sleeping = state && (state.props?.is_sleeping || state.residency?.state === 'suspended');
  engineAction(sleeping ? '/api/model/resume' : '/api/model/suspend', sleeping ? 'toast.resumed' : 'toast.suspended');
});
$('#liveRefresh').addEventListener('click', () => poll(false));

/* ─────────────────────────── artifacts ─────────────────────────── */
async function loadLog() {
  try {
    const data = await api('/api/log');
    const el = $('#engLog');
    el.textContent = (data.lines || []).join('\n') || '—';
    el.scrollTop = el.scrollHeight;
    $('#logPath').textContent = data.path || '—';
  } catch (e) {
    $('#engLog').textContent = e.message;
  }
}
$('#logRefresh').addEventListener('click', loadLog);

let artifactsLoaded = false;
async function loadArtifacts(force) {
  if (artifactsLoaded && !force) return;
  try {
    const data = await api('/api/files');
    const rows = data.files || [];
    const body = $('#artBody');
    if (!rows.length) {
      body.innerHTML = `<tr class="empty-row"><td colspan="2">${t('model.noFiles')}</td></tr>`;
    } else {
      body.innerHTML = rows.map((r) => {
        const loaded = data.loaded && data.loaded.endsWith(r.name);
        return `<tr><td class="mono tiny">${r.name}${loaded ? ' <span class="tag good">●</span>' : ''}</td><td>${bytesText(r.bytes)}</td></tr>`;
      }).join('');
    }
    $('#artTotal').textContent = data.disk_free
      ? t('model.freeDisk', { free: bytesText(data.disk_free), total: bytesText(data.disk_total) })
      : '';
    artifactsLoaded = true;
  } catch (e) {
    $('#artBody').innerHTML = `<tr class="empty-row"><td colspan="2">${e.message}</td></tr>`;
  }
}

/* ─────────────────────────── chat ─────────────────────────── */
const messages = [];          // {role, content, images?: [dataUrl]}
let attachments = [];         // pending images
let busyChat = false;
let abortCtrl = null;

function renderTurns() {
  const n = messages.length;
  $('#chatTurns').textContent = n ? t(n === 1 ? 'chat.turnsOne' : 'chat.turns', { n }) : '';
}

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

function messageNode(msg) {
  const wrap = el('div', 'msg ' + (msg.role === 'user' ? 'user' : 'assistant'));
  wrap.appendChild(el('div', 'avatar', msg.role === 'user' ? 'EU' : 'AI'));
  const body = el('div', 'body');
  const who = el('div', 'who', msg.role === 'user' ? t('chat.you') : t('chat.ai'));
  body.appendChild(who);

  if (msg.images && msg.images.length) {
    const att = el('div', 'attachments');
    msg.images.forEach((src) => {
      const img = document.createElement('img');
      img.src = src;
      att.appendChild(img);
    });
    body.appendChild(att);
  }
  if (msg.reasoning) {
    const det = document.createElement('details');
    det.className = 'thinking';
    const sum = el('summary', null, t('chat.thinkingTokens', { n: msg.reasoning.length }));
    det.appendChild(sum);
    det.appendChild(el('div', 'think-body', msg.reasoning));
    body.appendChild(det);
  }
  const bubble = el('div', 'bubble', msg.content || '');
  body.appendChild(bubble);
  if (msg.stats) body.appendChild(statsRow(msg.stats));
  wrap.appendChild(body);
  return { wrap, bubble, body };
}

function statsRow(stats) {
  const row = el('div', 'stats');
  const chip = (label, value, unit, title, cls) => {
    const c = el('span', 'chip' + (cls ? ' ' + cls : ''));
    c.title = title || '';
    c.innerHTML = `<span class="k">${label}</span><b>${value}</b>${unit ? ' ' + unit : ''}`;
    row.appendChild(c);
  };
  if (stats.outputTokens != null) {
    chip(t('stat.gen'), fmtInt(stats.outputTokens), 'tok', t('stat.genTitle'));
  }
  if (stats.tps != null) {
    // the number people quote as "speed" is decode throughput, never the prompt
    chip(t('stat.decode'), fmt1(stats.tps), 'tok/s', t('stat.decodeTitle'), 'good');
  }
  if (stats.prefill != null) {
    chip(t('stat.prefill'), fmt1(stats.prefill), 'tok/s', t('stat.prefillTitle'));
  }
  if (stats.accept != null) {
    chip(t('stat.accept'), Math.round(stats.accept * 100) + '%', '', t('stat.acceptTitle'));
  }
  if (stats.ttft != null) {
    chip(t('stat.ttft'), fmt1(stats.ttft), 's', t('stat.ttftTitle'));
  }
  if (stats.total != null) {
    chip(t('stat.total'), fmt1(stats.total), 's', t('stat.totalTitle'));
  }
  return row;
}

function scrollLog() {
  const log = $('#log');
  log.scrollTop = log.scrollHeight;
}

function renderChat() {
  const log = $('#log');
  log.innerHTML = '';
  if (!messages.length) {
    const empty = el('div', 'empty');
    empty.innerHTML = `<svg class="ico"><use href="/icons.svg#i-hub"></use></svg>`;
    empty.appendChild(el('strong', null, t('chat.emptyTitle')));
    empty.appendChild(el('span', null, t('chat.emptyBody')));
    log.appendChild(empty);
  } else {
    messages.forEach((m) => log.appendChild(messageNode(m).wrap));
  }
  renderTurns();
  scrollLog();
}

function renderAttachments() {
  const row = $('#attachRow');
  row.innerHTML = '';
  row.hidden = attachments.length === 0;
  attachments.forEach((src, i) => {
    const chip = el('div', 'attach-chip');
    const img = document.createElement('img');
    img.src = src;
    chip.appendChild(img);
    const x = el('button', null, '×');
    x.type = 'button';
    x.addEventListener('click', () => { attachments.splice(i, 1); renderAttachments(); });
    chip.appendChild(x);
    row.appendChild(chip);
  });
}

async function addFiles(fileList) {
  const files = Array.from(fileList || []).slice(0, 4);
  for (const f of files) {
    if (!f.type.startsWith('image/')) continue;
    const dataUrl = await new Promise((resolve, reject) => {
      const fr = new FileReader();
      fr.onload = () => resolve(fr.result);
      fr.onerror = reject;
      fr.readAsDataURL(f);
    });
    attachments.push(dataUrl);
  }
  renderAttachments();
  $('#input').focus();
}

$('#btnImage').addEventListener('click', () => $('#file').click());
$('#file').addEventListener('change', (e) => { addFiles(e.target.files); e.target.value = ''; });
$('#input').addEventListener('paste', (e) => {
  const items = Array.from(e.clipboardData?.items || []).filter((i) => i.type.startsWith('image/'));
  if (items.length) addFiles(items.map((i) => i.getAsFile()).filter(Boolean));
});
document.addEventListener('dragover', (e) => { if (!$('#view-chat').hidden) e.preventDefault(); });
document.addEventListener('drop', (e) => {
  if ($('#view-chat').hidden) return;
  e.preventDefault();
  addFiles(e.dataTransfer?.files);
});

function autoGrow() {
  const ta = $('#input');
  ta.style.height = 'auto';
  ta.style.height = Math.min(180, ta.scrollHeight) + 'px';
}
$('#input').addEventListener('input', autoGrow);
$('#input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('#form').requestSubmit(); }
});
$('#btnFocus').addEventListener('click', () => { showView('chat'); $('#input').focus(); });
$('#btnClear').addEventListener('click', () => {
  if (busyChat) return;
  messages.length = 0;
  attachments = [];
  renderAttachments();
  renderChat();
});

function buildPayload(text) {
  const content = [];
  if (text) content.push({ type: 'text', text });
  attachments.forEach((src) => content.push({ type: 'image_url', image_url: { url: src } }));
  const history = messages
    .filter((m) => m.content || (m.images && m.images.length))
    .map((m) => {
      if (m.role === 'user' && m.images && m.images.length) {
        const parts = [];
        if (m.content) parts.push({ type: 'text', text: m.content });
        m.images.forEach((src) => parts.push({ type: 'image_url', image_url: { url: src } }));
        return { role: m.role, content: parts };
      }
      return { role: m.role, content: m.content };
    });
  history.push({ role: 'user', content: content.length > 1 || (content[0] && content[0].type !== 'text') ? content : text });
  return {
    messages: history,
    stream: true,
    stream_options: { include_usage: true },
    temperature: Number($('#optTemp').value) || 0,
    max_tokens: Number($('#optMax').value) || 1024,
    reasoning_effort: $('#optThink').value === 'none' ? 'none' : $('#optThink').value
  };
}

$('#form').addEventListener('submit', async (e) => {
  e.preventDefault();
  if (busyChat) { if (abortCtrl) abortCtrl.abort(); return; }
  const text = $('#input').value.trim();
  if (!text && !attachments.length) return;

  const images = attachments.slice();
  const payload = buildPayload(text);
  messages.push({ role: 'user', content: text, images });
  attachments = [];
  renderAttachments();
  $('#input').value = '';
  autoGrow();
  renderChat();

  const target = { role: 'assistant', content: '', reasoning: '', stats: null };
  messages.push(target);
  renderChat();

  const nodes = $$('#log .msg');
  const node = nodes[nodes.length - 1];
  const bubble = node.querySelector('.bubble');
  bubble.classList.add('streaming');

  busyChat = true;
  abortCtrl = new AbortController();
  const btn = $('#btnSend');
  btn.classList.remove('btn-primary');
  btn.innerHTML = `<svg class="ico"><use href="/icons.svg#i-stop"></use></svg><span>${t('chat.stop')}</span>`;
  $('#chatState').className = 'tag warn';
  $('#chatState').querySelector('span:last-child').textContent = t('chat.busy');

  const started = performance.now();
  let firstTokenAt = null;
  let outTokens = 0;
  let usage = null;
  let timings = null;   // engine-reported prompt/decode split, far better than our own clock

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Ninfer-Studio': '1' },
      body: JSON.stringify(payload),
      signal: abortCtrl.signal
    });
    if (!res.ok) {
      const body = await res.text();
      throw new Error(body.slice(0, 300) || res.status);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    let reasoningEl = null, thinkBody = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const raw of lines) {
        const line = raw.trim();
        if (!line || line.startsWith(':') || line.startsWith('event:')) continue;
        if (!line.startsWith('data:')) continue;
        const data = line.slice(5).trim();
        if (data === '[DONE]') continue;
        let json;
        try { json = JSON.parse(data); } catch (_) { continue; }
        if (json.usage) usage = json.usage;
        if (json.timings) timings = json.timings;
        const delta = json.choices?.[0]?.delta || {};
        if (delta.reasoning_content) {
          target.reasoning += delta.reasoning_content;
          if (!reasoningEl) {
            const det = document.createElement('details');
            det.className = 'thinking';
            det.open = true;
            const sum = el('summary', null, t('chat.thinkingOpen'));
            thinkBody = el('div', 'think-body');
            det.appendChild(sum);
            det.appendChild(thinkBody);
            node.querySelector('.body').insertBefore(det, bubble);
            reasoningEl = det;
          }
          thinkBody.textContent = target.reasoning;
        }
        if (delta.content) {
          if (firstTokenAt == null) {
            firstTokenAt = performance.now();
            // collapse the thinking block once the answer starts
            if (reasoningEl) reasoningEl.open = false;
          }
          target.content += delta.content;
          bubble.textContent = target.content;
          scrollLog();
        }
        outTokens = json.choices?.[0]?.usage?.completion_tokens ?? outTokens;
      }
    }
    bubble.classList.remove('streaming');
    const elapsed = (performance.now() - started) / 1000;
    const out = usage?.completion_tokens ?? outTokens ?? null;
    const prompt = usage?.prompt_tokens ?? null;
    const ttft = firstTokenAt ? (firstTokenAt - started) / 1000 : null;
    // Prefer the engine's own numbers: timings.predicted_* is pure decode,
    // timings.prompt_* is pure prompt reading (incl. cache hits).
    const decodeTps = timings?.predicted_per_second ?? null;
    const prefillTps = timings?.prompt_per_second ?? null;
    const accept = timings?.draft_n ? timings.draft_n_accepted / timings.draft_n : null;
    const tps = decodeTps ?? (out && firstTokenAt ? out / ((performance.now() - firstTokenAt) / 1000) : null);
    const prefill = prefillTps ?? (prompt && ttft && ttft > 0.01 ? prompt / ttft : null);
    target.stats = {
      outputTokens: timings?.predicted_n ?? out,
      promptTokens: prompt,
      tps,
      prefill,
      accept,
      ttft,
      total: elapsed
    };
    const who = node.querySelector('.who');
    if (who) who.textContent = t('chat.ai');
    node.querySelector('.body').appendChild(statsRow(target.stats));
  } catch (err) {
    bubble.classList.remove('streaming');
    if (err.name === 'AbortError') {
      toast(t('toast.stopped'));
    } else {
      target.content = target.content || '';
      bubble.textContent = target.content + (target.content ? '\n\n' : '') + '⚠ ' + err.message;
      toast(t('toast.failed', { msg: err.message }), 'err');
    }
  } finally {
    busyChat = false;
    abortCtrl = null;
    btn.classList.add('btn-primary');
    btn.innerHTML = `<svg class="ico"><use href="/icons.svg#i-send"></use></svg><span>${t('chat.send')}</span>`;
    $('#chatState').className = 'tag good';
    $('#chatState').querySelector('span:last-child').textContent = t('chat.ready');
    scrollLog();
    poll(true);
  }
});

/* ─────────────────────────── boot ─────────────────────────── */
applyLang();
showView(location.hash.slice(1) || 'chat');
renderChat();
poll(false);
startPolling();
loadArtifacts();
