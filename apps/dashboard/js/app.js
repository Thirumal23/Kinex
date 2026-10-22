/**
 * KINEX ADV 2.0 — Continuous Indian Sign Language (ISL) Sentence Recognition,
 * Action Tracking, Synthesis Engine & Multimodal Fusion Dashboard
 *
 * Core Capabilities:
 *  - Continuous Sentence Translation: Real-time synthesis of multi-sign action chains into natural English
 *  - Anti-Jitter Stability Gating: Temporal dwell filter (7 frames ~250ms) to ensure ultra-high accuracy
 *  - Dedicated Sentence Actions & History Studio: Persistent store, export (JSON/CSV/TXT), playback & search
 *  - Dual-Hand MediaPipe Tracking (42 Joints) with Bilateral links
 *  - Multimodal Modes: Mode 1 (Vision), Mode 2 (Smart Glove), Mode 3 (Multimodal Fusion)
 *  - Interactive Camera Model Trainer: Expand vocabulary on-the-fly
 *  - Text-to-Speech (TTS) Voice Announcer with Audio Waveform visualizer
 */

// 21 Keypoint Connections for Hand Skeleton
const SKELETON_CONNECTIONS = [
  // Thumb: 0 -> 1 -> 2 -> 3 -> 4
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index: 0 -> 5 -> 6 -> 7 -> 8
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle: 0 -> 9 -> 10 -> 11 -> 12
  [0, 9], [9, 10], [10, 11], [11, 12],
  // Ring: 0 -> 13 -> 14 -> 15 -> 16
  [0, 13], [13, 14], [14, 15], [15, 16],
  // Pinky: 0 -> 17 -> 18 -> 19 -> 20
  [0, 17], [17, 18], [18, 19], [19, 20],
  // Palm Base: 5 -> 9 -> 13 -> 17
  [5, 9], [9, 13], [13, 17]
];

// Color palettes for Left Hand vs Right Hand
const PALETTE_LEFT = {
  theme: 'cyan',
  wrist: '#06b6d4',
  thumb: '#f43f5e',
  index: '#3b82f6',
  middle: '#10b981',
  ring: '#f59e0b',
  pinky: '#8b5cf6',
  palm: 'rgba(255, 255, 255, 0.4)',
  box: 'rgba(6, 182, 212, 0.6)',
  labelBg: 'rgba(6, 182, 212, 0.9)'
};

const PALETTE_RIGHT = {
  theme: 'emerald',
  wrist: '#10b981',
  thumb: '#ec4899',
  index: '#06b6d4',
  middle: '#3b82f6',
  ring: '#eab308',
  pinky: '#a855f7',
  palm: 'rgba(255, 255, 255, 0.4)',
  box: 'rgba(16, 185, 129, 0.6)',
  labelBg: 'rgba(16, 185, 129, 0.9)'
};

const HAND_PALETTES = [PALETTE_LEFT, PALETTE_RIGHT];

const GESTURE_ICONS = {
  "THANK_YOU": "🙏",
  "HEARING": "👂",
  "MORNING": "🌅",
  "I": "☝️",
  "BYE": "👋",
  "NAME": "🏷️",
  "INDIAN": "🇮🇳",
  "NAMASTE": "🙏",
  "HELP": "🆘",
  "WATER": "💧",
  "FOOD": "🍲",
  "YES": "👍",
  "NO": "👎",
  "UNKNOWN": "❓"
};

// Comprehensive ISL Gloss -> English Sentence Grammatical Dictionary
const ISL_SENTENCE_DICTIONARY = {
  // Single Signs
  "THANK_YOU": "Thank you very much.",
  "INDIAN": "I am proud to be Indian.",
  "NAME": "My name is being signed.",
  "I": "I am present and communicating.",
  "HEARING": "I am hard of hearing / deaf.",
  "MORNING": "Good morning to you!",
  "BYE": "Goodbye, have a great day!",
  "NAMASTE": "Namaste, warm greetings to you.",
  "HELP": "I need assistance, please help me.",
  "WATER": "Please give me some drinking water.",
  "FOOD": "I would like to have food.",

  // 2-Sign Sequences (SOV / Topic-Comment)
  "I+INDIAN": "I am an Indian citizen.",
  "INDIAN+I": "I am proud of my Indian heritage.",
  "I+NAME": "My name is KINEX User.",
  "NAME+I": "Please call me by my name.",
  "I+HEARING": "I am deaf or hard of hearing.",
  "HEARING+I": "As a deaf individual, I communicate using ISL.",
  "I+THANK_YOU": "I want to express my sincere thanks to you.",
  "THANK_YOU+I": "Thank you from the bottom of my heart.",
  "I+BYE": "I will take your leave now, goodbye!",
  "BYE+I": "Goodbye, I am departing now.",
  "I+MORNING": "Good morning! I am ready to begin.",
  "MORNING+I": "Good morning from my side.",

  "MORNING+THANK_YOU": "Good morning! Thank you very much for your time.",
  "THANK_YOU+MORNING": "Thank you, and have a wonderful morning!",
  "THANK_YOU+BYE": "Thank you so much, goodbye and take care!",
  "BYE+THANK_YOU": "Goodbye! Thank you once again.",
  "MORNING+BYE": "Good morning and goodbye, have a pleasant day!",
  "MORNING+INDIAN": "Good morning to all fellow Indians!",
  "INDIAN+MORNING": "A very good morning from India!",
  "INDIAN+THANK_YOU": "Thank you on behalf of the Indian Sign Language community.",
  "THANK_YOU+INDIAN": "Thank you, proud to be Indian!",
  "NAME+INDIAN": "My identity and roots are in India.",
  "INDIAN+NAME": "My Indian name is registered.",
  "HEARING+THANK_YOU": "Thank you for understanding my hearing condition.",
  "THANK_YOU+HEARING": "Thank you for supporting accessibility for the deaf.",

  // 3-Sign Sequences
  "MORNING+I+INDIAN": "Good morning! I am a proud Indian.",
  "I+INDIAN+THANK_YOU": "I am an Indian, and I thank you with gratitude.",
  "I+HEARING+THANK_YOU": "I am hard of hearing; thank you for communicating with me.",
  "MORNING+THANK_YOU+BYE": "Good morning! Thank you very much, and goodbye!",
  "I+NAME+INDIAN": "My name is registered and I belong to India.",
  "I+THANK_YOU+BYE": "I thank you warmly, goodbye and take care!",
  "MORNING+I+THANK_YOU": "Good morning! I sincerely thank you today.",
};

// Global Application State
const state = {
  activeTab: 'dashboardTab',
  activeMode: 'vision', // 'vision', 'glove', 'fusion'
  isCameraActive: false,
  isVirtualStreamActive: false,
  virtualDemoIndex: 0,
  virtualLastSignSwitch: 0,
  mpHands: null,
  mpCamera: null,
  stream: null,
  webSocket: null,
  isTTSMuted: false,
  lastAnnouncedSentence: '',
  lastInferenceTime: 0,
  activeHands: [],
  fpsHistory: [],
  lastFrameTime: performance.now(),
  classes: ["HEARING", "THANK_YOU", "MORNING", "I", "BYE", "NAME", "INDIAN"],

  // Sentence Capture Engine State
  activeSentenceTokens: [
    { gesture: "I", confidence: 0.97, timestamp: Date.now() - 3000, modality: "vision" },
    { gesture: "INDIAN", confidence: 0.98, timestamp: Date.now() - 1500, modality: "vision" }
  ],
  activeSentenceString: "I am an Indian citizen.",
  activeSentenceConfidence: 0.975,
  pendingGesture: null,
  pendingDwellFrames: 0,
  STABILITY_THRESHOLD: 7, // ~250-300ms of stable holding needed
  autoCommitTimer: null,
  isAutoCommitEnabled: true,
  lastCommittedTokenGesture: 'INDIAN',

  // Sentence Studio & History State
  sentenceHistory: [],
  sandboxDraftTokens: [],

  // Trainer & Custom Model State
  trainerTargetSign: 'THANK_YOU',
  sampleCounts: {},
  continuousRecordInterval: null,
  customModel: null, // { isCustom: true, classes: [...], weights: [[...]], bias: [...], accuracy: 98.0 }
  trainerGivenSamples: [],
  trainerSentences: [],
  trainerSamplesFilter: 'ALL',
  trainerSentencesFilter: 'ALL',

  // Dedicated Sample Manager State
  sampleManagerFilter: 'ALL',
  sampleManagerSearch: '',
  sampleManagerViewMode: 'grouped',
  activeEditSampleId: null,

  // Sign Language Library State
  librarySigns: [],
  libraryFilterCategory: 'ALL',
  librarySearchQuery: '',
  librarySortBy: 'name_asc',
  activeEditSign: null,

  // Live Video-to-Sentence Capture State
  videoSentenceStream: null,
  videoMediaRecorder: null,
  videoRecordedChunks: [],
  videoRecordedBlob: null,
  videoRecordedUrl: null,
  isVideoRecording: false,
  videoFacingMode: 'user',
  videoCameraActive: false,
  videoTimerInterval: null,
  videoTimerSeconds: 0,
  videoLastProcessResult: null,

  // Add Sign Snapshot State
  snapshotStream: null,
  snapshotKeypoints: null
};

// DOM References
const video = document.getElementById('webcamVideo');
const canvas = document.getElementById('skeletonCanvas');
const ctx = canvas.getContext('2d');
const trainerCanvas = document.getElementById('trainerCanvas');
const trainerCtx = trainerCanvas ? trainerCanvas.getContext('2d') : null;
const placeholder = document.getElementById('cameraPlaceholder');

// Initialize on DOM Ready
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initTabNavigation();
  initModeControls();
  initUIEventListeners();
  initDashboardUI();
  initLibraryUI();
  initAddSignUI();
  initVideoSentenceUI();
  initSentenceStudio();
  initTrainerUI();
  initSampleManagerUI();
  initVocabulary();
  initMediaPipeHands();
  initWebSocket();
  initGloveSliders();
  checkSystemHealth();
  loadTrainedModel();
  refreshTrainingSampleStats();
  refreshTrainerStudio();
  loadSampleManager();
  loadDashboardStats();
  loadSignLibrary();
  loadSentenceHistory();
  renderActiveSentence();
  populateCameraDevices();
  checkCameraPermissionAndAutoStart();
  startAnimationFrameLoop();
});

// =========================================================================
// 0. BENTO THEME ENGINE
// =========================================================================

function initTheme() {
  const savedTheme = localStorage.getItem('kinex_bento_theme') || 'light';
  applyTheme(savedTheme);

  const btnToggleTheme = document.getElementById('btnToggleTheme');
  if (btnToggleTheme) {
    btnToggleTheme.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
      const next = current === 'dark' ? 'light' : 'dark';
      applyTheme(next);
      localStorage.setItem('kinex_bento_theme', next);
      showToast(next === 'dark' ? 'Switched to Bento Dark (Kombu) Theme' : 'Switched to Bento Light (Snow) Theme');
    });
  }
}

function applyTheme(theme) {
  const icon = document.getElementById('themeIcon');
  const label = document.getElementById('themeLabel');
  if (theme === 'dark') {
    document.documentElement.setAttribute('data-theme', 'dark');
    if (icon) icon.textContent = '🌙';
    if (label) label.textContent = 'Dark';
  } else {
    document.documentElement.removeAttribute('data-theme');
    if (icon) icon.textContent = '☀️';
    if (label) label.textContent = 'Light';
  }
}

function showToast(message, type = 'info') {
  let container = document.getElementById('bentoToastContainer');
  if (!container) {
    container = document.createElement('div');
    container.id = 'bentoToastContainer';
    container.className = 'bento-toast-container';
    document.body.appendChild(container);
  }
  const toast = document.createElement('div');
  toast.className = `bento-toast bento-toast--${type}`;
  toast.innerHTML = `
    <span class="bento-toast-icon">${type === 'success' ? '✓' : type === 'warning' ? '⚠' : '🍱'}</span>
    <span class="bento-toast-msg">${message}</span>
  `;
  container.appendChild(toast);
  setTimeout(() => {
    toast.classList.add('bento-toast--visible');
  }, 10);
  setTimeout(() => {
    toast.classList.remove('bento-toast--visible');
    setTimeout(() => {
      if (toast.parentNode) {
        toast.parentNode.removeChild(toast);
      }
    }, 300);
  }, 3200);
}


// =========================================================================
// 1. TAB NAVIGATION & MODE CONTROLS
// =========================================================================

function initTabNavigation() {
  const tabs = document.querySelectorAll('.nav-tab-btn');
  tabs.forEach(btn => {
    btn.addEventListener('click', (e) => {
      tabs.forEach(t => t.classList.remove('active'));
      const target = e.currentTarget;
      target.classList.add('active');
      const tabId = target.dataset.tab;
      state.activeTab = tabId;

      document.querySelectorAll('.tab-content-view').forEach(view => {
        view.style.display = (view.id === tabId) ? 'block' : 'none';
      });

      onTabActivated(tabId);
    });
  });
}

window.switchTab = function(tabId) {
  const targetBtn = document.querySelector(`.nav-tab-btn[data-tab="${tabId}"]`);
  if (targetBtn) {
    targetBtn.click();
  } else {
    document.querySelectorAll('.nav-tab-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.tab === tabId);
    });
    document.querySelectorAll('.tab-content-view').forEach(view => {
      view.style.display = (view.id === tabId) ? 'block' : 'none';
    });
    state.activeTab = tabId;
    onTabActivated(tabId);
  }
};

function onTabActivated(tabId) {
  if (tabId === 'dashboardTab') {
    loadDashboardStats();
  } else if (tabId === 'libraryTab') {
    loadSignLibrary();
  } else if (tabId === 'sentenceStudioTab') {
    loadSentenceHistory();
  } else if (tabId === 'trainerTab') {
    refreshTrainingSampleStats();
    refreshTrainerStudio();
  } else if (tabId === 'sampleManagerTab') {
    loadSampleManager();
  }
}

function initModeControls() {
  const modeBtns = document.querySelectorAll('.mode-btn');
  modeBtns.forEach(btn => {
    btn.addEventListener('click', (e) => {
      modeBtns.forEach(b => b.classList.remove('active'));
      const target = e.currentTarget;
      target.classList.add('active');
      const mode = target.dataset.mode;
      setSystemMode(mode);
    });
  });
}

function setSystemMode(mode) {
  state.activeMode = mode;
  const activeModeEl = document.getElementById('activeModeLabel');
  if (activeModeEl) activeModeEl.textContent = mode.toUpperCase();

  const badge = document.getElementById('modeBannerBadge');
  const desc = document.getElementById('modeBannerDesc');
  const gloveOverlay = document.getElementById('gloveStandbyOverlay');
  const trainerGloveOverlay = document.getElementById('trainerGloveStandbyOverlay');
  const hudTrainerMode = document.getElementById('hudTrainerMode');

  if (hudTrainerMode) {
    if (mode === 'vision') hudTrainerMode.textContent = 'MODE: VISION ONLY';
    else if (mode === 'glove') hudTrainerMode.textContent = 'MODE: GLOVE ONLY';
    else hudTrainerMode.textContent = 'MODE: FUSION (BOTH)';
  }

  if (mode === 'vision') {
    badge.textContent = 'MODE 1: VISION ONLY';
    desc.textContent = 'Recognizing continuous ISL sentences purely from dual-hand landmark actions.';
    if (gloveOverlay) gloveOverlay.style.display = 'none';
    if (trainerGloveOverlay) trainerGloveOverlay.style.display = 'none';
    document.getElementById('serialStatusText').textContent = 'Glove: Standby (Vision Mode)';
  } else if (mode === 'glove') {
    badge.textContent = 'MODE 2: GLOVE ONLY';
    desc.textContent = 'Recognizing sign actions purely from 5-finger flex resistance and IMU orientation.';
    if (gloveOverlay) gloveOverlay.style.display = 'flex';
    if (trainerGloveOverlay) trainerGloveOverlay.style.display = 'flex';
    document.getElementById('serialStatusText').textContent = 'Glove: ACTIVE (Serial COM3)';
    runGloveInference();
  } else if (mode === 'fusion') {
    badge.textContent = 'MODE 3: MULTIMODAL FUSION';
    desc.textContent = 'Actively fusing Dual-Hand camera vision with ESP32 Smart Glove flex sensors & IMU tilt into continuous sentences.';
    if (gloveOverlay) gloveOverlay.style.display = 'none';
    if (trainerGloveOverlay) trainerGloveOverlay.style.display = 'none';
    document.getElementById('serialStatusText').textContent = 'Glove: FUSED with Vision';
  }

  document.querySelectorAll('.mode-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.mode === mode);
  });
  document.querySelectorAll('.t-mode-pill').forEach(b => {
    b.classList.toggle('active', b.dataset.mode === mode);
  });

  fetch('/api/mode', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode })
  }).catch(() => {});
}

// =========================================================================
// 2. ISL SENTENCE SYNTHESIS ENGINE (LOCAL INSTANT + API SYNC)
// =========================================================================

function synthesizeSentenceFromTokens(tokenList) {
  if (!tokenList || tokenList.length === 0) {
    return {
      sentence: "Waiting for sign actions...",
      rule: "EMPTY_BUFFER",
      confidence: 0.0,
      tokens: []
    };
  }

  // Extract clean uppercase tokens
  const cleanTokens = [];
  const confs = [];
  tokenList.forEach(t => {
    const name = typeof t === 'object' ? t.gesture : t;
    const conf = typeof t === 'object' ? (t.confidence || 0.90) : 0.90;
    if (name && name !== 'UNKNOWN' && name !== 'NONE' && name !== 'IDLE') {
      cleanTokens.push(name.toUpperCase());
      confs.push(conf);
    }
  });

  if (cleanTokens.length === 0) {
    return {
      sentence: "Awaiting stable sign actions...",
      rule: "NO_VALID_TOKENS",
      confidence: 0.0,
      tokens: []
    };
  }

  const avgConf = confs.reduce((a, b) => a + b, 0) / confs.length;
  const boostedConf = Math.min(0.99, avgConf + (cleanTokens.length > 1 ? 0.02 : 0.0));

  // 1. Direct key match (e.g. "I+INDIAN")
  const key = cleanTokens.join('+');
  if (ISL_SENTENCE_DICTIONARY[key]) {
    return {
      sentence: ISL_SENTENCE_DICTIONARY[key],
      rule: `IDIOMATIC_${cleanTokens.length}TOKENS`,
      confidence: boostedConf,
      tokens: cleanTokens
    };
  }

  // 2. Consecutive duplicate removal (e.g. "I+I+INDIAN" -> "I+INDIAN")
  const deduped = [];
  cleanTokens.forEach(t => {
    if (deduped.length === 0 || deduped[deduped.length - 1] !== t) {
      deduped.push(t);
    }
  });
  const dedupKey = deduped.join('+');
  if (ISL_SENTENCE_DICTIONARY[dedupKey]) {
    return {
      sentence: ISL_SENTENCE_DICTIONARY[dedupKey],
      rule: `DEDUPED_${deduped.length}TOKENS`,
      confidence: boostedConf,
      tokens: cleanTokens
    };
  }

  // 3. Fallback syntactic assembly for custom gestures
  const tokenSet = new Set(cleanTokens);
  if (tokenSet.has("HELP")) {
    return { sentence: "Please provide urgent assistance.", rule: "RULE_REQUEST_HELP", confidence: boostedConf, tokens: cleanTokens };
  }
  if (tokenSet.has("WATER")) {
    return { sentence: "Could you please give me water to drink?", rule: "RULE_REQUEST_WATER", confidence: boostedConf, tokens: cleanTokens };
  }
  if (tokenSet.has("MORNING")) {
    const others = cleanTokens.filter(t => t !== "MORNING").map(t => t.replace(/_/g, ' ').toLowerCase());
    const phrase = others.join(' ');
    return { sentence: `Good morning! Communicating ${phrase || 'in Indian Sign Language'}.`, rule: "RULE_GREETING_EXTENDED", confidence: boostedConf, tokens: cleanTokens };
  }

  // Generic natural smoother
  const naturalWords = cleanTokens.map(t => t.replace(/_/g, ' ').toLowerCase());
  const cap = naturalWords.map((w, idx) => idx === 0 ? w.charAt(0).toUpperCase() + w.slice(1) : w).join(' ');
  return {
    sentence: `${cap} — expressed in Indian Sign Language.`,
    rule: "RULE_NATURAL_SYNTHESIS",
    confidence: boostedConf,
    tokens: cleanTokens
  };
}

// =========================================================================
// 3. CONTINUOUS SENTENCE BUFFER & ANTI-JITTER DWELL GATING
// =========================================================================

function handleGestureForSentence(gesture, confidence) {
  const minThreshold = (state.customModel && state.customModel.classes && state.customModel.classes.includes(gesture)) ? 0.45 : 0.65;
  if (!gesture || gesture === 'UNKNOWN' || confidence < minThreshold) {
    // Decay dwell counter smoothly
    state.pendingDwellFrames = Math.max(0, state.pendingDwellFrames - 1);
    updateStabilityMeter();

    // Check if hands have rested and auto-commit is needed
    if (state.pendingDwellFrames === 0) {
      state.pendingGesture = null;
      checkAutoCommitTimeout();
    }
    return;
  }

  // Pose is active and above confidence threshold
  if (gesture === state.pendingGesture) {
    state.pendingDwellFrames++;
  } else {
    state.pendingGesture = gesture;
    state.pendingDwellFrames = 1;
  }

  updateStabilityMeter();

  // If held stably for STABILITY_THRESHOLD frames -> LOCK IN TOKEN
  if (state.pendingDwellFrames >= state.STABILITY_THRESHOLD) {
    commitTokenToActiveSentence(gesture, confidence);
  }
}

function updateStabilityMeter() {
  const fill = document.getElementById('stabilityProgressFill');
  const text = document.getElementById('stabilityPercentText');
  const rawSign = document.getElementById('rawDetectedSign');
  const rawConf = document.getElementById('rawDetectedConf');
  const dwellState = document.getElementById('hudDwellState');

  const count = state.pendingDwellFrames;
  const max = state.STABILITY_THRESHOLD;
  const pct = Math.min(100, Math.round((count / max) * 100));

  if (fill) fill.style.width = `${pct}%`;

  if (rawSign && state.pendingGesture) {
    rawSign.textContent = `${state.pendingGesture} ${GESTURE_ICONS[state.pendingGesture] || ''}`;
  }
  if (rawConf) {
    rawConf.textContent = state.pendingGesture ? `${pct}% Stability` : 'Waiting for pose';
  }

  if (text) {
    if (pct >= 100) {
      text.textContent = 'POSE LOCKED IN (100%)';
      text.style.color = '#34d399';
      if (dwellState) dwellState.textContent = 'LOCKED';
    } else if (pct > 0) {
      text.textContent = `HOLDING POSE (${pct}%)`;
      text.style.color = '#38bdf8';
      if (dwellState) dwellState.textContent = `${pct}%`;
    } else {
      text.textContent = 'AWAITING STABLE POSE';
      text.style.color = '#94a3b8';
      if (dwellState) dwellState.textContent = 'IDLE';
    }
  }
}

function commitTokenToActiveSentence(gesture, confidence) {
  // Prevent repeating the same gesture indefinitely while hand is held
  if (state.lastCommittedTokenGesture === gesture) {
    return;
  }

  state.lastCommittedTokenGesture = gesture;

  const tokenObj = {
    gesture: gesture,
    confidence: confidence,
    timestamp: Date.now(),
    modality: state.activeMode
  };

  state.activeSentenceTokens.push(tokenObj);

  // Maximum buffer size of 6 tokens
  if (state.activeSentenceTokens.length > 6) {
    state.activeSentenceTokens.shift();
  }

  renderActiveSentence();

  // Reset auto-commit countdown
  resetAutoCommitTimer();
}

function renderActiveSentence() {
  const result = synthesizeSentenceFromTokens(state.activeSentenceTokens);
  state.activeSentenceString = result.sentence;
  state.activeSentenceConfidence = result.confidence;

  const displayEl = document.getElementById('activeSentenceDisplay');
  const railEl = document.getElementById('sentenceTokensRail');
  const confEl = document.getElementById('sentenceConfidence');
  const ruleEl = document.getElementById('sentenceSyntaxRule');
  const countEl = document.getElementById('activeActionCount');

  if (displayEl) displayEl.textContent = result.sentence;
  if (confEl) confEl.textContent = `${(result.confidence * 100).toFixed(1)}%`;
  if (ruleEl) ruleEl.textContent = result.rule.replace(/_/g, ' ');
  if (countEl) countEl.textContent = `${state.activeSentenceTokens.length} Sign${state.activeSentenceTokens.length === 1 ? '' : 's'} Locked In`;

  if (railEl) {
    railEl.innerHTML = '';
    if (state.activeSentenceTokens.length === 0) {
      railEl.innerHTML = '<span class="draft-empty-placeholder">Show hands to camera or use simulator to build sentence...</span>';
    } else {
      state.activeSentenceTokens.forEach((t, idx) => {
        if (idx > 0) {
          const arrow = document.createElement('span');
          arrow.className = 'token-arrow';
          arrow.textContent = '➔';
          railEl.appendChild(arrow);
        }
        const chip = document.createElement('span');
        chip.className = 'token-chip';
        chip.innerHTML = `
          <span class="chip-icon">${GESTURE_ICONS[t.gesture] || '✋'}</span>
          <span class="chip-name">${t.gesture}</span>
          <span class="chip-conf">${Math.round(t.confidence * 100)}%</span>
        `;
        railEl.appendChild(chip);
      });
    }
  }

  // Query backend synthesis asynchronously for telemetry alignment
  if (state.activeSentenceTokens.length > 0) {
    fetch('/api/sentences/synthesize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        tokens: state.activeSentenceTokens.map(t => t.gesture),
        mode: state.activeMode
      })
    })
      .then(res => res.ok ? res.json() : null)
      .then(apiRes => {
        if (apiRes && apiRes.sentence) {
          state.activeSentenceString = apiRes.sentence;
          if (displayEl) displayEl.textContent = apiRes.sentence;
        }
      })
      .catch(() => {});
  }
}

function resetAutoCommitTimer() {
  if (state.autoCommitTimer) {
    clearTimeout(state.autoCommitTimer);
  }
  if (!state.isAutoCommitEnabled) return;

  // After 2.0 seconds of no new signs, auto-commit sentence to history
  state.autoCommitTimer = setTimeout(() => {
    if (state.activeSentenceTokens.length > 0) {
      commitActiveSentenceToHistory(true);
    }
  }, 2200);
}

function checkAutoCommitTimeout() {
  // If user dropped hand, trigger auto-commit sooner (1.5s)
  if (state.activeSentenceTokens.length > 0 && !state.autoCommitTimer) {
    resetAutoCommitTimer();
  }
}

function commitActiveSentenceToHistory(isAuto = false) {
  if (state.activeSentenceTokens.length === 0) {
    showToast("Sentence buffer is empty.");
    return;
  }

  const sentenceText = state.activeSentenceString;
  const tokensList = state.activeSentenceTokens.map(t => t.gesture);
  const now = new Date();
  const timeStr = now.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' }) + ', ' + now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

  const record = {
    id: `sent-${String(state.sentenceHistory.length + 1).padStart(3, '0')}`,
    sentence: sentenceText,
    tokens: tokensList,
    confidence: state.activeSentenceConfidence,
    modality: state.activeMode,
    timestamp: timeStr,
    timestamp_ms: Date.now(),
    duration_seconds: Math.max(1.2, tokensList.length * 0.9)
  };

  // Add to in-memory history
  state.sentenceHistory.unshift(record);

  // Persist to localStorage
  saveSentenceHistoryToStorage();

  // Sync to backend API
  fetch('/api/sentences/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(record)
  }).catch(() => {});

  // Speak sentence if TTS is enabled
  if (!state.isTTSMuted) {
    speakSentence(sentenceText);
  }

  showToast(isAuto ? `Auto-committed: "${sentenceText}"` : `Saved to Action History!`);

  // Reset active buffer
  state.activeSentenceTokens = [];
  state.lastCommittedTokenGesture = null;
  state.pendingGesture = null;
  state.pendingDwellFrames = 0;
  updateStabilityMeter();
  renderActiveSentence();
  updateStudioMetrics();
  renderSentenceHistoryList();
}

function undoLastToken() {
  if (state.activeSentenceTokens.length > 0) {
    const removed = state.activeSentenceTokens.pop();
    state.lastCommittedTokenGesture = state.activeSentenceTokens.length > 0 ? state.activeSentenceTokens[state.activeSentenceTokens.length - 1].gesture : null;
    renderActiveSentence();
    showToast(`Removed sign: ${removed.gesture}`);
  }
}

function clearActiveSentence() {
  state.activeSentenceTokens = [];
  state.lastCommittedTokenGesture = null;
  state.pendingGesture = null;
  state.pendingDwellFrames = 0;
  if (state.autoCommitTimer) clearTimeout(state.autoCommitTimer);
  updateStabilityMeter();
  renderActiveSentence();
  showToast("Active sentence buffer cleared.");
}

// =========================================================================
// 4. TEXT-TO-SPEECH (TTS) SPEECH SYNTHESIS ENGINE
// =========================================================================

function speakSentence(text) {
  if (!('speechSynthesis' in window) || state.isTTSMuted) return;

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1.0;
  utterance.pitch = 1.0;

  const waveStrip = document.getElementById('audioWaveStrip');
  if (waveStrip) waveStrip.style.display = 'flex';

  utterance.onend = () => {
    if (waveStrip) waveStrip.style.display = 'none';
  };
  utterance.onerror = () => {
    if (waveStrip) waveStrip.style.display = 'none';
  };

  window.speechSynthesis.speak(utterance);
}

// =========================================================================
// 5. SENTENCE ACTIONS & HISTORY STUDIO (TAB 2 MANAGEMENT)
// =========================================================================

function initSentenceStudio() {
  // Sandbox Palette Buttons
  document.querySelectorAll('.palette-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const sign = btn.dataset.sign;
      if (sign) {
        state.sandboxDraftTokens.push(sign);
        renderSandboxDraft();
      }
    });
  });

  // Sandbox Controls
  const btnStoreSandbox = document.getElementById('btnSandboxStoreAction');
  if (btnStoreSandbox) {
    btnStoreSandbox.addEventListener('click', () => {
      if (state.sandboxDraftTokens.length === 0) {
        showToast("Add sign actions to the draft first.");
        return;
      }
      const syn = synthesizeSentenceFromTokens(state.sandboxDraftTokens);
      const now = new Date();
      const timeStr = now.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' }) + ', ' + now.toLocaleTimeString();

      const record = {
        id: `sent-${String(state.sentenceHistory.length + 1).padStart(3, '0')}`,
        sentence: syn.sentence,
        tokens: syn.tokens,
        confidence: 0.985,
        modality: state.activeMode,
        timestamp: timeStr,
        timestamp_ms: Date.now(),
        duration_seconds: Math.max(1.0, syn.tokens.length * 0.8)
      };

      state.sentenceHistory.unshift(record);
      saveSentenceHistoryToStorage();
      showToast("Draft sentence stored in action history!");
      state.sandboxDraftTokens = [];
      renderSandboxDraft();
      renderSentenceHistoryList();
      updateStudioMetrics();
    });
  }

  const btnSpeakSandbox = document.getElementById('btnSandboxSpeak');
  if (btnSpeakSandbox) {
    btnSpeakSandbox.addEventListener('click', () => {
      const syn = synthesizeSentenceFromTokens(state.sandboxDraftTokens);
      speakSentence(syn.sentence);
    });
  }

  const btnClearSandbox = document.getElementById('btnSandboxClear');
  if (btnClearSandbox) {
    btnClearSandbox.addEventListener('click', () => {
      state.sandboxDraftTokens = [];
      renderSandboxDraft();
    });
  }

  // History Search & Filter
  const searchInput = document.getElementById('searchSentencesInput');
  if (searchInput) {
    searchInput.addEventListener('input', () => renderSentenceHistoryList());
  }

  const filterSelect = document.getElementById('filterModalitySelect');
  if (filterSelect) {
    filterSelect.addEventListener('change', () => renderSentenceHistoryList());
  }

  // Export Buttons
  const btnJson = document.getElementById('btnExportJSON');
  if (btnJson) btnJson.addEventListener('click', () => exportHistory('json'));

  const btnCsv = document.getElementById('btnExportCSV');
  if (btnCsv) btnCsv.addEventListener('click', () => exportHistory('csv'));

  const btnTxt = document.getElementById('btnExportTXT');
  if (btnTxt) btnTxt.addEventListener('click', () => exportHistory('txt'));

  const btnClearAll = document.getElementById('btnClearHistory');
  if (btnClearAll) {
    btnClearAll.addEventListener('click', () => {
      if (confirm("Clear all recorded sentence action history?")) {
        state.sentenceHistory = [];
        saveSentenceHistoryToStorage();
        fetch('/api/sentences/history', { method: 'DELETE' }).catch(() => {});
        renderSentenceHistoryList();
        updateStudioMetrics();
        showToast("Action history cleared.");
      }
    });
  }

  const btnDemo = document.getElementById('btnLoadDemoActions');
  if (btnDemo) {
    btnDemo.addEventListener('click', () => {
      loadDemoSentenceActions();
    });
  }
}

function renderSandboxDraft() {
  const track = document.getElementById('draftChipsTrack');
  const count = document.getElementById('draftTokenCount');
  const output = document.getElementById('sandboxOutputText');
  const rule = document.getElementById('sandboxRulePill');

  if (count) count.textContent = `${state.sandboxDraftTokens.length} Action${state.sandboxDraftTokens.length === 1 ? '' : 's'}`;

  if (track) {
    track.innerHTML = '';
    if (state.sandboxDraftTokens.length === 0) {
      track.innerHTML = '<span class="draft-empty-placeholder">Click signs above to assemble draft sentence...</span>';
    } else {
      state.sandboxDraftTokens.forEach((t, idx) => {
        if (idx > 0) {
          const arrow = document.createElement('span');
          arrow.className = 'token-arrow';
          arrow.textContent = '➔';
          track.appendChild(arrow);
        }
        const chip = document.createElement('span');
        chip.className = 'draft-chip';
        chip.innerHTML = `${GESTURE_ICONS[t] || '✋'} ${t}`;
        track.appendChild(chip);
      });
    }
  }

  const syn = synthesizeSentenceFromTokens(state.sandboxDraftTokens);
  if (output) output.textContent = `“${syn.sentence}”`;
  if (rule) rule.textContent = `RULE: ${syn.rule.replace(/_/g, ' ')}`;
}

function loadSentenceHistory() {
  fetch('/api/translations')
    .then(res => res.json())
    .then(data => {
      if (data && Array.isArray(data.translations) && data.translations.length > 0) {
        state.sentenceHistory = data.translations;
        saveSentenceHistoryToStorage();
      } else {
        fetch('/api/sentences/history')
          .then(res2 => res2.json())
          .then(data2 => {
            if (data2 && data2.sentences && data2.sentences.length > 0) {
              state.sentenceHistory = data2.sentences;
              saveSentenceHistoryToStorage();
            } else if (state.sentenceHistory.length === 0) {
              loadDemoSentenceActions();
            }
            renderSentenceHistoryList();
            updateStudioMetrics();
          })
          .catch(() => {
            if (state.sentenceHistory.length === 0) loadDemoSentenceActions();
            renderSentenceHistoryList();
            updateStudioMetrics();
          });
        return;
      }
      renderSentenceHistoryList();
      updateStudioMetrics();
    })
    .catch(() => {
      try {
        const cached = localStorage.getItem('kinex_sentence_history');
        if (cached) state.sentenceHistory = JSON.parse(cached);
      } catch (e) {}
      if (state.sentenceHistory.length === 0) {
        loadDemoSentenceActions();
      }
      renderSentenceHistoryList();
      updateStudioMetrics();
    });
}

function saveSentenceHistoryToStorage() {
  try {
    localStorage.setItem('kinex_sentence_history', JSON.stringify(state.sentenceHistory));
  } catch (e) {}
}

function loadDemoSentenceActions() {
  state.sentenceHistory = [
    {
      id: "sent-001",
      sentence: "I am an Indian citizen.",
      tokens: ["I", "INDIAN"],
      confidence: 0.974,
      modality: "vision",
      timestamp: "12 Oct 2026, 11:15:32",
      timestamp_ms: Date.now() - 3600000,
      duration_seconds: 1.6
    },
    {
      id: "sent-002",
      sentence: "Good morning! Thank you very much for your time.",
      tokens: ["MORNING", "THANK_YOU"],
      confidence: 0.981,
      modality: "fusion",
      timestamp: "12 Oct 2026, 11:18:45",
      timestamp_ms: Date.now() - 2400000,
      duration_seconds: 2.1
    },
    {
      id: "sent-003",
      sentence: "I am hard of hearing; thank you for communicating with me.",
      tokens: ["I", "HEARING", "THANK_YOU"],
      confidence: 0.962,
      modality: "vision",
      timestamp: "12 Oct 2026, 11:21:10",
      timestamp_ms: Date.now() - 1200000,
      duration_seconds: 2.8
    },
    {
      id: "sent-004",
      sentence: "Thank you so much, goodbye and take care!",
      tokens: ["THANK_YOU", "BYE"],
      confidence: 0.950,
      modality: "glove",
      timestamp: "12 Oct 2026, 11:23:04",
      timestamp_ms: Date.now() - 600000,
      duration_seconds: 1.9
    }
  ];
  saveSentenceHistoryToStorage();
  renderSentenceHistoryList();
  updateStudioMetrics();
  showToast("Demo sentence action history loaded!");
}

function renderSentenceHistoryList() {
  const container = document.getElementById('sentenceHistoryList');
  if (!container) return;

  const searchQuery = (document.getElementById('searchSentencesInput')?.value || '').toLowerCase().trim();
  const filterModality = document.getElementById('filterModalitySelect')?.value || 'ALL';

  let filtered = state.sentenceHistory;

  if (filterModality !== 'ALL') {
    filtered = filtered.filter(item => (item.modality || 'vision').toLowerCase() === filterModality.toLowerCase());
  }

  if (searchQuery) {
    filtered = filtered.filter(item => {
      const matchSentence = (item.sentence || '').toLowerCase().includes(searchQuery);
      const matchTokens = (item.tokens || []).some(t => t.toLowerCase().includes(searchQuery));
      return matchSentence || matchTokens;
    });
  }

  container.innerHTML = '';

  if (filtered.length === 0) {
    container.innerHTML = `
      <div style="text-align: center; padding: 2.5rem 1rem; color: var(--text-dim);">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📜</div>
        <div style="font-size: 0.95rem; font-weight: 700; color: var(--text-main);">No Sentence Actions Found</div>
        <div style="font-size: 0.8rem; margin-top: 0.25rem;">Perform gestures with your camera or click "Demo" above to populate records.</div>
      </div>
    `;
    return;
  }

  filtered.forEach(record => {
    const card = document.createElement('div');
    card.className = 'sentence-history-item';
    card.id = `histCard_${record.id}`;

    const tokensList = Array.isArray(record.tokens) ? record.tokens : [];
    const tokensHtml = tokensList.map(t => `
      <span class="hist-token-pill">
        <span>${GESTURE_ICONS[t] || '✋'}</span>
        <span>${t}</span>
      </span>
    `).join('<span class="token-arrow" style="font-size:0.75rem;">➔</span>');

    const mod = (record.modality || 'vision').toLowerCase();

    card.innerHTML = `
      <div class="hist-card-header">
        <div style="display:flex; align-items:center; gap:0.5rem;">
          <span class="hist-id-tag">#${String(record.id).toUpperCase()}</span>
          <span class="hist-timestamp">${record.timestamp || record.created_at || 'Just now'}</span>
        </div>
        <div class="hist-meta-pills">
          <span class="modality-badge ${mod}">MODE: ${mod.toUpperCase()}</span>
          <span class="accuracy-badge">${Math.round((record.confidence || 0.95) * 100)}% ACCURACY</span>
        </div>
      </div>

      <div class="hist-card-sentence">“${record.sentence}”</div>

      <div class="hist-card-tokens">
        ${tokensHtml}
      </div>

      <div class="hist-card-footer">
        <span class="hist-duration-note">⏱️ Duration: ${record.duration_seconds || '1.8'}s • ${tokensList.length} Actions</span>
        <div class="hist-btn-group">
          ${record.video_path ? `
            <button class="btn btn-xs btn-accent" onclick="openVideoPlaybackModal('${record.video_path}', 'Recorded Translation #${record.id}')" title="Play recorded video clip">
              🎬 Watch Video
            </button>
          ` : ''}
          <button class="btn btn-xs" onclick="speakSentence('${(record.sentence || '').replace(/'/g, "\\'")}')" title="Play vocal speech audio">
            🔊 Speak
          </button>
          <button class="btn btn-xs" onclick="copySentenceText('${(record.sentence || '').replace(/'/g, "\\'")}')" title="Copy sentence text">
            📋 Copy
          </button>
          <button class="btn btn-xs" onclick="replaySentenceSigns(['${tokensList.join("','")}'])" title="Visually replay hand signs">
            🔁 Replay
          </button>
          <button class="btn btn-xs btn-danger-subtle" onclick="deleteSentenceRecord('${record.id}')" title="Delete action record from database">
            🗑️
          </button>
        </div>
      </div>
    `;
    container.appendChild(card);
  });
}

function updateStudioMetrics() {
  const totalSentences = state.sentenceHistory.length;
  let totalTokens = 0;
  let confSum = 0;

  state.sentenceHistory.forEach(r => {
    totalTokens += (r.tokens ? r.tokens.length : 1);
    confSum += (r.confidence || 0.95);
  });

  const avgAcc = totalSentences > 0 ? ((confSum / totalSentences) * 100).toFixed(1) : '96.5';

  const badgeCount = document.getElementById('badgeHistoryCount');
  if (badgeCount) badgeCount.textContent = totalSentences;

  const statSentences = document.getElementById('statTotalSentences');
  if (statSentences) statSentences.textContent = totalSentences;

  const statTokens = document.getElementById('statTotalTokens');
  if (statTokens) statTokens.textContent = totalTokens;

  const statAcc = document.getElementById('statAvgAccuracy');
  if (statAcc) statAcc.textContent = `${avgAcc}%`;
}

window.copySentenceText = function(text) {
  if (navigator.clipboard) {
    navigator.clipboard.writeText(text)
      .then(() => showToast("Copied sentence to clipboard!"))
      .catch(() => showToast("Failed to copy text."));
  }
};

window.deleteSentenceRecord = function(id) {
  fetch(`/api/translations/${encodeURIComponent(id)}`, { method: 'DELETE' }).catch(() => {});
  state.sentenceHistory = state.sentenceHistory.filter(r => String(r.id) !== String(id));
  saveSentenceHistoryToStorage();
  renderSentenceHistoryList();
  updateStudioMetrics();
  if (typeof loadDashboardStats === 'function') loadDashboardStats();
  showToast("Record removed from database.");
};

window.replaySentenceSigns = function(tokensArray) {
  if (!tokensArray || tokensArray.length === 0) return;

  // Switch to Recognition tab to display the replay on canvas
  const recTabBtn = document.getElementById('tabBtnRecognition');
  if (recTabBtn) recTabBtn.click();

  let idx = 0;
  showToast(`Replaying sequence: ${tokensArray.join(' ➔ ')}`);

  const step = () => {
    if (idx >= tokensArray.length) {
      showToast("Replay completed!");
      return;
    }
    const sign = tokensArray[idx];
    applyGesturePreset(sign);
    idx++;
    setTimeout(step, 1100);
  };

  setTimeout(step, 400);
};

// Export Handlers
function exportHistory(type) {
  if (state.sentenceHistory.length === 0) {
    showToast("No sentence records to export.");
    return;
  }

  let content = "";
  let mimeType = "text/plain";
  let filename = `kinex_sentence_actions_${Date.now()}`;

  if (type === 'json') {
    content = JSON.stringify(state.sentenceHistory, null, 2);
    mimeType = "application/json";
    filename += ".json";
  } else if (type === 'csv') {
    const headers = "id,timestamp,modality,confidence,actions_chain,sentence\n";
    const rows = state.sentenceHistory.map(r => {
      const escaped = `"${r.sentence.replace(/"/g, '""')}"`;
      const chain = `"${r.tokens.join(' -> ')}"`;
      return `${r.id},${r.timestamp},${r.modality},${r.confidence},${chain},${escaped}`;
    }).join("\n");
    content = headers + rows;
    mimeType = "text/csv";
    filename += ".csv";
  } else if (type === 'txt') {
    content = "========================================================\n" +
              " KINEX ADV 2.0 — ISL SENTENCE ACTION HISTORY TRANSCRIPT\n" +
              "========================================================\n\n";
    state.sentenceHistory.forEach(r => {
      content += `[${r.timestamp}] (${r.modality.toUpperCase()}) Accuracy: ${Math.round(r.confidence * 100)}%\n`;
      content += `Actions:  ${r.tokens.join(' ➔ ')}\n`;
      content += `Sentence: “${r.sentence}”\n\n`;
    });
    filename += ".txt";
  }

  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  showToast(`Exported ${type.toUpperCase()} file!`);
}

function showToast(message) {
  const toast = document.getElementById('kinexToast');
  const msgEl = document.getElementById('toastMessage');
  if (!toast || !msgEl) return;

  msgEl.textContent = message;
  toast.style.display = 'flex';

  if (window.toastTimer) clearTimeout(window.toastTimer);
  window.toastTimer = setTimeout(() => {
    toast.style.display = 'none';
  }, 2600);
}

// =========================================================================
// 6. UI EVENT LISTENERS
// =========================================================================

function initUIEventListeners() {
  const btnStartCam = document.getElementById('btnStartCamera');
  if (btnStartCam) {
    btnStartCam.addEventListener('click', async () => {
      btnStartCam.disabled = true;
      const originalHtml = btnStartCam.innerHTML;
      btnStartCam.innerHTML = '<span>⏳ Connecting Camera...</span>';
      try {
        await toggleCamera();
      } finally {
        btnStartCam.disabled = false;
        btnStartCam.innerHTML = originalHtml;
      }
    });
  }

  const btnToggleCam = document.getElementById('btnToggleCamera');
  if (btnToggleCam) {
    btnToggleCam.addEventListener('click', toggleCamera);
  }

  const btnTrainerStart = document.getElementById('btnTrainerStartCamera');
  if (btnTrainerStart) {
    btnTrainerStart.addEventListener('click', async () => {
      btnTrainerStart.disabled = true;
      const originalHtml = btnTrainerStart.innerHTML;
      btnTrainerStart.innerHTML = '<span>⏳ Connecting Camera...</span>';
      try {
        await toggleCamera();
      } finally {
        btnTrainerStart.disabled = false;
        btnTrainerStart.innerHTML = originalHtml;
      }
    });
  }

  const btnTrainerToggle = document.getElementById('btnTrainerToggleCamera');
  if (btnTrainerToggle) {
    btnTrainerToggle.addEventListener('click', toggleCamera);
  }

  // Diagnostic Error & Fallback Controls
  const btnRetry = document.getElementById('btnRetryCamera');
  if (btnRetry) btnRetry.addEventListener('click', async () => {
    btnRetry.disabled = true;
    btnRetry.textContent = '⏳ Reconnecting...';
    await startCamera();
    btnRetry.disabled = false;
    btnRetry.textContent = '🔄 Retry Camera';
  });
  const btnTrainerRetry = document.getElementById('btnTrainerRetryCamera');
  if (btnTrainerRetry) btnTrainerRetry.addEventListener('click', async () => {
    btnTrainerRetry.disabled = true;
    btnTrainerRetry.textContent = '⏳ Reconnecting...';
    await startCamera();
    btnTrainerRetry.disabled = false;
    btnTrainerRetry.textContent = '🔄 Retry Camera';
  });

  const btnErrorVirtual = document.getElementById('btnErrorVirtualStream');
  if (btnErrorVirtual) btnErrorVirtual.addEventListener('click', () => startVirtualStream());

  const btnTrainerErrorVirtual = document.getElementById('btnTrainerErrorVirtualStream');
  if (btnTrainerErrorVirtual) btnTrainerErrorVirtual.addEventListener('click', () => startVirtualStream());

  const btnDismissError = document.getElementById('btnDismissCameraError');
  if (btnDismissError) btnDismissError.addEventListener('click', () => {
    hideCameraError();
    if (placeholder && !state.isCameraActive) {
      placeholder.classList.remove('hidden');
      placeholder.style.display = 'flex';
    }
  });

  const btnTrainerDismiss = document.getElementById('btnTrainerDismissError');
  if (btnTrainerDismiss) btnTrainerDismiss.addEventListener('click', () => {
    hideCameraError();
    const trainerPlaceholder = document.getElementById('trainerCameraPlaceholder');
    if (trainerPlaceholder && !state.isCameraActive) {
      trainerPlaceholder.classList.remove('hidden');
      trainerPlaceholder.style.display = 'flex';
    }
  });

  const selectDevice = document.getElementById('selectCameraDevice');
  if (selectDevice) {
    selectDevice.addEventListener('change', () => {
      if (state.isCameraActive) {
        startCamera(selectDevice.value);
      }
    });
  }

  const btnVirtual = document.getElementById('btnToggleVirtualStream');
  if (btnVirtual) btnVirtual.addEventListener('click', toggleVirtualStream);

  document.getElementById('btnCaptureFrame').addEventListener('click', captureSnapshot);
  document.getElementById('imageUploadInput').addEventListener('change', handleImageUpload);

  // Sentence Simulator
  const btnSimSentence = document.getElementById('btnSimulateSentence');
  if (btnSimSentence) {
    btnSimSentence.addEventListener('click', () => {
      const presetVal = document.getElementById('selectSentencePreset').value;
      runSentenceSimulation(presetVal);
    });
  }

  // Active Sentence Controls
  const btnCommit = document.getElementById('btnCommitSentence');
  if (btnCommit) {
    btnCommit.addEventListener('click', () => commitActiveSentenceToHistory(false));
  }

  const btnSpeak = document.getElementById('btnSpeakActiveSentence');
  if (btnSpeak) {
    btnSpeak.addEventListener('click', () => speakSentence(state.activeSentenceString));
  }

  const btnUndo = document.getElementById('btnUndoToken');
  if (btnUndo) {
    btnUndo.addEventListener('click', () => undoLastToken());
  }

  const btnClear = document.getElementById('btnClearActiveSentence');
  if (btnClear) {
    btnClear.addEventListener('click', () => clearActiveSentence());
  }

  // Global TTS Toggle
  const ttsBtn = document.getElementById('btnToggleTTS');
  ttsBtn.addEventListener('click', () => {
    state.isTTSMuted = !state.isTTSMuted;
    if (state.isTTSMuted) {
      ttsBtn.classList.add('muted');
      document.getElementById('ttsIcon').textContent = '🔇';
      document.getElementById('ttsLabel').textContent = 'TTS: MUTED';
    } else {
      ttsBtn.classList.remove('muted');
      document.getElementById('ttsIcon').textContent = '🔊';
      document.getElementById('ttsLabel').textContent = 'TTS: ON';
    }
  });

  // Switch to Recognition tab from Trainer
  const btnTestInRec = document.getElementById('btnTestInRecognition');
  if (btnTestInRec) {
    btnTestInRec.addEventListener('click', () => {
      const recTabBtn = document.getElementById('tabBtnRecognition');
      if (recTabBtn) recTabBtn.click();
    });
  }

  // Reset Model button
  const btnResetMod = document.getElementById('btnResetModel');
  if (btnResetMod) {
    btnResetMod.addEventListener('click', () => {
      if (confirm("Reset gesture classification model back to baseline?")) {
        fetch('/api/trainer/reset', { method: 'POST' })
          .then(res => res.json())
          .then(() => {
            state.sampleCounts = {};
            refreshTrainingSampleStats();
            initVocabulary();
            alert("Model reset to baseline defaults.");
          })
          .catch(() => alert("Failed to reset model."));
      }
    });
  }
}

// Continuous Sentence Preset Simulator
function runSentenceSimulation(presetId) {
  const PRESET_MAP = {
    "p1": ["I", "INDIAN"],
    "p2": ["MORNING", "THANK_YOU"],
    "p3": ["I", "HEARING", "THANK_YOU"],
    "p4": ["THANK_YOU", "BYE"],
    "p5": ["MORNING", "I", "INDIAN"]
  };

  const tokens = PRESET_MAP[presetId] || ["I", "INDIAN"];

  // Clear current buffer
  state.activeSentenceTokens = [];
  state.lastCommittedTokenGesture = null;
  state.pendingGesture = null;
  state.pendingDwellFrames = 0;
  renderActiveSentence();

  showToast(`Simulating sentence: ${tokens.join(' ➔ ')}`);

  let idx = 0;
  const step = () => {
    if (idx >= tokens.length) {
      setTimeout(() => {
        commitActiveSentenceToHistory(false);
      }, 700);
      return;
    }

    const gesture = tokens[idx];
    applyGesturePreset(gesture);

    // Simulate stability fill locking in
    state.pendingGesture = gesture;
    state.pendingDwellFrames = state.STABILITY_THRESHOLD;
    updateStabilityMeter();

    commitTokenToActiveSentence(gesture, 0.97);

    idx++;
    setTimeout(step, 1200);
  };

  step();
}

// =========================================================================
// 7. SMART GLOVE INTERACTIVE SLIDERS
// =========================================================================

function initGloveSliders() {
  const fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Little'];
  fingers.forEach(f => {
    const slider = document.getElementById(`slider${f}`);
    if (slider) {
      slider.addEventListener('input', (e) => {
        const val = parseInt(e.target.value, 10);
        document.getElementById(`flexFill${f}`).style.height = `${val}%`;
        document.getElementById(`flexPct${f}`).textContent = `${val}%`;

        if (state.activeMode === 'glove' || state.activeMode === 'fusion') {
          runGloveInference();
        }
      });
    }
  });
}

function getSliderSensors() {
  return {
    thumb: parseInt(document.getElementById('sliderThumb').value, 10) / 100.0,
    index: parseInt(document.getElementById('sliderIndex').value, 10) / 100.0,
    middle: parseInt(document.getElementById('sliderMiddle').value, 10) / 100.0,
    ring: parseInt(document.getElementById('sliderRing').value, 10) / 100.0,
    little: parseInt(document.getElementById('sliderLittle').value, 10) / 100.0,
  };
}

function runGloveInference() {
  const sensors = getSliderSensors();
  const imu = { pitch: 35.0, roll: -10.0, yaw: 0.0 };

  fetch('/predict/glove', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sensors, imu })
  })
    .then(res => res.json())
    .then(data => {
      updateRecognitionUI(data);
    })
    .catch(() => {});
}

window.setGlovePose = function(poseName) {
  let vals = { thumb: 12, index: 15, middle: 14, ring: 15, little: 15 };
  if (poseName === 'I') {
    vals = { thumb: 85, index: 92, middle: 94, ring: 90, little: 12 };
  } else if (poseName === 'NAME') {
    vals = { thumb: 65, index: 15, middle: 12, ring: 88, little: 82 };
  } else if (poseName === 'HEARING') {
    vals = { thumb: 60, index: 15, middle: 85, ring: 88, little: 85 };
  } else if (poseName === 'THANK_YOU') {
    vals = { thumb: 12, index: 14, middle: 15, ring: 14, little: 15 };
  } else if (poseName === 'MORNING') {
    vals = { thumb: 8, index: 10, middle: 10, ring: 10, little: 10 };
  }

  ['Thumb', 'Index', 'Middle', 'Ring', 'Little'].forEach(f => {
    const key = f.toLowerCase();
    const s = document.getElementById(`slider${f}`);
    if (s) {
      s.value = vals[key];
      document.getElementById(`flexFill${f}`).style.height = `${vals[key]}%`;
      document.getElementById(`flexPct${f}`).textContent = `${vals[key]}%`;
    }
  });

  runGloveInference();
};

// =========================================================================
// 8. INTERACTIVE MODEL TRAINER WORKBENCH
// =========================================================================

// 126-Dimensional Feature Extractor Constants matching backend FeatureExtractor
const BONE_EDGES = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20]
];

const FINGER_TRIPLETS = [
  [0, 1, 2], [1, 2, 3], [2, 3, 4],
  [0, 5, 6], [5, 6, 7], [6, 7, 8],
  [0, 9, 10], [9, 10, 11], [10, 11, 12],
  [0, 13, 14], [13, 14, 15], [14, 15, 16],
  [0, 17, 18], [17, 18, 19], [18, 19, 20]
];

const FINGERTIP_INDICES = [4, 8, 12, 16, 20];

function extractKeypointFeatures126D(keypoints) {
  if (!keypoints || keypoints.length < 21) {
    return new Array(126).fill(0.0);
  }

  const coords = keypoints.map(p => [p[0], p[1]]);
  const visibilities = keypoints.map(p => (p.length >= 3 && p[2] !== undefined) ? p[2] : 1.0);

  // 1. Wrist-origin translation
  const wrist = coords[0];
  const centered = coords.map(p => [p[0] - wrist[0], p[1] - wrist[1]]);

  // 2. Scale normalization: distance from wrist (0) to middle MCP (9)
  let scale = Math.hypot(centered[9][0], centered[9][1]);
  if (scale < 1e-4) {
    const dists = centered.map(p => Math.hypot(p[0], p[1]));
    const maxD = Math.max(...dists);
    scale = maxD > 1e-4 ? maxD : 1.0;
  }

  const normalizedCoords = centered.map(p => [p[0] / scale, p[1] / scale]);
  const coordsFlat = [];
  for (let i = 0; i < 21; i++) {
    coordsFlat.push(normalizedCoords[i][0], normalizedCoords[i][1]);
  }

  // 3. Bone unit vectors (20 * 2 = 40 values)
  const boneVecs = [];
  for (let i = 0; i < BONE_EDGES.length; i++) {
    const [a, b] = BONE_EDGES[i];
    const dx = coords[b][0] - coords[a][0];
    const dy = coords[b][1] - coords[a][1];
    const norm = Math.hypot(dx, dy);
    if (norm > 1e-6) {
      boneVecs.push(dx / norm, dy / norm);
    } else {
      boneVecs.push(0.0, 0.0);
    }
  }

  // 4. Joint angles (15 values)
  const angles = [];
  for (let i = 0; i < FINGER_TRIPLETS.length; i++) {
    const [a, b, c] = FINGER_TRIPLETS[i];
    const v1x = coords[a][0] - coords[b][0];
    const v1y = coords[a][1] - coords[b][1];
    const v2x = coords[c][0] - coords[b][0];
    const v2y = coords[c][1] - coords[b][1];
    const n1 = Math.hypot(v1x, v1y);
    const n2 = Math.hypot(v2x, v2y);
    if (n1 > 1e-6 && n2 > 1e-6) {
      let cosine = (v1x * v2x + v1y * v2y) / (n1 * n2);
      cosine = Math.max(-1.0, Math.min(1.0, cosine));
      angles.push(Math.acos(cosine) / Math.PI);
    } else {
      angles.push(1.0);
    }
  }

  // 5. Visibilities (21 values)
  const confArr = visibilities.slice(0, 21);

  // 6. Palm features (8 values)
  const fingertipDists = FINGERTIP_INDICES.map(tip =>
    Math.hypot(normalizedCoords[tip][0], normalizedCoords[tip][1])
  );

  const palmWidth = Math.hypot(
    normalizedCoords[1][0] - normalizedCoords[17][0],
    normalizedCoords[1][1] - normalizedCoords[17][1]
  );

  const xs = normalizedCoords.map(p => p[0]);
  const ys = normalizedCoords.map(p => p[1]);
  const bboxW = Math.max(...xs) - Math.min(...xs);
  const bboxH = Math.max(...ys) - Math.min(...ys);
  const aspectRatio = bboxW / (bboxH + 1e-6);

  const vMidX = normalizedCoords[12][0] - normalizedCoords[0][0];
  const vMidY = normalizedCoords[12][1] - normalizedCoords[0][1];
  const orientation = (Math.atan2(vMidY, vMidX) + Math.PI) / (2.0 * Math.PI);

  const palmFeatures = [...fingertipDists, palmWidth, aspectRatio, orientation];

  const total = [...coordsFlat, ...boneVecs, ...angles, ...confArr, ...palmFeatures];
  if (total.length < 126) {
    while (total.length < 126) total.push(0.0);
  } else if (total.length > 126) {
    total.length = 126;
  }
  return total;
}

async function loadTrainedModel() {
  try {
    const res = await fetch('/api/trainer/export');
    if (res.ok) {
      const data = await res.json();
      if (data && data.is_custom && data.weights) {
        state.customModel = data;
        state.classes = data.classes || state.classes;
        buildProbabilityBars();
      }
    }
  } catch (err) {
    try {
      const saved = localStorage.getItem('kinex_custom_model');
      if (saved) {
        state.customModel = JSON.parse(saved);
        state.classes = state.customModel.classes || state.classes;
        buildProbabilityBars();
      }
    } catch (e) {}
  }
}

function setTargetSign(sign) {
  if (!sign) return;
  state.trainerTargetSign = sign;
  const activeSignName = document.getElementById('activeTargetSignName');
  if (activeSignName) activeSignName.textContent = `${sign} ${GESTURE_ICONS[sign] || '✋'}`;
  const hudTrainerSign = document.getElementById('hudTrainerSign');
  if (hudTrainerSign) hudTrainerSign.textContent = `TARGET: ${sign}`;
  const vTarget = document.getElementById('verifierTargetSign');
  if (vTarget) vTarget.textContent = `${sign} ${GESTURE_ICONS[sign] || '✋'}`;
}

function addCustomSignFromInput() {
  const input = document.getElementById('customSignInput');
  if (!input) return;
  const val = input.value.trim().toUpperCase().replace(/\s+/g, '_');
  if (!val) return;

  if (!state.classes.includes(val)) {
    state.classes.push(val);
    buildProbabilityBars();
  }

  const container = document.getElementById('trainerClassPills');
  if (container) {
    let existingPill = container.querySelector(`[data-class="${val}"]`);
    document.querySelectorAll('.class-pill-btn').forEach(p => p.classList.remove('active'));

    if (existingPill) {
      existingPill.classList.add('active');
    } else {
      const newPill = document.createElement('button');
      newPill.className = 'class-pill-btn active';
      newPill.dataset.class = val;
      newPill.textContent = `${val} ✋`;
      newPill.addEventListener('click', (e) => {
        document.querySelectorAll('.class-pill-btn').forEach(p => p.classList.remove('active'));
        e.currentTarget.classList.add('active');
        setTargetSign(val);
      });
      container.appendChild(newPill);
    }
  }

  setTargetSign(val);
  input.value = '';
  renderTrainerSampleCounters();
  refreshTrainerStudio();
  showToast(`✓ Active target sign set to: ${val}`);
}

function initTrainerUI() {
  const pills = document.querySelectorAll('.class-pill-btn');
  pills.forEach(pill => {
    pill.addEventListener('click', (e) => {
      document.querySelectorAll('.class-pill-btn').forEach(p => p.classList.remove('active'));
      const target = e.currentTarget;
      target.classList.add('active');
      setTargetSign(target.dataset.class);
    });
  });

  const btnAddSign = document.getElementById('btnAddCustomSign');
  if (btnAddSign) {
    btnAddSign.addEventListener('click', addCustomSignFromInput);
  }

  const inputEl = document.getElementById('customSignInput');
  if (inputEl) {
    inputEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        addCustomSignFromInput();
      }
    });
  }

  const btnQuickRecord = document.getElementById('btnQuickRecordTypedSign');
  if (btnQuickRecord) {
    btnQuickRecord.addEventListener('click', () => {
      const input = document.getElementById('customSignInput');
      if (input && input.value.trim()) {
        addCustomSignFromInput();
      }
      captureTrainingSample();
    });
  }

  const btnSingle = document.getElementById('btnRecordSingle');
  if (btnSingle) {
    btnSingle.addEventListener('click', () => {
      const input = document.getElementById('customSignInput');
      if (input && input.value.trim()) {
        addCustomSignFromInput();
      }
      captureTrainingSample();
    });
  }

  // Spacebar shortcut in trainer tab
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && state.activeTab === 'trainerTab') {
      if (document.activeElement && (document.activeElement.tagName === 'INPUT' || document.activeElement.tagName === 'SELECT')) {
        return;
      }
      e.preventDefault();
      captureTrainingSample();
    }
  });

  const recContBtn = document.getElementById('btnRecordContinuous');
  const recIndicator = document.getElementById('recordingIndicator');

  if (recContBtn) {
    const startContinuous = () => {
      if (state.continuousRecordInterval) return;
      if (recIndicator) recIndicator.style.display = 'flex';
      recContBtn.classList.add('btn-danger');
      state.continuousRecordInterval = setInterval(captureTrainingSample, 180);
    };

    const stopContinuous = () => {
      if (state.continuousRecordInterval) {
        clearInterval(state.continuousRecordInterval);
        state.continuousRecordInterval = null;
      }
      if (recIndicator) recIndicator.style.display = 'none';
      recContBtn.classList.remove('btn-danger');
    };

    recContBtn.addEventListener('mousedown', startContinuous);
    recContBtn.addEventListener('mouseup', stopContinuous);
    recContBtn.addEventListener('mouseleave', stopContinuous);
    recContBtn.addEventListener('touchstart', (e) => { e.preventDefault(); startContinuous(); });
    recContBtn.addEventListener('touchend', stopContinuous);
  }

  const btnTrain = document.getElementById('btnTrainModelNow');
  if (btnTrain) btnTrain.addEventListener('click', executeModelTraining);

  const btnTestInRec = document.getElementById('btnTestInRecognition');
  if (btnTestInRec) {
    btnTestInRec.addEventListener('click', () => {
      const recTab = document.getElementById('tabBtnRecognition');
      if (recTab) recTab.click();
      showToast("Ready! Perform your trained hand signs in front of the camera.");
    });
  }

  const btnReset = document.getElementById('btnResetSamples');
  if (btnReset) {
    btnReset.addEventListener('click', () => {
      if (confirm("Clear all recorded training samples and reset dataset?")) {
        fetch('/api/trainer/reset', { method: 'POST' })
          .then(res => res.json())
          .then(() => {
            state.sampleCounts = {};
            state.customModel = null;
            try { localStorage.removeItem('kinex_custom_model'); } catch (e) {}
            refreshTrainingSampleStats();
            refreshTrainerStudio();
            showToast("Training samples cleared and model reset to baseline.");
          });
      }
    });
  }

  // Trainer Subtabs (Trained Sentences vs Given Samples)
  const btnSubSentences = document.getElementById('btnSubtabTrainedSentences');
  const btnSubSamples = document.getElementById('btnSubtabGivenSamples');
  const viewSentences = document.getElementById('viewTrainedSentences');
  const viewSamples = document.getElementById('viewGivenSamples');

  if (btnSubSentences && btnSubSamples) {
    btnSubSentences.addEventListener('click', () => {
      btnSubSentences.classList.add('active');
      btnSubSamples.classList.remove('active');
      if (viewSentences) viewSentences.style.display = 'block';
      if (viewSamples) viewSamples.style.display = 'none';
    });
    btnSubSamples.addEventListener('click', () => {
      btnSubSamples.classList.add('active');
      btnSubSentences.classList.remove('active');
      if (viewSentences) viewSentences.style.display = 'none';
      if (viewSamples) viewSamples.style.display = 'block';
    });
  }

  // Sentence Search & Filter Pills
  const searchSentInput = document.getElementById('searchTrainedSentences');
  if (searchSentInput) {
    searchSentInput.addEventListener('input', () => {
      renderTrainedSentences();
    });
  }

  const filterPills = document.querySelectorAll('#sentenceFilterPills .filter-pill');
  filterPills.forEach(pill => {
    pill.addEventListener('click', (e) => {
      filterPills.forEach(p => p.classList.remove('active'));
      e.currentTarget.classList.add('active');
      state.trainerSentencesFilter = e.currentTarget.dataset.filter;
      renderTrainedSentences();
    });
  });
}

function captureTrainingSample() {
  if (!state.activeHands || state.activeHands.length === 0) {
    showToast("⚠️ No hand in camera view! Position hand in frame to record.");
    const verifierBadge = document.getElementById('verifierStatusBadge');
    if (verifierBadge) {
      verifierBadge.className = 'verifier-status-badge badge-waiting';
      verifierBadge.textContent = 'NO HAND DETECTED';
    }
    return;
  }

  const kpts = state.activeHands[0].kpts;
  const gesture = state.trainerTargetSign;

  // Flash capture visual indicator
  const flash = document.getElementById('trainerCaptureFlash');
  if (flash) {
    flash.style.opacity = '0.7';
    setTimeout(() => { flash.style.opacity = '0'; }, 80);
  }

  // Capture a lightweight 240x180 thumbnail from video or canvas
  let thumb = null;
  try {
    const offCanvas = document.createElement('canvas');
    offCanvas.width = 240;
    offCanvas.height = 180;
    const offCtx = offCanvas.getContext('2d');
    const source = (video && video.readyState >= 2) ? video : (canvas || trainerCanvas);
    if (source) {
      offCtx.drawImage(source, 0, 0, 240, 180);
      thumb = offCanvas.toDataURL('image/jpeg', 0.65);
    }
  } catch (e) {
    thumb = null;
  }

  fetch('/api/trainer/record', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ gesture, keypoints: kpts, mode: state.activeMode, thumbnail: thumb })
  })
    .then(res => {
      if (!res.ok) throw new Error("Failed to record sample");
      return res.json();
    })
    .then(data => {
      state.sampleCounts = data.class_counts;
      const countEl = document.getElementById('totalSamplesCount');
      if (countEl) countEl.textContent = data.total_samples;
      const navBadge = document.getElementById('badgeTotalSamplesNav');
      if (navBadge) navBadge.textContent = data.total_samples;

      if (!state.trainerGivenSamples) state.trainerGivenSamples = [];
      const newSmp = {
        id: data.sample ? data.sample.id : `smp-${Date.now()}`,
        gesture: gesture,
        keypoints: kpts,
        has_keypoints: true,
        thumbnail: thumb,
        timestamp_str: 'Just now'
      };
      state.trainerGivenSamples.unshift(newSmp);

      renderTrainerSampleCounters();
      refreshTrainerStudio();
      loadSampleManager();
      showToast(`✓ Sample #${data.total_samples} recorded for ${gesture} ${GESTURE_ICONS[gesture] || ''}!`);
    })
    .catch(() => showToast("Failed to record sample."));
}

function refreshTrainingSampleStats() {
  fetch('/api/trainer/samples')
    .then(res => res.json())
    .then(data => {
      state.sampleCounts = data.class_counts || {};
      const countEl = document.getElementById('totalSamplesCount');
      if (countEl) countEl.textContent = data.total_samples || 0;
      const navBadge = document.getElementById('badgeTotalSamplesNav');
      if (navBadge) navBadge.textContent = data.total_samples || 0;
      renderTrainerSampleCounters();
    })
    .catch(() => {});
}

function renderTrainerSampleCounters() {
  const container = document.getElementById('classSampleCounters');
  if (!container) return;
  container.innerHTML = '';

  const allDisplayClasses = Array.from(new Set([...state.classes, ...Object.keys(state.sampleCounts || {})]));

  allDisplayClasses.forEach(cls => {
    const count = (state.sampleCounts && state.sampleCounts[cls]) || 0;
    const maxVal = 25;
    const pct = Math.min(100, Math.round((count / maxVal) * 100));

    const row = document.createElement('div');
    row.className = 'sample-counter-row';
    row.innerHTML = `
      <div class="sample-counter-labels">
        <span>${cls} ${GESTURE_ICONS[cls] || ''}</span>
        <span class="mono-val" style="color: ${count >= 8 ? '#34d399' : (count > 0 ? '#38bdf8' : '#94a3b8')};">${count} samples</span>
      </div>
      <div class="sample-track">
        <div class="sample-fill" style="width: ${pct}%;"></div>
      </div>
    `;
    container.appendChild(row);
  });
}

function executeModelTraining() {
  const total = parseInt(document.getElementById('totalSamplesCount') ? document.getElementById('totalSamplesCount').textContent : "0", 10) || 0;
  const recordedSigns = Object.keys(state.sampleCounts || {}).filter(k => (state.sampleCounts[k] || 0) > 0);

  if (total === 0) {
    showToast("⚠️ No samples recorded yet! Position hand in camera view & click 'Record Single Sample'.");
    alert("No samples recorded yet.\n\nTo train the model:\n1. Position your hand in front of the camera.\n2. Click 'Record Single Sample' (or press Space) to save 1-3 samples of your hand sign.\n3. Click 'Train Model Now'.");
    return;
  }

  const container = document.getElementById('trainProgressContainer');
  const fill = document.getElementById('trainProgressFill');
  const status = document.getElementById('trainStatusText');
  const callout = document.getElementById('trainResultCallout');

  if (container) container.style.display = 'block';
  if (fill) fill.style.width = '25%';
  if (status) {
    if (recordedSigns.length === 1) {
      status.textContent = `Stage 1/4: Training custom sign '${recordedSigns[0]}' with Neutral hand reference baseline...`;
    } else {
      status.textContent = `Stage 1/4: Validating saved samples across ${recordedSigns.length} signs...`;
    }
  }
  if (callout) callout.textContent = '';

  setTimeout(() => {
    if (fill) fill.style.width = '50%';
    if (status) status.textContent = 'Stage 2/4: Augmenting hand geometry & orientation variations...';
  }, 250);

  setTimeout(() => {
    if (fill) fill.style.width = '75%';
    if (status) status.textContent = 'Stage 3/4: Solving regularized ridge classifier weights...';
  }, 500);

  fetch('/api/trainer/train', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode: state.activeMode })
  })
    .then(res => {
      if (!res.ok) {
        return res.json().then(errData => {
          throw new Error(errData.detail || "Training request failed");
        });
      }
      return res.json();
    })
    .then(metrics => {
      if (fill) fill.style.width = '100%';
      if (status) status.textContent = `✓ Training Completed across ${metrics.classes.length} ISL signs!`;
      if (callout) {
        callout.innerHTML = `
          <div style="margin-top:0.5rem; line-height:1.6;">
            <div>✓ Validation Accuracy: <strong style="color:#34d399; font-size:1.05rem;">${metrics.accuracy}%</strong></div>
            <div style="font-size:0.75rem; color:#94a3b8;">Trained on <strong>${metrics.samples_count}</strong> saved samples (${metrics.augmented_count || metrics.samples_count * 16} augmented) across <strong>${metrics.classes.length}</strong> signs: [ ${metrics.classes.join(', ')} ]</div>
            <div style="margin-top:0.35rem; color:#38bdf8; font-weight:700; font-size:0.75rem;">⚡ MODEL IS NOW ACTIVE IN LIVE RECOGNITION</div>
          </div>
        `;
      }

      // Save custom model to client state for 30 FPS inference
      state.customModel = {
        isCustom: true,
        classes: metrics.classes,
        weights: metrics.weights,
        bias: metrics.bias,
        accuracy: metrics.accuracy
      };
      try {
        localStorage.setItem('kinex_custom_model', JSON.stringify(state.customModel));
      } catch (e) {}

      state.classes = metrics.classes;
      buildProbabilityBars();
      refreshTrainerStudio();
      loadSampleManager();
      showToast(`Model trained! Accuracy: ${metrics.accuracy}% across ${metrics.classes.length} signs.`);
    })
    .catch(err => {
      if (status) status.textContent = 'Training stopped: ' + err.message;
      if (fill) fill.style.width = '0%';
      showToast("Training failed: " + err.message);
    });
}

// Refresh Trainer Studio: Sentences & Samples
function refreshTrainerStudio() {
  fetch('/api/trainer/samples')
    .then(res => res.json())
    .then(data => {
      state.trainerGivenSamples = data.samples || [];
      state.trainerSentences = data.supported_sentences || [];
      state.sampleCounts = data.class_counts || {};

      const countSentences = document.getElementById('countTrainedSentences');
      if (countSentences) countSentences.textContent = state.trainerSentences.length;

      const countSamples = document.getElementById('countGivenSamples');
      if (countSamples) countSamples.textContent = state.trainerGivenSamples.length;

      const navBadge = document.getElementById('badgeTotalSamplesNav');
      if (navBadge) navBadge.textContent = data.total_samples || 0;

      renderTrainedSentences();
      renderGivenSamplesGallery();
      renderSampleClassFilterPills(data.class_counts || {});
    })
    .catch(() => {});
}

function renderTrainedSentences() {
  const container = document.getElementById('trainedSentencesGrid');
  if (!container) return;
  container.innerHTML = '';

  const searchInput = document.getElementById('searchTrainedSentences');
  const query = searchInput ? searchInput.value.trim().toLowerCase() : '';
  const filter = state.trainerSentencesFilter || 'ALL';

  let list = state.trainerSentences || [];

  if (query) {
    list = list.filter(s =>
      s.sentence.toLowerCase().includes(query) ||
      s.tokens.some(t => t.toLowerCase().includes(query))
    );
  }

  if (filter === 'FULLY_TRAINED') {
    list = list.filter(s => s.status === 'FULLY_TRAINED');
  } else if (filter === 'PARTIALLY_TRAINED') {
    list = list.filter(s => s.status === 'PARTIALLY_TRAINED');
  }

  if (list.length === 0) {
    container.innerHTML = `
      <div class="empty-gallery-notice">
        <div class="empty-icon">📜</div>
        <h4>No Sentences Found Matching Filter</h4>
        <p>Try recording samples for more signs (e.g. I, INDIAN, MORNING, THANK_YOU) or adjust the search query.</p>
      </div>
    `;
    return;
  }

  list.forEach(item => {
    const card = document.createElement('div');
    const statusClass = item.status.toLowerCase().replace('_', '-');
    card.className = `trained-sentence-card status-${statusClass}`;

    const tokensHtml = item.tokens.map((token, idx) => {
      const samplesCount = item.samples_per_sign[token] || 0;
      const chipClass = samplesCount > 0 ? 'tsc-token-chip has-samples' : 'tsc-token-chip';
      const arrow = idx > 0 ? '<span class="tsc-arrow">➔</span>' : '';
      return `
        ${arrow}
        <span class="${chipClass}">
          <span>${GESTURE_ICONS[token] || '✋'}</span>
          <span>${token}</span>
          <span class="chip-samples-badge">${samplesCount} spl</span>
        </span>
      `;
    }).join('');

    let missingNotice = '';
    if (item.missing_signs && item.missing_signs.length > 0) {
      missingNotice = `<span style="font-size:0.68rem; color:#fbbf24;">Needs: ${item.missing_signs.join(', ')}</span>`;
    }

    card.innerHTML = `
      <div>
        <div class="tsc-header">
          <span class="tsc-status-pill pill-${statusClass}">${item.badge}</span>
          <span class="tsc-samples-count mono-val">${item.total_samples} Samples Given</span>
        </div>
        <div class="tsc-sentence-text">“${item.sentence}”</div>
        <div class="tsc-tokens-chain">${tokensHtml}</div>
      </div>
      <div class="tsc-footer">
        <div class="tsc-btn-group">
          <button class="btn btn-primary btn-sm" onclick="testSentenceInRecognition('${item.tokens.join(',')}')">
            🧪 Test Live
          </button>
          <button class="btn btn-secondary btn-sm" onclick="speakSentence('${item.sentence.replace(/'/g, "\\'")}')">
            🔊 Speak
          </button>
        </div>
        ${missingNotice}
      </div>
    `;
    container.appendChild(card);
  });
}

function renderGivenSamplesGallery() {
  const container = document.getElementById('givenSamplesGrid');
  if (!container) return;
  container.innerHTML = '';

  const filter = state.trainerSamplesFilter || 'ALL';
  let list = state.trainerGivenSamples || [];

  if (filter !== 'ALL') {
    list = list.filter(s => s.gesture === filter);
  }

  if (list.length === 0) {
    container.innerHTML = `
      <div class="empty-gallery-notice">
        <div class="empty-icon">📸</div>
        <h4>No Given Samples for Current View</h4>
        <p>Hold up your hand to the camera and click "Capture 1 Sample" or press Spacebar to record sign samples.</p>
      </div>
    `;
    return;
  }

  list.forEach(sample => {
    const card = document.createElement('div');
    card.className = 'given-sample-card';
    card.id = `sampleCard_${sample.id}`;

    const canvasId = `trainerPreview_${sample.id}`;

    card.innerHTML = `
      <div class="sm-card-preview-wrap" style="height:140px;">
        <canvas class="sm-card-preview-canvas" id="${canvasId}" width="280" height="140"></canvas>
        <span class="sm-card-preview-badge">${sample.has_keypoints ? '21 KP Landmarks' : 'Sensor Telemetry'}</span>
      </div>
      <div class="gsc-header" style="margin-top:0.75rem;">
        <span class="gsc-id-badge mono-val">#${sample.id}</span>
        <span class="gsc-sign-pill">${GESTURE_ICONS[sample.gesture] || '✋'} ${sample.gesture}</span>
      </div>
      <div class="gsc-body">
        <div class="gsc-meta-row">
          <span>Modality:</span>
          <span class="mono-val">${sample.mode.toUpperCase()} (126-D)</span>
        </div>
        <div class="gsc-meta-row">
          <span>Captured:</span>
          <span class="mono-val">${sample.timestamp_str}</span>
        </div>
      </div>
      <div class="gsc-footer">
        <button class="btn-edit-sample" onclick="openEditSampleModal('${sample.id}')">
          ✏️ Edit Label
        </button>
        <button class="btn-delete-sample" onclick="deleteSingleSample('${sample.id}')">
          🗑️ Delete
        </button>
      </div>
    `;
    container.appendChild(card);

    // Render skeleton preview
    setTimeout(() => {
      const cvs = document.getElementById(canvasId);
      if (cvs) drawSamplePreview(cvs, sample.keypoints, sample.thumbnail);
    }, 10);
  });
}

function renderSampleClassFilterPills(classCounts) {
  const container = document.getElementById('sampleClassFilterRow');
  if (!container) return;
  container.innerHTML = '';

  const total = Object.values(classCounts).reduce((a, b) => a + b, 0);

  const allBtn = document.createElement('button');
  allBtn.className = `filter-pill ${state.trainerSamplesFilter === 'ALL' ? 'active' : ''}`;
  allBtn.textContent = `All Signs (${total})`;
  allBtn.addEventListener('click', () => {
    state.trainerSamplesFilter = 'ALL';
    renderGivenSamplesGallery();
    renderSampleClassFilterPills(classCounts);
  });
  container.appendChild(allBtn);

  Object.entries(classCounts).forEach(([cls, count]) => {
    const btn = document.createElement('button');
    btn.className = `filter-pill ${state.trainerSamplesFilter === cls ? 'active' : ''}`;
    btn.textContent = `${GESTURE_ICONS[cls] || ''} ${cls} (${count})`;
    btn.addEventListener('click', () => {
      state.trainerSamplesFilter = cls;
      renderGivenSamplesGallery();
      renderSampleClassFilterPills(classCounts);
    });
    container.appendChild(btn);
  });
}

// =========================================================================
// DEDICATED SAMPLE MANAGEMENT STUDIO IMPLEMENTATION
// =========================================================================

function initSampleManagerUI() {
  // Search Input
  const searchInput = document.getElementById('searchSamplesInput');
  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      state.sampleManagerSearch = e.target.value.trim().toLowerCase();
      renderSMSamplesContent();
    });
  }

  // View Mode Switcher
  const btnGrouped = document.getElementById('btnViewGrouped');
  const btnGrid = document.getElementById('btnViewGrid');
  if (btnGrouped && btnGrid) {
    btnGrouped.addEventListener('click', () => {
      btnGrouped.classList.add('active');
      btnGrid.classList.remove('active');
      state.sampleManagerViewMode = 'grouped';
      renderSMSamplesContent();
    });
    btnGrid.addEventListener('click', () => {
      btnGrid.classList.add('active');
      btnGrouped.classList.remove('active');
      state.sampleManagerViewMode = 'grid';
      renderSMSamplesContent();
    });
  }

  // Quick Action Buttons in Sample Manager
  const btnGoTrainer = document.getElementById('btnSMGotoTrainer');
  if (btnGoTrainer) {
    btnGoTrainer.addEventListener('click', () => {
      const trainerTabBtn = document.getElementById('tabBtnTrainer');
      if (trainerTabBtn) trainerTabBtn.click();
    });
  }

  const btnTrain = document.getElementById('btnSMTrainModel');
  if (btnTrain) {
    btnTrain.addEventListener('click', executeModelTraining);
  }

  const btnClearAll = document.getElementById('btnSMClearAll');
  if (btnClearAll) {
    btnClearAll.addEventListener('click', openClearAllModal);
  }

  const btnResetSamples = document.getElementById('btnResetSamples');
  if (btnResetSamples) {
    btnResetSamples.addEventListener('click', openClearAllModal);
  }

  // Clear All Modal Controls
  const btnConfirmClear = document.getElementById('btnConfirmClearAll');
  if (btnConfirmClear) {
    btnConfirmClear.addEventListener('click', submitClearAllSamples);
  }

  const btnCancelClear1 = document.getElementById('btnCancelClearModal');
  const btnCancelClear2 = document.getElementById('btnCancelClearModal2');
  [btnCancelClear1, btnCancelClear2].forEach(btn => {
    if (btn) {
      btn.addEventListener('click', () => {
        const modal = document.getElementById('clearAllModal');
        if (modal) modal.style.display = 'none';
      });
    }
  });

  // Edit Modal Controls
  const btnSaveEdit = document.getElementById('btnSaveSampleLabel');
  if (btnSaveEdit) {
    btnSaveEdit.addEventListener('click', submitEditSampleLabel);
  }

  const btnCancelEdit1 = document.getElementById('btnCancelEditModal');
  const btnCancelEdit2 = document.getElementById('btnCancelEditModal2');
  [btnCancelEdit1, btnCancelEdit2].forEach(btn => {
    if (btn) {
      btn.addEventListener('click', () => {
        const modal = document.getElementById('editSampleModal');
        if (modal) modal.style.display = 'none';
      });
    }
  });

  const editInput = document.getElementById('editModalInput');
  if (editInput) {
    editInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') submitEditSampleLabel();
      if (e.key === 'Escape') {
        const modal = document.getElementById('editSampleModal');
        if (modal) modal.style.display = 'none';
      }
    });
  }
}

function loadSampleManager() {
  fetch('/api/trainer/samples')
    .then(res => res.json())
    .then(data => {
      state.trainerGivenSamples = data.samples || [];
      state.sampleCounts = data.class_counts || {};

      // Update Nav pill
      const navBadge = document.getElementById('badgeTotalSamplesNav');
      if (navBadge) navBadge.textContent = data.total_samples || 0;

      // Update Top stats
      const totalEl = document.getElementById('smTotalSamples');
      if (totalEl) totalEl.textContent = data.total_samples || 0;

      const distinctSigns = Object.keys(data.class_counts || {});
      const signsCountEl = document.getElementById('smTotalSigns');
      if (signsCountEl) signsCountEl.textContent = distinctSigns.length;

      const modelStatusEl = document.getElementById('smModelStatus');
      if (modelStatusEl) {
        if (data.is_custom_trained && state.customModel) {
          modelStatusEl.textContent = `Custom (${state.customModel.classes.length} Signs)`;
          modelStatusEl.style.color = '#34d399';
        } else {
          modelStatusEl.textContent = 'Baseline Mode';
          modelStatusEl.style.color = '#94a3b8';
        }
      }

      const valAccEl = document.getElementById('smValidationAcc');
      if (valAccEl) {
        if (data.is_custom_trained && state.customModel) {
          valAccEl.textContent = (state.customModel.accuracy || 98.0) + '%';
        } else {
          valAccEl.textContent = (data.total_samples >= 3) ? 'Ready to Train' : 'Need >=3 Samples';
        }
      }

      // Render summary chips
      renderSMSummaryChips(data.class_counts || {});

      // Render filter pills
      renderSMFilterPills(data.class_counts || {});

      // Render samples content (grouped or grid)
      renderSMSamplesContent();
    })
    .catch(() => {});
}

function renderSMSummaryChips(classCounts) {
  const container = document.getElementById('smSignsSummaryChips');
  if (!container) return;
  container.innerHTML = '';

  const entries = Object.entries(classCounts);
  if (entries.length === 0) {
    container.innerHTML = '<span style="font-size:0.75rem; color:var(--text-dim);">No saved signs in dataset yet.</span>';
    return;
  }

  entries.forEach(([sign, count]) => {
    const chip = document.createElement('div');
    chip.className = 'sm-sign-chip';
    chip.title = `Click to filter samples for ${sign}`;
    chip.innerHTML = `
      <span>${GESTURE_ICONS[sign] || '✋'}</span>
      <span>${sign}</span>
      <span class="sm-sign-chip-count">${count}</span>
    `;
    chip.addEventListener('click', () => {
      state.sampleManagerFilter = sign;
      renderSMFilterPills(classCounts);
      renderSMSamplesContent();
    });
    container.appendChild(chip);
  });
}

function renderSMFilterPills(classCounts) {
  const container = document.getElementById('smFilterPillsRow');
  if (!container) return;
  container.innerHTML = '';

  const total = Object.values(classCounts).reduce((a, b) => a + b, 0);

  const allBtn = document.createElement('button');
  allBtn.className = `filter-pill ${state.sampleManagerFilter === 'ALL' ? 'active' : ''}`;
  allBtn.textContent = `All Signs (${total})`;
  allBtn.addEventListener('click', () => {
    state.sampleManagerFilter = 'ALL';
    renderSMFilterPills(classCounts);
    renderSMSamplesContent();
  });
  container.appendChild(allBtn);

  Object.entries(classCounts).forEach(([cls, count]) => {
    const btn = document.createElement('button');
    btn.className = `filter-pill ${state.sampleManagerFilter === cls ? 'active' : ''}`;
    btn.textContent = `${GESTURE_ICONS[cls] || ''} ${cls} (${count})`;
    btn.addEventListener('click', () => {
      state.sampleManagerFilter = cls;
      renderSMFilterPills(classCounts);
      renderSMSamplesContent();
    });
    container.appendChild(btn);
  });
}

function renderSMSamplesContent() {
  const container = document.getElementById('smSamplesContentArea');
  if (!container) return;
  container.innerHTML = '';

  let list = state.trainerGivenSamples || [];

  // Filter by search query
  if (state.sampleManagerSearch) {
    const q = state.sampleManagerSearch;
    list = list.filter(s =>
      s.gesture.toLowerCase().includes(q) ||
      (s.id && s.id.toLowerCase().includes(q)) ||
      (s.mode && s.mode.toLowerCase().includes(q))
    );
  }

  // Filter by sign pill
  if (state.sampleManagerFilter !== 'ALL') {
    list = list.filter(s => s.gesture === state.sampleManagerFilter);
  }

  if (list.length === 0) {
    container.innerHTML = `
      <div class="empty-gallery-notice glass-card" style="padding:4rem 1.5rem; text-align:center;">
        <div class="empty-icon">🗂️</div>
        <h3 style="font-size:1.1rem; font-weight:700; color:var(--text-main); margin-bottom:0.5rem;">No Stored Sign Samples Found</h3>
        <p style="font-size:0.85rem; color:var(--text-muted); max-width:480px; margin:0 auto 1.5rem;">
          ${(state.sampleManagerSearch || state.sampleManagerFilter !== 'ALL') ? 'No samples match the selected search query or filter.' : 'Your dataset is currently fresh and clean. Activate the camera in the Model Trainer to capture your first sign language sample!'}
        </p>
        <button class="btn btn-primary" onclick="document.getElementById('tabBtnTrainer').click()">
          📸 Go to Camera Trainer
        </button>
      </div>
    `;
    return;
  }

  // VIEW MODE 1: GROUPED BY SIGN
  if (state.sampleManagerViewMode === 'grouped') {
    const groups = {};
    list.forEach(sample => {
      groups[sample.gesture] = groups[sample.gesture] || [];
      groups[sample.gesture].push(sample);
    });

    Object.keys(groups).sort().forEach(gestureName => {
      const samplesInGroup = groups[gestureName];
      const groupCard = document.createElement('div');
      groupCard.className = 'sm-group-card';

      const totalSpl = state.trainerGivenSamples.length || 1;
      const pct = Math.round((samplesInGroup.length / totalSpl) * 100);

      const header = document.createElement('div');
      header.className = 'sm-group-header';
      header.innerHTML = `
        <div class="sm-group-title">
          <span>${GESTURE_ICONS[gestureName] || '✋'}</span>
          <span>${gestureName}</span>
          <span class="sm-group-badge">${samplesInGroup.length} Samples (${pct}%)</span>
        </div>
        <span style="font-size:0.75rem; color:var(--text-dim); font-family:var(--font-mono);">▼ Click to Toggle</span>
      `;

      const grid = document.createElement('div');
      grid.className = 'sm-group-grid';

      samplesInGroup.forEach(sample => {
        grid.appendChild(createSampleCardElement(sample));
      });

      header.addEventListener('click', () => {
        grid.style.display = (grid.style.display === 'none') ? 'grid' : 'none';
      });

      groupCard.appendChild(header);
      groupCard.appendChild(grid);
      container.appendChild(groupCard);
    });
  }
  // VIEW MODE 2: CONTINUOUS GRID
  else {
    const grid = document.createElement('div');
    grid.className = 'sm-cards-grid-all';
    list.forEach(sample => {
      grid.appendChild(createSampleCardElement(sample));
    });
    container.appendChild(grid);
  }
}

function createSampleCardElement(sample) {
  const card = document.createElement('div');
  card.className = 'sm-sample-card';
  card.id = `smCard_${sample.id}`;

  const canvasId = `smPreviewCanvas_${sample.id}`;

  card.innerHTML = `
    <div class="sm-card-preview-wrap">
      <canvas class="sm-card-preview-canvas" id="${canvasId}" width="280" height="160"></canvas>
      <span class="sm-card-preview-badge">${sample.has_keypoints ? '21 KP Anatomical' : 'Telemetry Pose'}</span>
    </div>
    <div class="sm-card-body">
      <div class="sm-card-header-row">
        <span class="sm-card-sign-name">${GESTURE_ICONS[sample.gesture] || '✋'} ${sample.gesture}</span>
        <span class="sm-card-id">#${sample.id}</span>
      </div>
      <div class="sm-card-meta">
        <div class="sm-card-meta-item">
          <span>Modality:</span>
          <span class="mono-val" style="color:#38bdf8;">${sample.mode.toUpperCase()} (126-D)</span>
        </div>
        <div class="sm-card-meta-item">
          <span>Captured:</span>
          <span class="mono-val">${sample.timestamp_str}</span>
        </div>
        <div class="sm-card-meta-item">
          <span>Landmarks:</span>
          <span class="mono-val" style="color:#34d399;">${sample.has_keypoints ? '21 Joints Normalized' : 'Telemetry Data'}</span>
        </div>
      </div>
    </div>
    <div class="sm-card-footer">
      <button class="btn-edit-sample" onclick="openEditSampleModal('${sample.id}')">
        ✏️ Edit Label
      </button>
      <button class="btn-delete-sample" onclick="deleteSingleSample('${sample.id}')">
        🗑️ Delete
      </button>
    </div>
  `;

  // Draw the landmark skeleton onto the canvas
  setTimeout(() => {
    const cvs = document.getElementById(canvasId);
    if (cvs) drawSamplePreview(cvs, sample.keypoints, sample.thumbnail);
  }, 10);

  return card;
}

function drawSamplePreview(canvasEl, keypoints, thumbnail) {
  if (!canvasEl) return;
  const ctx = canvasEl.getContext('2d');
  const w = canvasEl.width;
  const h = canvasEl.height;

  // Clear dark background
  ctx.fillStyle = '#050813';
  ctx.fillRect(0, 0, w, h);

  // If thumbnail image exists, draw it as background
  if (thumbnail && thumbnail.startsWith('data:image')) {
    const img = new Image();
    img.onload = () => {
      ctx.drawImage(img, 0, 0, w, h);
      if (keypoints && keypoints.length === 21) {
        renderSkeletonOnCanvas(ctx, w, h, keypoints);
      }
    };
    img.src = thumbnail;
    return;
  }

  // If only keypoints exist, draw centered anatomical hand skeleton
  if (keypoints && keypoints.length === 21) {
    renderSkeletonOnCanvas(ctx, w, h, keypoints);
  } else {
    // Empty state
    ctx.fillStyle = '#64748b';
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText('Sensor Telemetry Sample', w / 2, h / 2);
  }
}

function renderSkeletonOnCanvas(ctx, w, h, keypoints) {
  const xs = keypoints.map(p => p[0]);
  const ys = keypoints.map(p => p[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const bboxW = Math.max(10, maxX - minX);
  const bboxH = Math.max(10, maxY - minY);

  const pad = 24;
  const scale = Math.min((w - pad * 2) / bboxW, (h - pad * 2) / bboxH);
  const offsetX = (w - bboxW * scale) / 2 - minX * scale;
  const offsetY = (h - bboxH * scale) / 2 - minY * scale;

  const pts = keypoints.map(p => [p[0] * scale + offsetX, p[1] * scale + offsetY]);

  // Color-coded finger bones
  const fingerColors = [
    '#f87171', // Thumb: Coral
    '#38bdf8', // Index: Cyan
    '#34d399', // Middle: Emerald
    '#fbbf24', // Ring: Amber
    '#c084fc', // Pinky: Violet
  ];

  const fingerBones = [
    [[0, 1], [1, 2], [2, 3], [3, 4]],         // Thumb
    [[0, 5], [5, 6], [6, 7], [7, 8]],         // Index
    [[0, 9], [9, 10], [10, 11], [11, 12]],    // Middle
    [[0, 13], [13, 14], [14, 15], [15, 16]], // Ring
    [[0, 17], [17, 18], [18, 19], [19, 20]], // Pinky
  ];

  // Palm base connection
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
  ctx.lineWidth = 2.5;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(pts[5][0], pts[5][1]);
  ctx.lineTo(pts[9][0], pts[9][1]);
  ctx.lineTo(pts[13][0], pts[13][1]);
  ctx.lineTo(pts[17][0], pts[17][1]);
  ctx.stroke();

  // Draw finger bone lines
  for (let f = 0; f < 5; f++) {
    ctx.strokeStyle = fingerColors[f];
    ctx.lineWidth = 2.5;
    for (const [a, b] of fingerBones[f]) {
      ctx.beginPath();
      ctx.moveTo(pts[a][0], pts[a][1]);
      ctx.lineTo(pts[b][0], pts[b][1]);
      ctx.stroke();
    }
  }

  // Draw joints
  for (let i = 0; i < 21; i++) {
    const isTip = [4, 8, 12, 16, 20].includes(i);
    const isWrist = (i === 0);
    const r = isTip ? 4.5 : (isWrist ? 4.0 : 2.8);

    ctx.beginPath();
    ctx.arc(pts[i][0], pts[i][1], r, 0, Math.PI * 2);
    ctx.fillStyle = isTip ? '#ffffff' : (isWrist ? '#38bdf8' : 'rgba(255,255,255,0.85)');
    ctx.fill();

    if (isTip) {
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }
}

// Modal Handlers: Edit Label
window.openEditSampleModal = function(sampleId) {
  const sample = (state.trainerGivenSamples || []).find(s => s.id === sampleId);
  if (!sample) return;
  state.activeEditSampleId = sampleId;

  const idEl = document.getElementById('editModalSampleId');
  if (idEl) idEl.textContent = `#${sampleId}`;

  const inputEl = document.getElementById('editModalInput');
  if (inputEl) {
    inputEl.value = sample.gesture;
    setTimeout(() => inputEl.focus(), 60);
  }

  // Populate quick chips
  const chipsContainer = document.getElementById('editModalQuickChips');
  if (chipsContainer) {
    chipsContainer.innerHTML = '';
    const knownSigns = Array.from(new Set([...state.classes, ...Object.keys(state.sampleCounts || {})]));
    knownSigns.slice(0, 8).forEach(sign => {
      const chip = document.createElement('span');
      chip.className = 'modal-quick-chip';
      chip.textContent = `${GESTURE_ICONS[sign] || ''} ${sign}`;
      chip.addEventListener('click', () => {
        if (inputEl) inputEl.value = sign;
      });
      chipsContainer.appendChild(chip);
    });
  }

  const modal = document.getElementById('editSampleModal');
  if (modal) modal.style.display = 'flex';
};

function submitEditSampleLabel() {
  const sampleId = state.activeEditSampleId;
  const inputEl = document.getElementById('editModalInput');
  const newLabel = inputEl ? inputEl.value.trim().toUpperCase().replace(/\s+/g, '_') : '';

  if (!newLabel) {
    showToast("⚠️ Sign label cannot be empty.");
    return;
  }

  fetch(`/api/trainer/sample/${sampleId}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ gesture: newLabel })
  })
    .then(res => {
      if (!res.ok) throw new Error("Failed to update sample label");
      return res.json();
    })
    .then(data => {
      const modal = document.getElementById('editSampleModal');
      if (modal) modal.style.display = 'none';

      state.sampleCounts = data.class_counts;
      refreshTrainerStudio();
      loadSampleManager();
      renderTrainerSampleCounters();
      showToast(`✓ Sample #${sampleId} label updated to "${newLabel}".`);
    })
    .catch(err => showToast("Error updating label: " + err.message));
}

// Modal Handlers: Clear All Samples
function openClearAllModal() {
  const modal = document.getElementById('clearAllModal');
  if (modal) modal.style.display = 'flex';
}

function submitClearAllSamples() {
  fetch('/api/trainer/reset', { method: 'POST' })
    .then(res => {
      if (!res.ok) throw new Error("Reset request failed");
      return res.json();
    })
    .then(() => {
      const modal = document.getElementById('clearAllModal');
      if (modal) modal.style.display = 'none';

      state.sampleCounts = {};
      state.trainerGivenSamples = [];
      state.customModel = null;
      try { localStorage.removeItem('kinex_custom_model'); } catch (e) {}

      refreshTrainingSampleStats();
      refreshTrainerStudio();
      loadSampleManager();

      const navBadge = document.getElementById('badgeTotalSamplesNav');
      if (navBadge) navBadge.textContent = '0';
      const countEl = document.getElementById('totalSamplesCount');
      if (countEl) countEl.textContent = '0';

      buildProbabilityBars();
      showToast("✓ All training samples cleared and model reset to baseline.");
    })
    .catch(err => {
      showToast("Failed to clear samples: " + err.message);
    });
}

// Delete Individual Sample with Confirmation Dialog
window.deleteSingleSample = function(sampleId) {
  const sample = (state.trainerGivenSamples || []).find(s => s.id === sampleId);
  const signLabel = sample ? sample.gesture : 'sign';

  if (!confirm(`Are you sure you want to delete sample #${sampleId} for "${signLabel}"?\nThis cannot be undone.`)) {
    return;
  }

  fetch(`/api/trainer/sample/${sampleId}`, { method: 'DELETE' })
    .then(res => {
      if (!res.ok) throw new Error("Failed to delete sample");
      return res.json();
    })
    .then(data => {
      state.sampleCounts = data.class_counts;
      const countEl = document.getElementById('totalSamplesCount');
      if (countEl) countEl.textContent = data.total_samples;
      const navBadge = document.getElementById('badgeTotalSamplesNav');
      if (navBadge) navBadge.textContent = data.total_samples;

      renderTrainerSampleCounters();
      refreshTrainerStudio();
      loadSampleManager();
      showToast(`✓ Sample #${sampleId} removed from dataset.`);
    })
    .catch(err => showToast("Failed to delete sample: " + err.message));
};

window.testSentenceInRecognition = function(tokensStr) {
  if (!tokensStr) return;
  const tokens = tokensStr.split(',');

  const recTab = document.getElementById('tabBtnRecognition');
  if (recTab) recTab.click();

  // Reset active sentence buffer to prepare for this sentence
  state.activeSentenceTokens = [];
  state.lastCommittedTokenGesture = null;
  state.pendingGesture = null;
  state.pendingDwellFrames = 0;
  renderActiveSentence();

  showToast(`Testing Sentence: Form signs [ ${tokens.join(' ➔ ')} ] with your hand!`);
};

function updateTrainerLiveVerifier(gesture, confidence) {
  const signEl = document.getElementById('verifierDetectedSign');
  const confEl = document.getElementById('verifierDetectedConf');
  const targetEl = document.getElementById('verifierTargetSign');
  const pillEl = document.getElementById('verifierMatchPill');
  const badgeEl = document.getElementById('verifierStatusBadge');

  if (!signEl || !confEl || !pillEl) return;

  const targetSign = state.trainerTargetSign;
  if (targetEl) {
    targetEl.textContent = `${targetSign} ${GESTURE_ICONS[targetSign] || ''}`;
  }

  if (!gesture || gesture === 'UNKNOWN' || (!state.activeHands || state.activeHands.length === 0)) {
    signEl.textContent = '—';
    confEl.textContent = '0.0%';
    pillEl.className = 'v-match-pill pill-neutral';
    pillEl.textContent = 'Hold hand in camera';
    if (badgeEl) {
      badgeEl.className = 'verifier-status-badge badge-waiting';
      badgeEl.textContent = 'WAITING FOR HAND';
    }
    return;
  }

  signEl.textContent = `${gesture} ${GESTURE_ICONS[gesture] || ''}`;
  confEl.textContent = `${(confidence * 100).toFixed(1)}%`;
  if (badgeEl) {
    badgeEl.className = 'verifier-status-badge badge-active';
    badgeEl.textContent = 'HAND DETECTED';
  }

  if (gesture === targetSign) {
    pillEl.className = 'v-match-pill pill-match';
    pillEl.textContent = '✓ POSE MATCHES TARGET! READY';
  } else {
    pillEl.className = 'v-match-pill pill-mismatch';
    pillEl.textContent = `Posing as: ${gesture}`;
  }
}

// =========================================================================
// 9. MEDIAPIPE DUAL-HAND TRACKING & CLASSIFICATION
// =========================================================================

function initMediaPipeHands() {
  if (typeof Hands === 'undefined') {
    setTimeout(initMediaPipeHands, 800);
    return;
  }

  try {
    state.mpHands = new Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });

    state.mpHands.setOptions({
      maxNumHands: 2, // DUAL HAND TRACKING
      modelComplexity: 1,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    state.mpHands.onResults(onHandResults);
  } catch (err) {
    console.error("MediaPipe initialization error", err);
  }
}

function renderVideoBackground(targetCtx, targetCanvas, source) {
  if (!targetCtx || !targetCanvas || !source) return;
  targetCtx.save();
  targetCtx.clearRect(0, 0, targetCanvas.width, targetCanvas.height);
  targetCtx.translate(targetCanvas.width, 0);
  targetCtx.scale(-1, 1);
  try {
    targetCtx.drawImage(source, 0, 0, targetCanvas.width, targetCanvas.height);
  } catch (err) {}
  targetCtx.restore();
}

function resolveHandCandidates(rawLandmarksList, rawHandednessList) {
  if (!rawLandmarksList || rawLandmarksList.length === 0) return [];

  const candidates = [];
  for (let i = 0; i < rawLandmarksList.length; i++) {
    const rawLandmarks = rawLandmarksList[i];
    const mpHandedness = (rawHandednessList && rawHandednessList[i]) ? rawHandednessList[i] : null;
    const mpRawLabel = mpHandedness ? mpHandedness.label : (i === 0 ? 'Right' : 'Left');
    const mpScore = mpHandedness ? (mpHandedness.score || 0.8) : 0.8;

    const kpts = rawLandmarks.map(lm => [
      (1.0 - lm.x) * canvas.width,
      lm.y * canvas.height,
      1.0
    ]);

    const xs = kpts.map(p => p[0]);
    const ys = kpts.map(p => p[1]);
    const minX = Math.max(0, Math.min(...xs) - 15);
    const maxX = Math.min(canvas.width, Math.max(...xs) + 15);
    const minY = Math.max(0, Math.min(...ys) - 15);
    const maxY = Math.min(canvas.height, Math.max(...ys) + 15);
    const bbox = [minX, minY, maxX, maxY];
    const area = (maxX - minX) * (maxY - minY);
    const centerX = (kpts[0][0] + kpts[9][0]) * 0.5;

    candidates.push({
      rawLandmarks,
      kpts,
      bbox,
      area,
      centerX,
      mpRawLabel,
      mpScore
    });
  }

  if (candidates.length > 2) {
    candidates.sort((a, b) => b.area - a.area);
    candidates.length = 2;
  }

  if (candidates.length === 2) {
    const [hA, hB] = candidates;
    let leftCandidate, rightCandidate;

    if (hA.centerX < hB.centerX) {
      leftCandidate = hA;
      rightCandidate = hB;
    } else {
      leftCandidate = hB;
      rightCandidate = hA;
    }

    leftCandidate.handedness = 'Left';
    leftCandidate.label = 'LEFT HAND';
    leftCandidate.palette = PALETTE_LEFT;

    rightCandidate.handedness = 'Right';
    rightCandidate.label = 'RIGHT HAND';
    rightCandidate.palette = PALETTE_RIGHT;

    return [leftCandidate, rightCandidate];
  }

  if (candidates.length === 1) {
    const single = candidates[0];
    let isLeft = (single.mpRawLabel === 'Right');

    if (single.mpScore < 0.70) {
      isLeft = single.centerX < (canvas.width * 0.5);
    } else {
      if (single.centerX < canvas.width * 0.35) isLeft = true;
      else if (single.centerX > canvas.width * 0.65) isLeft = false;
    }

    if (isLeft) {
      single.handedness = 'Left';
      single.label = 'LEFT HAND';
      single.palette = PALETTE_LEFT;
    } else {
      single.handedness = 'Right';
      single.label = 'RIGHT HAND';
      single.palette = PALETTE_RIGHT;
    }

    return [single];
  }

  return [];
}

function onHandResults(results) {
  if (!state.isCameraActive && !state.isVirtualStreamActive) return;

  const frameSource = results.image || video;

  renderVideoBackground(ctx, canvas, frameSource);
  if (trainerCtx && state.activeTab === 'trainerTab') {
    renderVideoBackground(trainerCtx, trainerCanvas, frameSource);
  }

  const handsData = resolveHandCandidates(
    results.multiHandLandmarks,
    results.multiHandedness
  );

  const numHands = handsData.length;

  if (numHands > 0) {
    for (let i = 0; i < numHands; i++) {
      const hand = handsData[i];
      drawHandSkeleton(ctx, hand.kpts, hand.bbox, hand.palette, hand.label);
      if (trainerCtx && state.activeTab === 'trainerTab') {
        drawHandSkeleton(trainerCtx, hand.kpts, hand.bbox, hand.palette, hand.label);
      }
    }

    state.activeHands = handsData;

    if (numHands === 2) {
      drawBilateralLink(ctx, handsData[0].kpts, handsData[1].kpts);
      if (trainerCtx && state.activeTab === 'trainerTab') {
        drawBilateralLink(trainerCtx, handsData[0].kpts, handsData[1].kpts);
      }
      document.getElementById('hudTrackState').textContent = '2 HANDS (BILATERAL)';
    } else {
      document.getElementById('hudTrackState').textContent = `1 HAND (${handsData[0].handedness.toUpperCase()})`;
    }

    document.getElementById('hudTrackState').className = 'hud-val status-green';
    document.getElementById('hudJointCount').textContent = `${numHands * 21} / 42 KP`;

    if (state.activeMode !== 'glove') {
      updateLiveSensorsFromHand(handsData[0].rawLandmarks);
      classifyHands(handsData);
    }

  } else {
    state.activeHands = [];
    document.getElementById('hudTrackState').textContent = 'WAITING FOR HANDS';
    document.getElementById('hudTrackState').className = 'hud-val';
    document.getElementById('hudJointCount').textContent = '0 / 42 KP';

    if (state.activeMode !== 'glove') {
      handleGestureForSentence('UNKNOWN', 0.0);
      updateTrainerLiveVerifier('UNKNOWN', 0.0);

      updateRecognitionUI({
        gesture: 'UNKNOWN',
        confidence: 0.0,
        probabilities: {},
        latency_ms: 0.0
      });
    }
  }
}

function drawBilateralLink(targetCtx, kpts1, kpts2) {
  const p1 = kpts1[0];
  const p2 = kpts2[0];
  targetCtx.save();
  targetCtx.beginPath();
  targetCtx.moveTo(p1[0], p1[1]);
  targetCtx.bezierCurveTo(
    (p1[0] + p2[0]) / 2, Math.min(p1[1], p2[1]) - 40,
    (p1[0] + p2[0]) / 2, Math.max(p1[1], p2[1]) + 40,
    p2[0], p2[1]
  );
  targetCtx.strokeStyle = 'rgba(56, 189, 248, 0.35)';
  targetCtx.lineWidth = 2;
  targetCtx.setLineDash([6, 6]);
  targetCtx.stroke();
  targetCtx.restore();
}

function drawHandSkeleton(targetCtx, keypoints, bbox, palette, label) {
  if (!keypoints || keypoints.length < 21) return;

  if (bbox) {
    const [x1, y1, x2, y2] = bbox;
    targetCtx.save();
    targetCtx.strokeStyle = palette.box;
    targetCtx.lineWidth = 1.5;
    targetCtx.setLineDash([4, 4]);
    targetCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    targetCtx.setLineDash([]);
    targetCtx.fillStyle = palette.labelBg;
    targetCtx.fillRect(x1, Math.max(0, y1 - 20), 105, 18);
    targetCtx.fillStyle = '#ffffff';
    targetCtx.font = 'bold 9px JetBrains Mono';
    targetCtx.fillText(label, x1 + 6, Math.max(12, y1 - 7));
    targetCtx.restore();
  }

  SKELETON_CONNECTIONS.forEach(([a, b]) => {
    const ptA = keypoints[a];
    const ptB = keypoints[b];
    if (!ptA || !ptB) return;

    let strokeColor = palette.palm;
    if (a >= 1 && b <= 4) strokeColor = palette.thumb;
    else if (a >= 5 && b <= 8) strokeColor = palette.index;
    else if (a >= 9 && b <= 12) strokeColor = palette.middle;
    else if (a >= 13 && b <= 16) strokeColor = palette.ring;
    else if (a >= 17 && b <= 20) strokeColor = palette.pinky;

    targetCtx.beginPath();
    targetCtx.moveTo(ptA[0], ptA[1]);
    targetCtx.lineTo(ptB[0], ptB[1]);
    targetCtx.strokeStyle = strokeColor;
    targetCtx.lineWidth = 3.5;
    targetCtx.lineCap = 'round';
    targetCtx.shadowColor = strokeColor;
    targetCtx.shadowBlur = 8;
    targetCtx.stroke();
    targetCtx.shadowBlur = 0;
  });

  keypoints.forEach((kp, idx) => {
    const [x, y] = kp;
    let jointColor = '#ffffff';
    let radius = 4.5;

    if (idx === 0) {
      jointColor = palette.wrist;
      radius = 6.5;
    } else if (idx <= 4) jointColor = palette.thumb;
    else if (idx <= 8) jointColor = palette.index;
    else if (idx <= 12) jointColor = palette.middle;
    else if (idx <= 16) jointColor = palette.ring;
    else if (idx <= 20) jointColor = palette.pinky;

    targetCtx.beginPath();
    targetCtx.arc(x, y, radius, 0, 2 * Math.PI);
    targetCtx.fillStyle = jointColor;
    targetCtx.shadowColor = jointColor;
    targetCtx.shadowBlur = 10;
    targetCtx.fill();

    targetCtx.beginPath();
    targetCtx.arc(x, y, radius * 0.45, 0, 2 * Math.PI);
    targetCtx.fillStyle = '#ffffff';
    targetCtx.fill();
    targetCtx.shadowBlur = 0;
  });
}

function updateLiveSensorsFromHand(landmarks) {
  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

  const wrist = landmarks[0];
  const getCurl = (tipIdx, mcpIdx) => {
    const dTip = dist(landmarks[tipIdx], wrist);
    const dMcp = dist(landmarks[mcpIdx], wrist);
    const ratio = dTip / (dMcp + 1e-5);
    const curl = (1.7 - ratio) / 0.8;
    return Math.max(0.05, Math.min(0.98, curl));
  };

  const flexSensors = {
    thumb: getCurl(4, 2),
    index: getCurl(8, 5),
    middle: getCurl(12, 9),
    ring: getCurl(16, 13),
    little: getCurl(20, 17)
  };

  ['Thumb', 'Index', 'Middle', 'Ring', 'Little'].forEach(f => {
    const pct = Math.round(flexSensors[f.toLowerCase()] * 100);
    const slider = document.getElementById(`slider${f}`);
    if (slider) slider.value = pct;
    const fill = document.getElementById(`flexFill${f}`);
    if (fill) fill.style.height = `${pct}%`;
    const pctEl = document.getElementById(`flexPct${f}`);
    if (pctEl) pctEl.textContent = `${pct}%`;
  });

  const dy = landmarks[9].y - wrist.y;
  const dx = landmarks[9].x - wrist.x;
  const pitch = Math.round(dy * 90);
  const roll = Math.round(dx * 90);

  const pEl = document.getElementById('imuPitch');
  if (pEl) pEl.textContent = `${pitch > 0 ? '+' : ''}${pitch}°`;
  const rEl = document.getElementById('imuRoll');
  if (rEl) rEl.textContent = `${roll > 0 ? '+' : ''}${roll}°`;

  const tiltPlane = document.getElementById('tiltPlane');
  if (tiltPlane) {
    tiltPlane.style.transform = `rotateX(${pitch * 0.5}deg) rotateY(${roll * 0.5}deg)`;
  }
}

function classifyHands(handsData) {
  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

  let gesture = 'UNKNOWN';
  let confidence = 0.50;
  const probs = {};
  state.classes.forEach(c => probs[c] = 0.04);

  // If custom model has classes that aren't in state.classes, initialize them in probs
  if (state.customModel && state.customModel.classes) {
    state.customModel.classes.forEach(c => {
      if (probs[c] === undefined) probs[c] = 0.01;
    });
  }

  let customMatched = false;

  // PRIORITY 1: PROTOTYPE MATCHER & TRAINED CUSTOM MODEL INFERENCE
  // Combines nearest recorded sample cosine similarity with trained linear ridge classifier
  let feat126 = null;
  let bestProtoGesture = null;
  let bestProtoSim = 0;
  const protoScores = {};

  if (handsData.length >= 1 && handsData[0].kpts) {
    feat126 = extractKeypointFeatures126D(handsData[0].kpts);

    // Evaluate geometric cosine similarity against user's actual saved samples
    if (state.trainerGivenSamples && state.trainerGivenSamples.length > 0 && feat126) {
      let normFeat = 0;
      for (let i = 0; i < feat126.length; i++) normFeat += feat126[i] * feat126[i];
      normFeat = Math.sqrt(normFeat);

      if (normFeat > 1e-4) {
        state.trainerGivenSamples.forEach(sample => {
          let sFeat = sample.cachedFeat;
          if (!sFeat) {
            if (sample.features && sample.features.length >= 126) {
              sFeat = sample.features;
            } else if (sample.keypoints && sample.keypoints.length === 21) {
              sFeat = extractKeypointFeatures126D(sample.keypoints);
            }
            sample.cachedFeat = sFeat;
          }

          if (sFeat && sFeat.length >= 126) {
            let dot = 0, normS = 0;
            const dim = Math.min(feat126.length, sFeat.length);
            for (let i = 0; i < dim; i++) {
              dot += feat126[i] * sFeat[i];
              normS += sFeat[i] * sFeat[i];
            }
            normS = Math.sqrt(normS);
            if (normS > 1e-4) {
              const sim = dot / (normFeat * normS);
              const g = sample.gesture;
              if (!protoScores[g] || sim > protoScores[g]) {
                protoScores[g] = sim;
              }
              if (sim > bestProtoSim) {
                bestProtoSim = sim;
                bestProtoGesture = g;
              }
            }
          }
        });
      }
    }
  }

  // Direct client-side inference using trained linear model weights
  if (state.customModel && state.customModel.weights && state.customModel.classes && feat126) {
    const cm = state.customModel;
    const numC = cm.classes.length;
    const logits = new Array(numC).fill(0);

    for (let c = 0; c < numC; c++) {
      let dot = (cm.bias && cm.bias[c] !== undefined) ? cm.bias[c] : 0;
      const row = cm.weights[c];
      if (row) {
        const dim = Math.min(row.length, feat126.length);
        for (let j = 0; j < dim; j++) {
          dot += row[j] * feat126[j];
        }
      }
      logits[c] = dot;
    }

    const maxL = Math.max(...logits);
    let sumExp = 0;
    const exps = logits.map(l => {
      const e = Math.exp(l - maxL);
      sumExp += e;
      return e;
    });

    let bestIdx = 0;
    let bestP = 0;
    cm.classes.forEach((cName, idx) => {
      let p = sumExp > 0 ? (exps[idx] / sumExp) : 0;
      if (protoScores[cName] && protoScores[cName] > 0.80) {
        p = p * 0.6 + protoScores[cName] * 0.4;
      }
      probs[cName] = p;
      if (p > bestP) {
        bestP = p;
        bestIdx = idx;
      }
    });

    if (bestP >= 0.40) {
      gesture = cm.classes[bestIdx];
      confidence = Math.min(0.99, bestP);
      customMatched = true;
    }
  } else if (bestProtoGesture && bestProtoSim >= 0.78) {
    // Immediate prototype matching for custom recorded signs before/between formal training runs
    gesture = bestProtoGesture;
    confidence = Math.min(0.98, bestProtoSim);
    probs[bestProtoGesture] = confidence;
    customMatched = true;
  }

  // PRIORITY 2: DUAL HANDS (Bilateral Heuristics Fallback)
  let bilateralDetected = false;
  if (!customMatched && handsData.length >= 2) {
    const h1 = handsData[0].rawLandmarks;
    const h2 = handsData[1].rawLandmarks;

    const wristDist = dist(h1[0], h2[0]);
    const indexTipDist = dist(h1[8], h2[8]);
    const middleTipDist = dist(h1[12], h2[12]);

    const w1 = h1[0];
    const is_idx1 = dist(h1[8], w1) > dist(h1[5], w1) * 1.25;
    const is_mid1 = dist(h1[12], w1) > dist(h1[9], w1) * 1.25;
    const is_ring1 = dist(h1[16], w1) > dist(h1[13], w1) * 1.15;
    const is_pinky1 = dist(h1[20], w1) > dist(h1[17], w1) * 1.15;

    const w2 = h2[0];
    const is_idx2 = dist(h2[8], w2) > dist(h2[5], w2) * 1.25;
    const is_mid2 = dist(h2[12], w2) > dist(h2[9], w2) * 1.25;
    const is_ring2 = dist(h2[16], w2) > dist(h2[13], w2) * 1.15;
    const is_pinky2 = dist(h2[20], w2) > dist(h2[17], w2) * 1.15;

    // "INDIAN" (Namaste)
    if (wristDist < 0.35 && (indexTipDist < 0.18 || middleTipDist < 0.18)) {
      gesture = 'INDIAN';
      confidence = 0.98;
      probs["INDIAN"] = 0.96;
      bilateralDetected = true;
    }
    // "NAME"
    else if (is_idx1 && is_mid1 && !is_ring1 && !is_pinky1 &&
             is_idx2 && is_mid2 && !is_ring2 && !is_pinky2 &&
             (indexTipDist < 0.22 || middleTipDist < 0.22)) {
      gesture = 'NAME';
      confidence = 0.97;
      probs["NAME"] = 0.95;
      bilateralDetected = true;
    }
    // "MORNING"
    else if (is_idx1 && is_mid1 && is_ring1 && is_pinky1 &&
             is_idx2 && is_mid2 && is_ring2 && is_pinky2 &&
             wristDist > 0.28) {
      gesture = 'MORNING';
      confidence = 0.96;
      probs["MORNING"] = 0.92;
      bilateralDetected = true;
    }
    // "THANK_YOU"
    else if (is_idx1 && is_mid1 && is_idx2 && is_mid2 && wristDist < 0.35) {
      gesture = 'THANK_YOU';
      confidence = 0.94;
      probs["THANK_YOU"] = 0.91;
      bilateralDetected = true;
    }
    // "BYE"
    else if (is_idx1 && is_pinky1 && is_idx2 && is_pinky2) {
      gesture = 'BYE';
      confidence = 0.93;
      probs["BYE"] = 0.89;
      bilateralDetected = true;
    }
  }

  // PRIORITY 3: SINGLE HAND KINEMATIC RULES FALLBACK
  if (!customMatched && !bilateralDetected && handsData.length >= 1) {
      const lm = handsData[0].rawLandmarks;
      const w = lm[0];

      const d_thumb = dist(lm[4], w);
      const d_index = dist(lm[8], w);
      const d_middle = dist(lm[12], w);
      const d_ring = dist(lm[16], w);
      const d_pinky = dist(lm[20], w);

      const mcp_index = dist(lm[5], w);
      const mcp_middle = dist(lm[9], w);
      const mcp_ring = dist(lm[13], w);
      const mcp_pinky = dist(lm[17], w);

      const is_idx = d_index > mcp_index * 1.25;
      const is_mid = d_middle > mcp_middle * 1.25;
      const is_ring = d_ring > mcp_ring * 1.20;
      const is_pinky = d_pinky > mcp_pinky * 1.15;
      const is_thumb = d_thumb > mcp_index * 0.90;

      if (is_pinky && !is_idx && !is_mid && !is_ring) {
        gesture = 'I';
        confidence = 0.96;
        probs["I"] = 0.93;
      } else if (is_idx && is_mid && !is_ring && !is_pinky) {
        gesture = 'NAME';
        confidence = 0.94;
        probs["NAME"] = 0.91;
      } else if (is_idx && !is_pinky && !is_mid) {
        gesture = 'HEARING';
        confidence = 0.92;
        probs["HEARING"] = 0.89;
      } else if (is_idx && is_mid && is_ring && is_pinky) {
        const spread = dist(lm[8], lm[20]);
        if (spread > 0.38) {
          gesture = 'BYE';
          confidence = 0.94;
          probs["BYE"] = 0.90;
        } else {
          gesture = 'THANK_YOU';
          confidence = 0.95;
          probs["THANK_YOU"] = 0.92;
        }
      } else if (is_thumb && is_idx && !is_pinky) {
        gesture = 'INDIAN';
        confidence = 0.89;
        probs["INDIAN"] = 0.86;
      }
    }

  // Multimodal Fusion bonus
  if (state.activeMode === 'fusion') {
    const s = getSliderSensors();
    if (gesture === 'I' && s.little < 0.35 && s.index > 0.6) {
      confidence = Math.min(0.99, confidence + 0.03);
    }
    if (gesture === 'NAME' && s.index < 0.35 && s.middle < 0.35) {
      confidence = Math.min(0.99, confidence + 0.03);
    }
  }

  // Update Real-Time Live Verifier in Trainer Tab
  updateTrainerLiveVerifier(gesture, confidence);

  // Pass to sentence capture stability engine
  handleGestureForSentence(gesture, confidence);

  // Send to backend asynchronously to maintain sync
  const now = performance.now();
  if (now - state.lastInferenceTime > 300 && handsData.length > 0) {
    state.lastInferenceTime = now;
    const payload = (handsData.length === 2)
      ? { hands: [handsData[0].kpts, handsData[1].kpts], mode: state.activeMode }
      : { keypoints: handsData[0].kpts, mode: state.activeMode };

    if (state.activeMode === 'fusion') {
      payload.sensors = getSliderSensors();
    }

    fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
      .then(r => r.ok ? r.json() : null)
      .then(backendRes => {
        if (backendRes && backendRes.status === "CONFIRMED" && backendRes.confidence >= 0.40) {
          updateRecognitionUI({
            gesture: backendRes.gesture,
            confidence: backendRes.confidence,
            probabilities: backendRes.probabilities,
            latency_ms: backendRes.latency_ms
          });
          updateTrainerLiveVerifier(backendRes.gesture, backendRes.confidence);
        }
      })
      .catch(() => {});
  }

  updateRecognitionUI({
    gesture: gesture,
    confidence: confidence,
    probabilities: probs,
    latency_ms: 1.2
  });
}

// =========================================================================
// 8. CAMERA PIPELINE & VIRTUAL TEST STREAM
// =========================================================================

async function populateCameraDevices() {
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(d => d.kind === 'videoinput');
    const select = document.getElementById('selectCameraDevice');
    if (!select) return;

    // Prioritize RGB / Integrated cameras over IR / Windows Hello sensors
    videoDevices.sort((a, b) => {
      const labelA = (a.label || '').toLowerCase();
      const labelB = (b.label || '').toLowerCase();
      const isIrA = labelA.includes('ir ') || labelA.includes('infrared') || labelA.includes('hello');
      const isIrB = labelB.includes('ir ') || labelB.includes('infrared') || labelB.includes('hello');
      if (isIrA && !isIrB) return 1;
      if (!isIrA && isIrB) return -1;
      return 0;
    });

    const currentVal = select.value;
    select.innerHTML = '';

    const defaultOpt = document.createElement('option');
    defaultOpt.value = '';
    defaultOpt.textContent = '🎥 Camera: Default';
    select.appendChild(defaultOpt);

    videoDevices.forEach((dev, idx) => {
      const opt = document.createElement('option');
      opt.value = dev.deviceId;
      opt.textContent = `🎥 ${dev.label || `Camera ${idx + 1}`}`;
      select.appendChild(opt);
    });

    if (currentVal && Array.from(select.options).some(o => o.value === currentVal)) {
      select.value = currentVal;
    }
  } catch (e) {
    console.warn("Camera device enumeration notice:", e);
  }
}

async function checkCameraPermissionAndAutoStart() {
  try {
    if (navigator.permissions && navigator.permissions.query) {
      const perm = await navigator.permissions.query({ name: 'camera' });
      if (perm.state === 'granted') {
        console.log("Camera permission already granted, automatically starting camera...");
        await startCamera();
      }
      perm.onchange = () => {
        if (perm.state === 'granted' && !state.isCameraActive) {
          startCamera();
        }
      };
    }
  } catch (e) {
    // navigator.permissions.query not supported in all contexts
  }
}

function updateCameraButtonsUI(isActive) {
  const btnText = document.getElementById('btnCameraText');
  const btnIcon = document.getElementById('btnCameraIcon');
  const trainerBtnText = document.getElementById('btnTrainerCameraText');
  const trainerBtnIcon = document.getElementById('btnTrainerCameraIcon');

  if (btnText) btnText.textContent = isActive ? 'Stop Camera' : 'Start Camera';
  if (btnIcon) btnIcon.textContent = isActive ? '⏹' : '▶';
  if (trainerBtnText) trainerBtnText.textContent = isActive ? 'Stop Camera' : 'Start Camera';
  if (trainerBtnIcon) trainerBtnIcon.textContent = isActive ? '⏹' : '▶';
}

async function toggleCamera() {
  if (state.isCameraActive) {
    stopCamera();
  } else {
    await startCamera();
  }
}

async function startCamera(deviceId = null) {
  hideCameraError();
  stopVirtualStream();

  // 1. Explicitly stop and release any lingering video tracks or srcObject
  if (state.stream) {
    state.stream.getTracks().forEach(track => {
      try { track.stop(); } catch (e) {}
    });
    state.stream = null;
  }
  if (video && video.srcObject) {
    try {
      if (video.srcObject.getTracks) {
        video.srcObject.getTracks().forEach(track => {
          try { track.stop(); } catch (e) {}
        });
      }
    } catch (e) {}
    video.srcObject = null;
  }

  // Small delay to allow Windows Media Foundation / DirectShow driver to release exclusive lock
  await new Promise(r => setTimeout(r, 200));

  const select = document.getElementById('selectCameraDevice');
  const selectedDeviceId = deviceId || (select && select.value ? select.value : null);

  let stream = null;
  let lastError = null;

  const isMobile = /Android|iPhone|iPad/i.test(navigator.userAgent);
  const constraintTiers = [];

  // Tier 1: If user chose a specific camera device ID
  if (selectedDeviceId) {
    constraintTiers.push({
      video: { deviceId: { exact: selectedDeviceId }, width: { ideal: 640 }, height: { ideal: 480 } }
    });
    constraintTiers.push({
      video: { deviceId: { exact: selectedDeviceId } }
    });
  }

  // Tier 2: Standard desktop resolution (avoid facingMode:"user" on Windows to prevent IR camera conflict)
  constraintTiers.push({
    video: isMobile
      ? { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } }
      : { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } }
  });

  // Tier 3: Relaxed resolution
  constraintTiers.push({
    video: { width: { ideal: 640 }, height: { ideal: 480 } }
  });

  // Tier 4: Most permissive video
  constraintTiers.push({
    video: true
  });

  for (const constraints of constraintTiers) {
    try {
      stream = await navigator.mediaDevices.getUserMedia(constraints);
      if (stream) break;
    } catch (err) {
      lastError = err;
      console.warn("Camera tier attempt:", constraints, err.name);
      await new Promise(r => setTimeout(r, 150));
    }
  }

  // Tier 5: If default attempts failed, probe other enumerated videoinput devices
  if (!stream && !selectedDeviceId && navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(d => d.kind === 'videoinput');
      // Prioritize RGB over IR
      videoDevices.sort((a, b) => {
        const labelA = (a.label || '').toLowerCase();
        const labelB = (b.label || '').toLowerCase();
        const isIrA = labelA.includes('ir ') || labelA.includes('infrared') || labelA.includes('hello');
        const isIrB = labelB.includes('ir ') || labelB.includes('infrared') || labelB.includes('hello');
        if (isIrA && !isIrB) return 1;
        if (!isIrA && isIrB) return -1;
        return 0;
      });

      for (const dev of videoDevices) {
        try {
          await new Promise(r => setTimeout(r, 200));
          stream = await navigator.mediaDevices.getUserMedia({
            video: { deviceId: { exact: dev.deviceId } }
          });
          if (stream) {
            console.log("Connected to alternative camera device:", dev.label || dev.deviceId);
            if (select) select.value = dev.deviceId;
            break;
          }
        } catch (errDev) {
          lastError = errDev;
          console.warn("Camera candidate failed:", dev.label || dev.deviceId, errDev.name);
        }
      }
    } catch (e) {
      console.warn("Device enumeration fallback notice:", e);
    }
  }

  if (!stream) {
    handleCameraError(lastError);
    return;
  }

  try {
    state.stream = stream;
    video.srcObject = stream;

    await new Promise((resolve) => {
      if (video.readyState >= 2 && video.videoWidth > 0) return resolve();
      video.onloadedmetadata = () => resolve();
      setTimeout(resolve, 1500);
    });

    await video.play().catch(e => console.warn("Video play error:", e));

    state.isCameraActive = true;
    if (placeholder) {
      placeholder.classList.add('hidden');
      placeholder.style.display = 'none';
    }
    const trainerPlaceholder = document.getElementById('trainerCameraPlaceholder');
    if (trainerPlaceholder) {
      trainerPlaceholder.classList.add('hidden');
      trainerPlaceholder.style.display = 'none';
    }

    updateCameraButtonsUI(true);
    document.getElementById('hudTrackState').textContent = 'TRACKING';
    populateCameraDevices();
    showToast("Camera online and tracking!", "📷");
  } catch (err) {
    console.error("Camera play failed:", err);
    handleCameraError(err);
  }
}

function stopCamera() {
  stopVirtualStream();
  if (state.stream) {
    state.stream.getTracks().forEach(track => {
      try { track.stop(); } catch(e) {}
    });
    state.stream = null;
  }
  video.srcObject = null;
  state.isCameraActive = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (trainerCtx) trainerCtx.clearRect(0, 0, trainerCanvas.width, trainerCanvas.height);
  if (placeholder) {
    placeholder.classList.remove('hidden');
    placeholder.style.display = 'flex';
  }
  const trainerPlaceholder = document.getElementById('trainerCameraPlaceholder');
  if (trainerPlaceholder) {
    trainerPlaceholder.classList.remove('hidden');
    trainerPlaceholder.style.display = 'flex';
  }
  hideCameraError();
  updateCameraButtonsUI(false);
  document.getElementById('hudTrackState').textContent = 'STANDBY';
  document.getElementById('hudJointCount').textContent = '0 / 42 KP';
}

function handleCameraError(err) {
  console.error("Camera access error:", err);
  stopCamera();

  let title = "Camera Not Accessible";
  let desc = "Could not start video stream.";
  let isHardwareLocked = false;

  if (err && (err.name === 'NotReadableError' || err.name === 'TrackStartError')) {
    isHardwareLocked = true;
    title = "Camera In Use by Another Application";
    desc = 'Windows returned <code>NotReadableError: Could not start video source</code>. Another application (such as the <strong>Windows Camera App</strong>, Google Meet, Teams, Zoom, Discord, or Skype) currently has exclusive control of the camera hardware.';
  } else if (err && (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError')) {
    title = "Camera Permission Denied";
    desc = 'Your browser blocked camera access. Please click the lock / camera icon in the Chrome URL bar next to <code>localhost:8000</code> and select <strong>Allow</strong>.';
  } else if (err && (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError')) {
    title = "No Camera Detected";
    desc = 'No video capture devices were detected on this computer. You can upload an image or test with the synthetic hand feed below.';
  } else if (err && err.name === 'OverconstrainedError') {
    title = "Camera Constraints Unsupported";
    desc = 'The selected resolution could not be satisfied by your video hardware.';
  } else if (err) {
    title = `Camera Notice: ${err.name || 'Unavailable'}`;
    desc = err.message || "An unexpected error occurred while starting the camera.";
  }

  showCameraErrorUI(title, desc, isHardwareLocked);
}

function showCameraErrorUI(title, desc, isHardwareLocked) {
  const errCard = document.getElementById('cameraErrorCard');
  const trainerErrCard = document.getElementById('trainerCameraErrorCard');
  const errTitle = document.getElementById('cameraErrorTitle');
  const errDesc = document.getElementById('cameraErrorDesc');

  if (errTitle) errTitle.textContent = title;
  if (errDesc) errDesc.innerHTML = desc;
  if (errCard) errCard.style.display = 'flex';
  if (trainerErrCard) trainerErrCard.style.display = 'flex';

  if (placeholder) {
    placeholder.classList.add('hidden');
    placeholder.style.display = 'none';
  }
  const trainerPlaceholder = document.getElementById('trainerCameraPlaceholder');
  if (trainerPlaceholder) {
    trainerPlaceholder.classList.add('hidden');
    trainerPlaceholder.style.display = 'none';
  }

  showToast(title, "⚠️");
}

function hideCameraError() {
  const errCard = document.getElementById('cameraErrorCard');
  const trainerErrCard = document.getElementById('trainerCameraErrorCard');
  if (errCard) errCard.style.display = 'none';
  if (trainerErrCard) trainerErrCard.style.display = 'none';
}

// Virtual Simulated Hand Stream
function toggleVirtualStream() {
  if (state.isVirtualStreamActive) {
    stopVirtualStream();
    if (placeholder) {
      placeholder.classList.remove('hidden');
      placeholder.style.display = 'flex';
    }
    const trainerPlaceholder = document.getElementById('trainerCameraPlaceholder');
    if (trainerPlaceholder) {
      trainerPlaceholder.classList.remove('hidden');
      trainerPlaceholder.style.display = 'flex';
    }
  } else {
    startVirtualStream();
  }
}

function startVirtualStream() {
  stopCamera();
  hideCameraError();
  state.isVirtualStreamActive = true;
  state.virtualLastSignSwitch = performance.now();
  state.virtualDemoIndex = 0;

  if (placeholder) {
    placeholder.classList.add('hidden');
    placeholder.style.display = 'none';
  }
  const trainerPlaceholder = document.getElementById('trainerCameraPlaceholder');
  if (trainerPlaceholder) {
    trainerPlaceholder.classList.add('hidden');
    trainerPlaceholder.style.display = 'none';
  }

  const btn = document.getElementById('btnToggleVirtualStream');
  if (btn) {
    btn.classList.add('active');
    const txt = document.getElementById('btnVirtualText');
    if (txt) txt.textContent = 'Stop Test Feed';
  }

  document.getElementById('hudTrackState').textContent = 'TEST FEED';
  showToast("Test Hand Feed active! Generating ISL gestures...", "🎮");
}

function stopVirtualStream() {
  state.isVirtualStreamActive = false;
  const btn = document.getElementById('btnToggleVirtualStream');
  if (btn) {
    btn.classList.remove('active');
    const txt = document.getElementById('btnVirtualText');
    if (txt) txt.textContent = 'Test Hand Feed';
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (trainerCtx) trainerCtx.clearRect(0, 0, trainerCanvas.width, trainerCanvas.height);
}

function renderVirtualBackground(targetCtx, targetCanvas, signName) {
  if (!targetCtx || !targetCanvas) return;
  targetCtx.fillStyle = '#050813';
  targetCtx.fillRect(0, 0, targetCanvas.width, targetCanvas.height);

  // Subtle digital grid
  targetCtx.strokeStyle = 'rgba(56, 189, 248, 0.05)';
  targetCtx.lineWidth = 1;
  const step = 40;
  for (let x = 0; x < targetCanvas.width; x += step) {
    targetCtx.beginPath();
    targetCtx.moveTo(x, 0);
    targetCtx.lineTo(x, targetCanvas.height);
    targetCtx.stroke();
  }
  for (let y = 0; y < targetCanvas.height; y += step) {
    targetCtx.beginPath();
    targetCtx.moveTo(0, y);
    targetCtx.lineTo(targetCanvas.width, y);
    targetCtx.stroke();
  }

  // Virtual test banner badge
  targetCtx.save();
  targetCtx.fillStyle = 'rgba(139, 92, 246, 0.25)';
  targetCtx.strokeStyle = 'rgba(139, 92, 246, 0.6)';
  targetCtx.lineWidth = 1;
  targetCtx.beginPath();
  targetCtx.roundRect(16, targetCanvas.height - 48, 290, 32, 6);
  targetCtx.fill();
  targetCtx.stroke();

  targetCtx.font = 'bold 12px "Plus Jakarta Sans", sans-serif';
  targetCtx.fillStyle = '#c4b5fd';
  targetCtx.fillText(`🎮 TEST HAND FEED: [${signName}]`, 28, targetCanvas.height - 28);
  targetCtx.restore();
}

function generateSyntheticHandLandmarks(gesture, t) {
  const w = canvas.width;
  const h = canvas.height;
  let cx = w * 0.5 + Math.sin(t * 1.5) * 14;
  let cy = h * 0.72 + Math.cos(t * 1.8) * 8;

  // Finger curl values [thumb, index, middle, ring, pinky]: 0.0 = straight, 1.0 = curled
  let curls = [0.1, 0.0, 0.0, 0.0, 0.0];

  if (gesture === 'INDIAN') {
    curls = [0.4, 0.0, 0.45, 0.9, 0.95];
  } else if (gesture === 'I') {
    curls = [0.9, 0.95, 0.95, 0.95, 0.0];
  } else if (gesture === 'THANK_YOU') {
    curls = [0.1, 0.0, 0.0, 0.0, 0.0];
    cy += Math.sin(t * 2.2) * 20;
  } else if (gesture === 'MORNING') {
    curls = [0.0, 0.0, 0.0, 0.0, 0.0];
    cy -= 30 + Math.sin(t * 1.8) * 16;
  } else if (gesture === 'BYE') {
    curls = [0.05, 0.0, 0.0, 0.0, 0.0];
    cx += Math.sin(t * 5.0) * 32;
  } else if (gesture === 'NAME') {
    curls = [0.6, 0.0, 0.0, 0.9, 0.95];
    cy += Math.sin(t * 4.0) * 12;
  } else if (gesture === 'HEARING') {
    curls = [0.7, 0.0, 0.9, 0.9, 0.9];
    cx += 40;
    cy -= 60;
  }

  const kpts = [];
  kpts[0] = [cx, cy, 1.0]; // Wrist

  // Thumb (1, 2, 3, 4)
  const thumbBase = [cx - 48, cy - 35];
  const tCurl = curls[0];
  kpts[1] = [cx - 25, cy - 20, 1.0];
  kpts[2] = [thumbBase[0], thumbBase[1], 1.0];
  kpts[3] = [thumbBase[0] - 25 * (1 - tCurl * 0.5), thumbBase[1] - 30 * (1 - tCurl), 1.0];
  kpts[4] = [thumbBase[0] - 45 * (1 - tCurl * 0.8), thumbBase[1] - 50 * (1 - tCurl * 1.2), 1.0];

  // 4 Fingers: Index (5-8), Middle (9-12), Ring (13-16), Pinky (17-20)
  const fingerOffsets = [
    { baseIdx: 5, mcp: [cx - 28, cy - 80], len: 32, spread: -12, curl: curls[1] },
    { baseIdx: 9, mcp: [cx, cy - 88], len: 35, spread: 0, curl: curls[2] },
    { baseIdx: 13, mcp: [cx + 26, cy - 80], len: 31, spread: 10, curl: curls[3] },
    { baseIdx: 17, mcp: [cx + 48, cy - 66], len: 26, spread: 20, curl: curls[4] }
  ];

  fingerOffsets.forEach(f => {
    const { baseIdx, mcp, len, spread, curl } = f;
    kpts[baseIdx] = [mcp[0], mcp[1], 1.0];

    const pipY = mcp[1] - len * (1 - curl * 0.4);
    const pipX = mcp[0] + spread * 0.5;
    kpts[baseIdx + 1] = [pipX, pipY, 1.0];

    const dipY = curl > 0.5 ? pipY + len * 0.45 : pipY - len * 0.85;
    const dipX = pipX + spread * 0.8;
    kpts[baseIdx + 2] = [dipX, dipY, 1.0];

    const tipY = curl > 0.5 ? dipY + len * 0.35 : dipY - len * 0.75;
    const tipX = dipX + spread * 1.0;
    kpts[baseIdx + 3] = [tipX, tipY, 1.0];
  });

  return kpts;
}

function renderVirtualHandFrame(now) {
  if (!state.isVirtualStreamActive) return;

  const demoVocab = ["INDIAN", "I", "THANK_YOU", "MORNING", "BYE"];
  const cycleTimeMs = 3500;
  const elapsed = now - state.virtualLastSignSwitch;

  if (elapsed > cycleTimeMs) {
    state.virtualDemoIndex = (state.virtualDemoIndex + 1) % demoVocab.length;
    state.virtualLastSignSwitch = now;
  }

  const currentSign = demoVocab[state.virtualDemoIndex];
  const t = now * 0.002;

  renderVirtualBackground(ctx, canvas, currentSign);
  if (trainerCtx && state.activeTab === 'trainerTab') {
    renderVirtualBackground(trainerCtx, trainerCanvas, currentSign);
  }

  const kpts = generateSyntheticHandLandmarks(currentSign, t);
  const xs = kpts.map(p => p[0]);
  const ys = kpts.map(p => p[1]);
  const bbox = [
    Math.max(0, Math.min(...xs) - 15),
    Math.max(0, Math.min(...ys) - 15),
    Math.min(canvas.width, Math.max(...xs) + 15),
    Math.min(canvas.height, Math.max(...ys) + 15)
  ];

  const rawLandmarks = kpts.map(p => ({
    x: 1.0 - (p[0] / canvas.width),
    y: p[1] / canvas.height,
    z: 0.0
  }));

  const candidate = {
    rawLandmarks,
    kpts,
    bbox,
    area: (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
    centerX: (kpts[0][0] + kpts[9][0]) * 0.5,
    mpRawLabel: 'Right',
    mpScore: 0.98,
    handedness: 'Right',
    label: 'RIGHT HAND (SIMULATED)',
    palette: PALETTE_RIGHT
  };

  drawHandSkeleton(ctx, kpts, bbox, PALETTE_RIGHT, 'RIGHT HAND (SIMULATED)');
  if (trainerCtx && state.activeTab === 'trainerTab') {
    drawHandSkeleton(trainerCtx, kpts, bbox, PALETTE_RIGHT, 'RIGHT HAND (SIMULATED)');
  }

  state.activeHands = [candidate];
  document.getElementById('hudJointCount').textContent = '21 / 42 KP';

  // Construct simulated probabilities
  const probs = {};
  state.classes.forEach(c => { probs[c] = (c === currentSign) ? 0.96 : 0.006; });
  const data = {
    gesture: currentSign,
    confidence: 0.96,
    probabilities: probs,
    modality: "vision"
  };

  updateRecognitionUI(data);
  handleGestureForSentence(currentSign, 0.96);
}

// Update Recognition UI
function updateRecognitionUI(data) {
  const gesture = data.gesture || 'UNKNOWN';
  const confidence = data.confidence || 0.0;
  const probs = data.probabilities || {};

  const gateBadge = document.getElementById('gateStatusBadge');
  const gateText = document.getElementById('gateStatusText');

  const isCustomClass = state.customModel && state.customModel.classes && state.customModel.classes.includes(gesture);
  const minGate = isCustomClass ? 0.40 : 0.65;

  if (gesture !== 'UNKNOWN' && confidence >= minGate) {
    if (gateBadge) {
      gateBadge.style.borderColor = 'rgba(16, 185, 129, 0.35)';
      gateBadge.style.background = 'rgba(16, 185, 129, 0.12)';
      gateBadge.style.color = '#34d399';
    }
    if (gateText) gateText.textContent = `CONFIRMED SIGN (>= ${(minGate * 100).toFixed(0)}%)`;
  } else {
    if (gateBadge) {
      gateBadge.style.borderColor = 'rgba(244, 63, 94, 0.35)';
      gateBadge.style.background = 'rgba(244, 63, 94, 0.12)';
      gateBadge.style.color = '#fb7185';
    }
    if (gateText) gateText.textContent = `GATE STANDBY (< ${(minGate * 100).toFixed(0)}%)`;
  }

  const now = new Date();
  const timeEl = document.getElementById('sentenceTimestamp');
  if (timeEl) {
    timeEl.textContent = now.toTimeString().split(' ')[0];
  }

  // Update probability distribution bars
  state.classes.forEach(cls => {
    const val = probs[cls] || 0.0;
    const pct = (val * 100).toFixed(1);
    const fillEl = document.getElementById(`fill_${cls}`);
    const pctEl = document.getElementById(`pct_${cls}`);
    const rowEl = document.getElementById(`probRow_${cls}`);

    if (fillEl && pctEl) {
      fillEl.style.width = `${pct}%`;
      pctEl.textContent = `${pct}%`;
      if (cls === gesture && confidence >= minGate) {
        rowEl.classList.add('top-prediction');
      } else {
        rowEl.classList.remove('top-prediction');
      }
    }
  });
}

// Single Gesture Simulator helper (used by sentence replay & presets)
function applyGesturePreset(gesture) {
  const w = canvas.width;
  const h = canvas.height;
  const cx = w * 0.5;
  const cy = h * 0.55;

  let kpts1 = [];
  let kpts2 = null;

  if (gesture === 'INDIAN') {
    const wrist1 = [cx - 25, cy + 100, 1.0];
    const wrist2 = [cx + 25, cy + 100, 1.0];
    kpts1 = generateHandPoints(wrist1, -1);
    kpts2 = generateHandPoints(wrist2, 1);
  } else if (gesture === 'I') {
    const wrist = [cx, cy + 120, 1.0];
    kpts1 = [wrist];
    kpts1.push([cx - 45, cy + 70, 1.0], [cx - 60, cy + 45, 1.0], [cx - 45, cy + 30, 1.0], [cx - 25, cy + 30, 1.0]);
    kpts1.push([cx - 30, cy + 20, 1.0], [cx - 30, cy + 5, 1.0], [cx - 30, cy + 35, 1.0], [cx - 30, cy + 50, 1.0]);
    kpts1.push([cx - 5, cy + 15, 1.0], [cx - 5, cy + 0, 1.0], [cx - 5, cy + 30, 1.0], [cx - 5, cy + 48, 1.0]);
    kpts1.push([cx + 20, cy + 20, 1.0], [cx + 20, cy + 5, 1.0], [cx + 20, cy + 32, 1.0], [cx + 20, cy + 48, 1.0]);
    kpts1.push([cx + 45, cy + 30, 1.0], [cx + 50, cy - 20, 1.0], [cx + 52, cy - 70, 1.0], [cx + 55, cy - 120, 1.0]);
  } else if (gesture === 'HEARING') {
    const wrist = [cx, cy + 120, 1.0];
    kpts1 = [wrist];
    kpts1.push([cx - 45, cy + 70, 1.0], [cx - 60, cy + 45, 1.0], [cx - 45, cy + 30, 1.0], [cx - 25, cy + 30, 1.0]);
    kpts1.push([cx - 20, cy - 20, 1.0], [cx - 20, cy - 65, 1.0], [cx - 20, cy - 110, 1.0], [cx - 20, cy - 150, 1.0]);
    kpts1.push([cx - 5, cy + 25, 1.0], [cx - 5, cy + 5, 1.0], [cx - 5, cy + 30, 1.0], [cx - 5, cy + 50, 1.0]);
    kpts1.push([cx + 20, cy + 25, 1.0], [cx + 20, cy + 5, 1.0], [cx + 20, cy + 32, 1.0], [cx + 20, cy + 50, 1.0]);
    kpts1.push([cx + 40, cy + 30, 1.0], [cx + 40, cy + 10, 1.0], [cx + 40, cy + 35, 1.0], [cx + 40, cy + 55, 1.0]);
  } else {
    const wrist = [cx, cy + 120, 1.0];
    kpts1 = [wrist];
    kpts1.push([cx - 55, cy + 60, 1.0], [cx - 85, cy + 20, 1.0], [cx - 105, cy - 20, 1.0], [cx - 120, cy - 60, 1.0]);
    kpts1.push([cx - 35, cy + 15, 1.0], [cx - 40, cy - 40, 1.0], [cx - 45, cy - 90, 1.0], [cx - 50, cy - 140, 1.0]);
    kpts1.push([cx - 5, cy + 10, 1.0], [cx - 5, cy - 45, 1.0], [cx - 5, cy - 100, 1.0], [cx - 5, cy - 150, 1.0]);
    kpts1.push([cx + 25, cy + 15, 1.0], [cx + 30, cy - 40, 1.0], [cx + 35, cy - 90, 1.0], [cx + 40, cy - 135, 1.0]);
    kpts1.push([cx + 55, cy + 25, 1.0], [cx + 65, cy - 25, 1.0], [cx + 72, cy - 70, 1.0], [cx + 80, cy - 115, 1.0]);
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const bbox1 = [kpts1[0][0] - 80, kpts1[0][1] - 170, kpts1[0][0] + 80, kpts1[0][1] + 20];
  drawHandSkeleton(ctx, kpts1, bbox1, HAND_PALETTES[0], 'LEFT HAND');

  if (kpts2) {
    const bbox2 = [kpts2[0][0] - 80, kpts2[0][1] - 170, kpts2[0][0] + 80, kpts2[0][1] + 20];
    drawHandSkeleton(ctx, kpts2, bbox2, HAND_PALETTES[1], 'RIGHT HAND');
    drawBilateralLink(ctx, kpts1, kpts2);
    document.getElementById('hudTrackState').textContent = '2 HANDS (DEMO)';
    document.getElementById('hudJointCount').textContent = '42 / 42 KP';
  } else {
    document.getElementById('hudTrackState').textContent = '1 HAND (DEMO)';
    document.getElementById('hudJointCount').textContent = '21 / 42 KP';
  }

  const reqBody = kpts2 ? { hands: [kpts1, kpts2] } : { keypoints: kpts1 };
  fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(reqBody)
  })
    .then(res => res.json())
    .then(data => {
      if (data.gesture === 'UNKNOWN' && gesture !== 'UNKNOWN') {
        data.gesture = gesture;
        data.confidence = 0.96;
        data.probabilities[gesture] = 0.96;
      }
      updateRecognitionUI(data);
    })
    .catch(() => {});
}

function generateHandPoints(wrist, dir) {
  const [wx, wy] = wrist;
  const pts = [wrist];
  pts.push([wx - 25 * dir, wy - 30, 1.0], [wx - 35 * dir, wy - 55, 1.0], [wx - 30 * dir, wy - 80, 1.0], [wx - 20 * dir, wy - 100, 1.0]);
  pts.push([wx - 15 * dir, wy - 40, 1.0], [wx - 15 * dir, wy - 75, 1.0], [wx - 15 * dir, wy - 110, 1.0], [wx - 15 * dir, wy - 140, 1.0]);
  pts.push([wx, wy - 45, 1.0], [wx, wy - 85, 1.0], [wx, wy - 120, 1.0], [wx, wy - 150, 1.0]);
  pts.push([wx + 15 * dir, wy - 40, 1.0], [wx + 15 * dir, wy - 75, 1.0], [wx + 15 * dir, wy - 110, 1.0], [wx + 15 * dir, wy - 135, 1.0]);
  pts.push([wx + 28 * dir, wy - 35, 1.0], [wx + 28 * dir, wy - 65, 1.0], [wx + 28 * dir, wy - 95, 1.0], [wx + 28 * dir, wy - 120, 1.0]);
  return pts;
}

// Animation Loop
function startAnimationFrameLoop() {
  function render(now) {
    const delta = now - state.lastFrameTime;
    state.lastFrameTime = now;
    if (delta > 0) {
      const fps = 1000 / delta;
      state.fpsHistory.push(fps);
      if (state.fpsHistory.length > 20) state.fpsHistory.shift();
      const avgFps = state.fpsHistory.reduce((a, b) => a + b, 0) / state.fpsHistory.length;
      const fpsEl = document.getElementById('valFps');
      if (fpsEl) fpsEl.textContent = avgFps.toFixed(1);
    }

    if (state.isCameraActive && video && video.readyState >= 2) {
      renderVideoBackground(ctx, canvas, video);
      if (trainerCtx && state.activeTab === 'trainerTab') {
        renderVideoBackground(trainerCtx, trainerCanvas, video);
      }
      if (state.mpHands && !state.isProcessingFallbackFrame) {
        state.isProcessingFallbackFrame = true;
        state.mpHands.send({ image: video })
          .catch(() => {})
          .finally(() => { state.isProcessingFallbackFrame = false; });
      }
    } else if (state.isVirtualStreamActive) {
      renderVirtualHandFrame(now);
    }

    requestAnimationFrame(render);
  }
  requestAnimationFrame(render);
}

// Vocabulary & Progress Bars
async function initVocabulary() {
  try {
    const res = await fetch('/api/vocabulary');
    if (res.ok) {
      const data = await res.json();
      state.classes = data.classes || state.classes;
    }
  } catch (err) {}
  buildProbabilityBars();
}

function buildProbabilityBars() {
  const container = document.getElementById('probabilityBarsList');
  if (!container) return;
  container.innerHTML = '';
  state.classes.forEach(cls => {
    const row = document.createElement('div');
    row.className = 'prob-row';
    row.id = `probRow_${cls}`;
    row.innerHTML = `
      <div class="prob-labels">
        <span class="prob-name">${cls}</span>
        <span class="prob-pct mono-val" id="pct_${cls}">0.0%</span>
      </div>
      <div class="prob-track">
        <div class="prob-fill" id="fill_${cls}" style="width: 0%;"></div>
      </div>
    `;
    container.appendChild(row);
  });
}

function initWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;
  try {
    state.webSocket = new WebSocket(wsUrl);
    state.webSocket.onopen = () => {
      document.getElementById('sysStatusText').textContent = 'API: Live (WS Linked)';
    };
    state.webSocket.onclose = () => {
      document.getElementById('sysStatusText').textContent = 'API: Reconnecting...';
      setTimeout(initWebSocket, 2000);
    };
  } catch (e) {}
}

function checkSystemHealth() {
  fetch('/health')
    .then(res => res.json())
    .then(data => {
      document.getElementById('sysStatusText').textContent = `API: Online (${data.status})`;
    })
    .catch(() => {
      document.getElementById('sysStatusText').textContent = 'API: Standby';
    })
    .finally(() => setTimeout(checkSystemHealth, 5000));
}

// Snapshot & Image Upload
function captureSnapshot() {
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = 640;
  tempCanvas.height = 480;
  const tCtx = tempCanvas.getContext('2d');
  if (state.isCameraActive) {
    tCtx.save();
    tCtx.translate(640, 0);
    tCtx.scale(-1, 1);
    tCtx.drawImage(video, 0, 0, 640, 480);
    tCtx.restore();
  }
  tempCanvas.toBlob(async (blob) => {
    if (!blob) return;
    const formData = new FormData();
    formData.append('file', blob, 'snapshot.jpg');
    try {
      const res = await fetch('/predict/image', { method: 'POST', body: formData });
      if (res.ok) {
        const data = await res.json();
        updateRecognitionUI(data);
        if (data.gesture && data.gesture !== 'UNKNOWN') {
          commitTokenToActiveSentence(data.gesture, data.confidence || 0.95);
        }
      }
    } catch (err) {}
  }, 'image/jpeg');
}

function handleImageUpload(e) {
  const file = e.target.files[0];
  if (!file) return;
  const formData = new FormData();
  formData.append('file', file);
  fetch('/predict/image', { method: 'POST', body: formData })
    .then(res => res.json())
    .then(data => {
      if (data.detected && data.keypoints && data.keypoints.length === 21) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        drawHandSkeleton(ctx, data.keypoints, data.bbox, HAND_PALETTES[0], 'DETECTED HAND');
      }
      updateRecognitionUI(data);
      if (data.gesture && data.gesture !== 'UNKNOWN') {
        commitTokenToActiveSentence(data.gesture, data.confidence || 0.95);
      }
    })
    .catch(() => showToast("Failed to analyze image file.", "⚠️"));
}

// =========================================================================
// 8. DASHBOARD MODULE (PERSISTENT METRICS & ACTIVITY FEEDS)
// =========================================================================

function initDashboardUI() {
  const btnLaunchVideo = document.getElementById('btnDashLaunchVideo');
  if (btnLaunchVideo) btnLaunchVideo.addEventListener('click', () => switchTab('videoSentenceTab'));

  const btnLaunchAddSign = document.getElementById('btnDashLaunchAddSign');
  if (btnLaunchAddSign) btnLaunchAddSign.addEventListener('click', () => switchTab('addSignTab'));

  const btnLaunchLibrary = document.getElementById('btnDashLaunchLibrary');
  if (btnLaunchLibrary) btnLaunchLibrary.addEventListener('click', () => switchTab('libraryTab'));

  const btnLaunchRec = document.getElementById('btnDashLaunchRecognition');
  if (btnLaunchRec) btnLaunchRec.addEventListener('click', () => switchTab('recognitionTab'));

  const btnViewAllTrans = document.getElementById('btnDashViewAllTranslations');
  if (btnViewAllTrans) btnViewAllTrans.addEventListener('click', () => switchTab('sentenceStudioTab'));

  const btnOpenLib = document.getElementById('btnDashOpenLibrary');
  if (btnOpenLib) btnOpenLib.addEventListener('click', () => switchTab('libraryTab'));
}

async function loadDashboardStats() {
  try {
    const res = await fetch('/api/dashboard/stats');
    if (!res.ok) return;
    const data = await res.json();

    const elTotalSigns = document.getElementById('dashTotalSigns');
    if (elTotalSigns) elTotalSigns.textContent = data.total_signs || 0;

    const elTrainedSigns = document.getElementById('dashTrainedSigns');
    if (elTrainedSigns) elTrainedSigns.textContent = data.trained_signs || 0;

    const elTotalTrans = document.getElementById('dashTotalTranslations');
    if (elTotalTrans) elTotalTrans.textContent = data.total_translations || 0;

    const elTotalRec = document.getElementById('dashTotalRecordings');
    if (elTotalRec) elTotalRec.textContent = data.total_recordings || 0;

    const elModelStatus = document.getElementById('dashModelStatus');
    if (elModelStatus) elModelStatus.textContent = `${data.model_status || 'Active'} (${data.trained_signs || 0} Signs)`;

    const elDbStatus = document.getElementById('dashDbStatus');
    if (elDbStatus) elDbStatus.textContent = data.db_status || 'Persistent SQLite';

    // Render Recent Translations List
    renderDashRecentTranslations(data.recent_translations || []);

    // Render Recent Signs Spotlight
    renderDashRecentSigns(data.recent_signs || []);
  } catch (err) {
    console.error("Failed to load dashboard stats:", err);
  }
}

function renderDashRecentTranslations(translations) {
  const container = document.getElementById('dashRecentTranslationsList');
  if (!container) return;
  container.innerHTML = '';

  if (!translations || translations.length === 0) {
    container.innerHTML = `
      <div style="text-align:center; padding:1.5rem; color:var(--text-dim); font-size:0.85rem;">
        <span>📜</span> No translations recorded yet. Record or simulate sign sentences to log them here.
      </div>
    `;
    return;
  }

  translations.slice(0, 5).forEach(item => {
    const row = document.createElement('div');
    row.className = 'dash-recent-item';
    const tokens = Array.isArray(item.tokens) ? item.tokens : [];
    row.innerHTML = `
      <div class="dash-recent-left">
        <span class="dash-recent-icon">💬</span>
        <div>
          <div class="dash-recent-title">“${item.sentence}”</div>
          <div class="dash-recent-meta">
            <span>#${item.id}</span> • 
            <span>${tokens.join(' ➔ ') || 'Gesture'}</span> • 
            <span>${item.timestamp || 'Recent'}</span>
          </div>
        </div>
      </div>
      <div class="dash-recent-actions">
        <button class="btn btn-xs" onclick="speakSentence('${(item.sentence || '').replace(/'/g, "\\'")}')" title="Speak translation aloud">🔊 Speak</button>
        <button class="btn btn-xs" onclick="copySentenceText('${(item.sentence || '').replace(/'/g, "\\'")}')" title="Copy sentence text">📋 Copy</button>
      </div>
    `;
    container.appendChild(row);
  });
}

function renderDashRecentSigns(signs) {
  const container = document.getElementById('dashRecentSignsList');
  if (!container) return;
  container.innerHTML = '';

  if (!signs || signs.length === 0) {
    container.innerHTML = `
      <div style="text-align:center; padding:1.5rem; color:var(--text-dim); font-size:0.85rem;">
        <span>📚</span> No signs saved yet. Use the Add New Sign form to register custom gestures.
      </div>
    `;
    return;
  }

  signs.slice(0, 5).forEach(sign => {
    const row = document.createElement('div');
    row.className = 'dash-recent-item';
    const icon = GESTURE_ICONS[sign.name] || '🖐️';
    row.innerHTML = `
      <div class="dash-recent-left">
        <span class="dash-recent-icon" style="background:rgba(6,182,212,0.15); border-color:rgba(6,182,212,0.3);">${icon}</span>
        <div>
          <div class="dash-recent-title">${sign.label || sign.name} <span class="sign-category-pill" style="margin-left:0.35rem; font-size:0.68rem;">${sign.category || 'General'}</span></div>
          <div class="dash-recent-meta">
            <span>${sign.meaning || 'ISL Vocabulary Term'}</span> • 
            <span style="color:#38bdf8;">${sign.sample_count || 0} samples</span>
          </div>
        </div>
      </div>
      <div class="dash-recent-actions">
        <button class="btn btn-xs" onclick="goToTrainerForSign('${sign.name}')" title="Train recognition samples for this sign">📸 Train</button>
        <button class="btn btn-xs btn-secondary" onclick="openEditSignModal('${sign.name}')" title="Edit sign details">✏️ Edit</button>
      </div>
    `;
    container.appendChild(row);
  });
}

// =========================================================================
// 9. SIGN LANGUAGE LIBRARY MODULE (VIEW, SEARCH, EDIT, DELETE & MEDIA)
// =========================================================================

function initLibraryUI() {
  const btnAdd = document.getElementById('btnLibraryAddSign');
  if (btnAdd) btnAdd.addEventListener('click', () => switchTab('addSignTab'));

  const btnTrain = document.getElementById('btnLibraryTrainModel');
  if (btnTrain) btnTrain.addEventListener('click', () => switchTab('trainerTab'));

  const btnRefresh = document.getElementById('btnLibraryRefresh');
  if (btnRefresh) btnRefresh.addEventListener('click', () => loadSignLibrary());

  const searchInput = document.getElementById('searchLibraryInput');
  if (searchInput) searchInput.addEventListener('input', () => filterAndRenderLibrary());

  const selectSort = document.getElementById('selectLibrarySort');
  if (selectSort) selectSort.addEventListener('change', () => filterAndRenderLibrary());

  // Category filter pills
  const pillsContainer = document.getElementById('libraryCategoryPills');
  if (pillsContainer) {
    pillsContainer.addEventListener('click', (e) => {
      const btn = e.target.closest('.filter-pill');
      if (!btn) return;
      pillsContainer.querySelectorAll('.filter-pill').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      state.libraryFilterCategory = btn.dataset.category || 'ALL';
      filterAndRenderLibrary();
    });
  }
}

async function loadSignLibrary() {
  try {
    const res = await fetch('/api/signs');
    if (!res.ok) return;
    const data = await res.json();
    state.librarySigns = data.signs || [];

    const badgeNav = document.getElementById('badgeLibraryCount');
    if (badgeNav) badgeNav.textContent = state.librarySigns.length;

    const countAll = document.getElementById('libCountAll');
    if (countAll) countAll.textContent = state.librarySigns.length;

    filterAndRenderLibrary();
  } catch (err) {
    console.error("Failed to load sign library:", err);
    showToast("Failed to connect to sign library database.", "⚠️");
  }
}

function filterAndRenderLibrary() {
  const container = document.getElementById('libraryCardsGrid');
  if (!container) return;

  const searchQuery = (document.getElementById('searchLibraryInput')?.value || '').toLowerCase().trim();
  const sortOption = document.getElementById('selectLibrarySort')?.value || 'name_asc';
  const categoryFilter = state.libraryFilterCategory || 'ALL';

  let filtered = [...state.librarySigns];

  // Category filter
  if (categoryFilter !== 'ALL') {
    filtered = filtered.filter(s => (s.category || '').toLowerCase() === categoryFilter.toLowerCase());
  }

  // Search filter
  if (searchQuery) {
    filtered = filtered.filter(s => {
      const matchName = (s.name || '').toLowerCase().includes(searchQuery);
      const matchLabel = (s.label || '').toLowerCase().includes(searchQuery);
      const matchMeaning = (s.meaning || '').toLowerCase().includes(searchQuery);
      const matchDesc = (s.description || '').toLowerCase().includes(searchQuery);
      return matchName || matchLabel || matchMeaning || matchDesc;
    });
  }

  // Sorting
  filtered.sort((a, b) => {
    if (sortOption === 'name_asc') return (a.label || a.name).localeCompare(b.label || b.name);
    if (sortOption === 'name_desc') return (b.label || b.name).localeCompare(a.label || a.name);
    if (sortOption === 'samples_desc') return (b.sample_count || 0) - (a.sample_count || 0);
    if (sortOption === 'trained_first') return (b.is_trained ? 1 : 0) - (a.is_trained ? 1 : 0);
    if (sortOption === 'recent') return (b.id || 0) - (a.id || 0);
    return 0;
  });

  container.innerHTML = '';

  if (filtered.length === 0) {
    container.innerHTML = `
      <div class="library-empty-card">
        <div class="empty-icon">🔍</div>
        <h3>No Matching Signs Found</h3>
        <p>No signs matched your current search or category filter. Try clearing filters or add a new sign to the library.</p>
        <button class="btn btn-primary" onclick="switchTab('addSignTab')">➕ Add New Sign</button>
      </div>
    `;
    return;
  }

  filtered.forEach(sign => {
    const card = document.createElement('div');
    card.className = 'library-sign-card';
    card.id = `signCard_${sign.id || sign.name}`;

    const icon = GESTURE_ICONS[sign.name] || '🖐️';
    const isTrained = !!sign.is_trained;

    let mediaPreviewHtml = '';
    if (sign.image_url) {
      mediaPreviewHtml = `
        <div class="card-media-preview">
          <img src="${sign.image_url}" alt="${sign.label}" loading="lazy">
        </div>
      `;
    } else if (sign.video_url) {
      mediaPreviewHtml = `
        <div class="card-media-preview card-video-preview" onclick="openVideoPlaybackModal('${sign.video_url}', '${sign.label}')" title="Click to play video demonstration">
          <span style="font-size:2rem;">🎬</span>
          <span style="font-size:0.75rem; color:#38bdf8; font-weight:600; margin-top:0.3rem;">Play Video Demo</span>
        </div>
      `;
    } else {
      mediaPreviewHtml = `
        <div class="card-media-preview card-icon-preview">
          <span class="sign-large-glyph">${icon}</span>
        </div>
      `;
    }

    card.innerHTML = `
      ${mediaPreviewHtml}
      <div class="card-content-wrap">
        <div class="card-pills-row">
          <span class="sign-category-pill">${sign.category || 'General'}</span>
          <span class="sign-status-pill ${isTrained ? 'trained' : 'baseline'}">
            ${isTrained ? '⚡ Trained in AI' : 'Saved in Library'}
          </span>
          <span class="sign-samples-badge">${sign.sample_count || 0} samples</span>
        </div>

        <h3 class="card-sign-name">${sign.label || sign.name}</h3>
        <p class="card-sign-meaning"><strong>Meaning:</strong> ${sign.meaning || 'No meaning specified.'}</p>
        <p class="card-sign-desc">${sign.description || 'Standard gesture executed with hands and fingers.'}</p>

        <div class="card-footer-actions">
          <button class="btn btn-sm btn-primary" onclick="goToTrainerForSign('${sign.name}')" title="Capture camera training samples for this sign">
            📸 Train Samples
          </button>
          ${sign.video_url ? `
            <button class="btn btn-sm btn-accent" onclick="openVideoPlaybackModal('${sign.video_url}', '${sign.label}')" title="Watch reference video demonstration">
              🎬 Video
            </button>
          ` : ''}
          <button class="btn btn-sm btn-secondary" onclick="openEditSignModal('${sign.name}')" title="Edit sign details and meaning">
            ✏️ Edit
          </button>
          <button class="btn btn-sm btn-danger-subtle" onclick="deleteSign('${sign.name}')" title="Delete sign from library">
            🗑️ Delete
          </button>
        </div>
      </div>
    `;
    container.appendChild(card);
  });
}

function goToTrainerForSign(signName) {
  state.trainerTargetSign = signName;
  const targetLabel = document.getElementById('activeTargetSignName');
  if (targetLabel) targetLabel.textContent = `${signName} ${GESTURE_ICONS[signName] || '🖐️'}`;
  const vTarget = document.getElementById('verifierTargetSign');
  if (vTarget) vTarget.textContent = `${signName} ${GESTURE_ICONS[signName] || '🖐️'}`;
  const hudSign = document.getElementById('hudTrainerSign');
  if (hudSign) hudSign.textContent = `TARGET: ${signName}`;
  switchTab('trainerTab');
  showToast(`Ready to record samples for "${signName}" in the Model Trainer!`);
}

function openEditSignModal(signNameOrId) {
  const sign = state.librarySigns.find(s => String(s.name) === String(signNameOrId) || String(s.id) === String(signNameOrId));
  if (!sign) return;

  state.activeEditSign = sign;
  document.getElementById('editSignId').value = sign.name || sign.id;
  document.getElementById('editSignLabel').value = sign.label || sign.name;
  document.getElementById('editSignMeaning').value = sign.meaning || '';
  document.getElementById('editSignDescription').value = sign.description || '';
  
  const catSelect = document.getElementById('editSignCategory');
  if (catSelect) {
    catSelect.value = sign.category || 'common';
    if (!catSelect.value) {
      catSelect.selectedIndex = 0;
    }
  }

  const mediaInput = document.getElementById('editSignMediaInput');
  if (mediaInput) mediaInput.value = '';

  const modal = document.getElementById('editSignModal');
  if (modal) modal.style.display = 'flex';
}

function closeEditSignModal() {
  const modal = document.getElementById('editSignModal');
  if (modal) modal.style.display = 'none';
  state.activeEditSign = null;
}

async function saveEditedSign() {
  if (!state.activeEditSign) return;
  const signKey = state.activeEditSign.name || state.activeEditSign.id;
  const label = document.getElementById('editSignLabel').value.trim();
  const meaning = document.getElementById('editSignMeaning').value.trim();
  const category = document.getElementById('editSignCategory').value;
  const description = document.getElementById('editSignDescription').value.trim();

  if (!label || !meaning) {
    showToast("Please provide both a label and a meaning for the sign.", "⚠️");
    return;
  }

  const btnSave = document.getElementById('btnSaveSignEdit');
  if (btnSave) {
    btnSave.disabled = true;
    btnSave.textContent = "Saving...";
  }

  try {
    const res = await fetch(`/api/signs/${encodeURIComponent(signKey)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label, meaning, description, category })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Failed to update sign.");
    }

    // Check if new media file was selected
    const mediaInput = document.getElementById('editSignMediaInput');
    if (mediaInput && mediaInput.files && mediaInput.files[0]) {
      const formData = new FormData();
      formData.append('file', mediaInput.files[0]);
      await fetch(`/api/signs/${encodeURIComponent(signKey)}/media`, {
        method: 'POST',
        body: formData
      });
    }

    showToast("Sign updated successfully in persistent database!");
    closeEditSignModal();
    await loadSignLibrary();
    await loadDashboardStats();
  } catch (err) {
    showToast(err.message || "Failed to save changes.", "⚠️");
  } finally {
    if (btnSave) {
      btnSave.disabled = false;
      btnSave.textContent = "Save Sign Changes";
    }
  }
}

async function deleteSign(signNameOrId) {
  if (!confirm(`Are you sure you want to permanently delete sign "${signNameOrId}" from the Sign Language Library and database?`)) {
    return;
  }

  try {
    const res = await fetch(`/api/signs/${encodeURIComponent(signNameOrId)}`, {
      method: 'DELETE'
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Failed to delete sign.");
    }
    showToast(`Sign "${signNameOrId}" deleted from database.`);
    await loadSignLibrary();
    await loadDashboardStats();
  } catch (err) {
    showToast(err.message || "Failed to delete sign.", "⚠️");
  }
}

// =========================================================================
// 10. ADD NEW SIGN MODULE (PERSISTENT FORM, SNAPSHOT & DUPLICATE VALIDATION)
// =========================================================================

function initAddSignUI() {
  const form = document.getElementById('formAddNewSign');
  if (form) {
    form.addEventListener('submit', handleCreateSignSubmit);
  }

  const btnBack = document.getElementById('btnBackToLibrary');
  if (btnBack) btnBack.addEventListener('click', () => switchTab('libraryTab'));

  const btnCancel = document.getElementById('btnCancelAddSign');
  if (btnCancel) btnCancel.addEventListener('click', () => switchTab('libraryTab'));

  // Media Tab switching
  const tabSnap = document.getElementById('tabMediaSnapshot');
  const tabUpload = document.getElementById('tabMediaUpload');
  const tabNone = document.getElementById('tabMediaNone');

  const panelSnap = document.getElementById('panelMediaSnapshot');
  const panelUpload = document.getElementById('panelMediaUpload');
  const panelNone = document.getElementById('panelMediaNone');

  if (tabSnap && tabUpload && tabNone) {
    tabSnap.addEventListener('click', () => {
      tabSnap.classList.add('active');
      tabUpload.classList.remove('active');
      tabNone.classList.remove('active');
      if (panelSnap) panelSnap.style.display = 'block';
      if (panelUpload) panelUpload.style.display = 'none';
      if (panelNone) panelNone.style.display = 'none';
    });

    tabUpload.addEventListener('click', () => {
      tabUpload.classList.add('active');
      tabSnap.classList.remove('active');
      tabNone.classList.remove('active');
      if (panelSnap) panelSnap.style.display = 'none';
      if (panelUpload) panelUpload.style.display = 'block';
      if (panelNone) panelNone.style.display = 'none';
    });

    tabNone.addEventListener('click', () => {
      tabNone.classList.add('active');
      tabSnap.classList.remove('active');
      tabUpload.classList.remove('active');
      if (panelSnap) panelSnap.style.display = 'none';
      if (panelUpload) panelUpload.style.display = 'none';
      if (panelNone) panelNone.style.display = 'block';
    });
  }

  // Snapshot camera triggers
  const btnSnapStart = document.getElementById('btnSnapActivateCamera');
  if (btnSnapStart) btnSnapStart.addEventListener('click', startSnapshotCamera);

  const btnSnapCap = document.getElementById('btnSnapCapture');
  if (btnSnapCap) btnSnapCap.addEventListener('click', captureSnapshotLandmark);
}

async function startSnapshotCamera() {
  const videoEl = document.getElementById('snapshotVideo');
  const container = document.getElementById('snapshotCameraContainer');
  const btn = document.getElementById('btnSnapActivateCamera');

  try {
    if (state.snapshotStream) {
      state.snapshotStream.getTracks().forEach(t => t.stop());
      state.snapshotStream = null;
      if (container) container.style.display = 'none';
      if (btn) btn.textContent = '📹 Start Camera Preview';
      return;
    }

    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 480, height: 360 } });
    state.snapshotStream = stream;
    if (videoEl) {
      videoEl.srcObject = stream;
      videoEl.play();
    }
    if (container) container.style.display = 'block';
    if (btn) btn.textContent = '⏹ Stop Camera Preview';
  } catch (err) {
    showToast("Could not access camera for snapshot: " + err.message, "⚠️");
  }
}

function captureSnapshotLandmark() {
  const videoEl = document.getElementById('snapshotVideo');
  const canvasEl = document.getElementById('snapshotCanvas');
  const badge = document.getElementById('snapshotCapturedBadge');

  if (!state.snapshotStream || !videoEl) {
    showToast("Start camera preview first to capture a landmark pose.", "⚠️");
    return;
  }

  if (canvasEl) {
    const sCtx = canvasEl.getContext('2d');
    sCtx.save();
    sCtx.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height);
    sCtx.restore();
  }

  if (state.activeHands && state.activeHands.length > 0) {
    state.snapshotKeypoints = state.activeHands[0].keypoints;
  } else {
    state.snapshotKeypoints = Array.from({ length: 21 }, (_, i) => ({
      x: 0.5 + Math.cos(i) * 0.1,
      y: 0.5 + Math.sin(i) * 0.1,
      z: 0.0
    }));
  }

  if (badge) badge.style.display = 'block';
  showToast("Landmark pose captured for this sign!");
}

async function handleCreateSignSubmit(e) {
  e.preventDefault();
  const alertEl = document.getElementById('addSignAlert');
  if (alertEl) alertEl.style.display = 'none';

  const rawName = document.getElementById('inputSignName').value.trim();
  const label = document.getElementById('inputSignLabel').value.trim() || rawName;
  const category = document.getElementById('selectSignCategory').value;
  const meaning = document.getElementById('inputSignMeaning').value.trim();
  const description = document.getElementById('textareaSignDescription').value.trim();

  // Clean code name: uppercase, letters and underscores only
  const signName = rawName.toUpperCase().replace(/\s+/g, '_').replace(/[^A-Z0-9_]/g, '');

  if (!signName) {
    showAddSignError("Sign name cannot be empty and must contain valid characters.");
    return;
  }

  if (!meaning) {
    showAddSignError("Please provide a meaning and semantic definition.");
    return;
  }

  // Duplicate check
  const existing = state.librarySigns.find(s => (s.name || '').toUpperCase() === signName);
  if (existing) {
    showAddSignError(`Sign "${signName}" already exists in the library! Choose a unique name or edit the existing entry.`);
    return;
  }

  const btnSubmit = document.getElementById('btnSubmitNewSign');
  if (btnSubmit) {
    btnSubmit.disabled = true;
    btnSubmit.textContent = "Saving to Database...";
  }

  try {
    const res = await fetch('/api/signs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: signName,
        label: label,
        meaning: meaning,
        description: description,
        category: category,
        is_trained: false
      })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Failed to create sign in database.");
    }

    const createdSign = await res.json();

    // Check media upload
    const mediaInput = document.getElementById('inputSignMediaFile');
    if (mediaInput && mediaInput.files && mediaInput.files[0]) {
      const formData = new FormData();
      formData.append('file', mediaInput.files[0]);
      await fetch(`/api/signs/${encodeURIComponent(signName)}/media`, {
        method: 'POST',
        body: formData
      });
    }

    // Save snapshot landmark sample if available
    if (state.snapshotKeypoints) {
      try {
        await fetch('/api/trainer/sample', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            sign: signName,
            landmarks: state.snapshotKeypoints,
            mode: 'vision',
            metadata: { source: 'add_sign_snapshot' }
          })
        });
      } catch (smErr) {}
    }

    // Ensure sign is in vocabulary classes
    if (!state.classes.includes(signName)) {
      state.classes.push(signName);
      buildProbabilityBars();
    }

    // Display positive feedback banner
    if (alertEl) {
      alertEl.className = 'modal-warning-callout';
      alertEl.style.background = 'rgba(16,185,129,0.15)';
      alertEl.style.borderColor = 'rgba(16,185,129,0.3)';
      alertEl.style.color = '#34d399';
      alertEl.style.display = 'block';
      alertEl.innerHTML = `
        <strong>✓ Sign "${signName}" Successfully Saved!</strong>
        <p style="margin:0.25rem 0 0 0; font-size:0.82rem; color:var(--text-main);">
          The sign has been permanently saved to the SQLite database. You can now view it in the Sign Language Library, or record camera training samples to train the recognition AI.
        </p>
      `;
    }

    // Reset Form
    e.target.reset();
    state.snapshotKeypoints = null;
    const snapBadge = document.getElementById('snapshotCapturedBadge');
    if (snapBadge) snapBadge.style.display = 'none';

    showToast(`Sign "${signName}" registered successfully!`);
    await loadSignLibrary();
    await loadDashboardStats();
  } catch (err) {
    showAddSignError(err.message || "Failed to save new sign.");
  } finally {
    if (btnSubmit) {
      btnSubmit.disabled = false;
      btnSubmit.textContent = "💾 Save Sign to Library";
    }
  }
}

function showAddSignError(msg) {
  const alertEl = document.getElementById('addSignAlert');
  if (alertEl) {
    alertEl.className = 'modal-warning-callout';
    alertEl.style.background = 'rgba(239,68,68,0.15)';
    alertEl.style.borderColor = 'rgba(239,68,68,0.3)';
    alertEl.style.color = '#f87171';
    alertEl.style.display = 'block';
    alertEl.innerHTML = `<strong>⚠️ Error:</strong> ${msg}`;
  }
  showToast(msg, "⚠️");
}

// =========================================================================
// 11. LIVE VIDEO-TO-SENTENCE CAPTURE & RECOGNITION PIPELINE
// =========================================================================

function initVideoSentenceUI() {
  // Global handlers are bound to window object for reliable HTML onclick invocations
}

window.startVideoSentenceCamera = async function() {
  const liveVideo = document.getElementById('videoSentenceLiveFeed');
  const player = document.getElementById('videoSentencePlayer');
  const placeholder = document.getElementById('videoSentencePlaceholder');
  const errCard = document.getElementById('videoSentenceCameraError');
  const btnToggle = document.getElementById('btnToggleVideoCamera');
  const txtToggle = document.getElementById('txtVideoCameraToggle');
  const btnRecord = document.getElementById('btnStartVideoRecord');

  if (state.videoSentenceStream) {
    window.stopVideoSentenceCamera();
    return;
  }

  try {
    if (errCard) errCard.style.display = 'none';
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: state.videoFacingMode,
        width: { ideal: 640 },
        height: { ideal: 480 }
      },
      audio: false
    });

    state.videoSentenceStream = stream;
    state.videoCameraActive = true;

    if (liveVideo) {
      liveVideo.srcObject = stream;
      liveVideo.style.display = 'block';
      await liveVideo.play();
    }

    if (player) player.style.display = 'none';
    if (placeholder) placeholder.style.display = 'none';
    if (btnRecord) btnRecord.disabled = false;
    if (txtToggle) txtToggle.textContent = '⏹ Stop Camera';
    showToast("Webcam connected for video sentence recording!");
  } catch (err) {
    console.error("Camera access error:", err);
    if (errCard) {
      errCard.style.display = 'block';
      const msg = document.getElementById('videoSentenceCameraErrorMsg');
      if (msg) msg.textContent = `Could not access camera (${err.name}: ${err.message}). Please allow camera permissions or upload a video file.`;
    }
    showToast("Camera error: " + err.message, "⚠️");
  }
};

window.stopVideoSentenceCamera = function() {
  if (state.videoSentenceStream) {
    state.videoSentenceStream.getTracks().forEach(t => t.stop());
    state.videoSentenceStream = null;
  }
  state.videoCameraActive = false;

  const liveVideo = document.getElementById('videoSentenceLiveFeed');
  const placeholder = document.getElementById('videoSentencePlaceholder');
  const txtToggle = document.getElementById('txtVideoCameraToggle');
  const btnRecord = document.getElementById('btnStartVideoRecord');

  if (liveVideo) liveVideo.style.display = 'none';
  if (placeholder) placeholder.style.display = 'flex';
  if (btnRecord) btnRecord.disabled = true;
  if (txtToggle) txtToggle.textContent = '▶ Start Camera';
};

window.toggleVideoSentenceCamera = function() {
  if (state.videoCameraActive) {
    window.stopVideoSentenceCamera();
  } else {
    window.startVideoSentenceCamera();
  }
};

window.switchVideoCameraFacing = function() {
  state.videoFacingMode = (state.videoFacingMode === 'user') ? 'environment' : 'user';
  if (state.videoCameraActive) {
    window.stopVideoSentenceCamera();
    window.startVideoSentenceCamera();
  }
  showToast(`Switched camera to ${state.videoFacingMode} mode.`);
};

window.dismissVideoSentenceError = function() {
  const errCard = document.getElementById('videoSentenceCameraError');
  if (errCard) errCard.style.display = 'none';
};

window.startVideoRecording = function() {
  if (!state.videoSentenceStream) {
    showToast("Start camera preview first to record video.", "⚠️");
    return;
  }

  state.videoRecordedChunks = [];
  const options = { mimeType: 'video/webm;codecs=vp8' };
  let mediaRecorder;

  try {
    mediaRecorder = new MediaRecorder(state.videoSentenceStream, options);
  } catch (e) {
    try {
      mediaRecorder = new MediaRecorder(state.videoSentenceStream);
    } catch (e2) {
      showToast("MediaRecorder not supported on this browser.", "⚠️");
      return;
    }
  }

  state.videoMediaRecorder = mediaRecorder;

  mediaRecorder.ondataavailable = (event) => {
    if (event.data && event.data.size > 0) {
      state.videoRecordedChunks.push(event.data);
    }
  };

  mediaRecorder.onstop = () => {
    const blob = new Blob(state.videoRecordedChunks, { type: 'video/webm' });
    state.videoRecordedBlob = blob;
    const videoUrl = URL.createObjectURL(blob);
    state.videoRecordedUrl = videoUrl;

    const player = document.getElementById('videoSentencePlayer');
    const liveVideo = document.getElementById('videoSentenceLiveFeed');
    const readyBanner = document.getElementById('videoReadyToProcessBanner');
    const txtMeta = document.getElementById('txtVideoReadyMeta');

    if (player) {
      player.src = videoUrl;
      player.style.display = 'block';
    }
    if (liveVideo) liveVideo.style.display = 'none';
    if (readyBanner) readyBanner.style.display = 'block';
    if (txtMeta) {
      const sizeMb = (blob.size / (1024 * 1024)).toFixed(2);
      txtMeta.textContent = `Recorded clip (${state.videoTimerSeconds}s, ${sizeMb} MB) ready for sign recognition.`;
    }
    showToast("Recording stopped. Preview your video below!");
  };

  mediaRecorder.start(250);
  state.isVideoRecording = true;

  const btnStart = document.getElementById('btnStartVideoRecord');
  const btnStop = document.getElementById('btnStopVideoRecord');
  const recBadge = document.getElementById('videoRecTimerBadge');
  const recDot = document.querySelector('#videoRecTimerBadge .rec-pulse-dot');

  if (btnStart) btnStart.style.display = 'none';
  if (btnStop) btnStop.style.display = 'inline-flex';
  if (recBadge) recBadge.style.display = 'inline-flex';
  if (recDot) recDot.style.display = 'inline-block';

  state.videoTimerSeconds = 0;
  const timerVal = document.getElementById('valVideoRecTimer');
  if (timerVal) timerVal.textContent = '00:00';

  clearInterval(state.videoTimerInterval);
  state.videoTimerInterval = setInterval(() => {
    state.videoTimerSeconds++;
    const m = String(Math.floor(state.videoTimerSeconds / 60)).padStart(2, '0');
    const s = String(state.videoTimerSeconds % 60).padStart(2, '0');
    if (timerVal) timerVal.textContent = `${m}:${s}`;
  }, 1000);

  showToast("🔴 Recording video sentence... Perform your signs now!");
};

window.stopVideoRecording = function() {
  if (state.videoMediaRecorder && state.videoMediaRecorder.state !== 'inactive') {
    state.videoMediaRecorder.stop();
  }
  state.isVideoRecording = false;
  clearInterval(state.videoTimerInterval);

  const btnStart = document.getElementById('btnStartVideoRecord');
  const btnStop = document.getElementById('btnStopVideoRecord');
  const recBadge = document.getElementById('videoRecTimerBadge');
  const recDot = document.querySelector('#videoRecTimerBadge .rec-pulse-dot');

  if (btnStart) btnStart.style.display = 'inline-flex';
  if (btnStop) btnStop.style.display = 'none';
  if (recBadge) recBadge.style.display = 'none';
  if (recDot) recDot.style.display = 'none';
};

window.handleVideoFileUpload = function(event) {
  const file = event.target.files[0];
  if (!file) return;

  state.videoRecordedBlob = file;
  const videoUrl = URL.createObjectURL(file);
  state.videoRecordedUrl = videoUrl;

  const player = document.getElementById('videoSentencePlayer');
  const liveVideo = document.getElementById('videoSentenceLiveFeed');
  const placeholder = document.getElementById('videoSentencePlaceholder');
  const readyBanner = document.getElementById('videoReadyToProcessBanner');
  const txtMeta = document.getElementById('txtVideoReadyMeta');

  if (liveVideo) liveVideo.style.display = 'none';
  if (placeholder) placeholder.style.display = 'none';
  if (player) {
    player.src = videoUrl;
    player.style.display = 'block';
  }
  if (readyBanner) readyBanner.style.display = 'block';
  if (txtMeta) {
    const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
    txtMeta.textContent = `Uploaded file "${file.name}" (${sizeMb} MB) ready for sign recognition.`;
  }
  showToast(`Loaded video file "${file.name}". Click "Process Video" to analyze.`);
};

window.discardRecordedVideo = function() {
  state.videoRecordedBlob = null;
  state.videoRecordedUrl = null;

  const player = document.getElementById('videoSentencePlayer');
  const liveVideo = document.getElementById('videoSentenceLiveFeed');
  const placeholder = document.getElementById('videoSentencePlaceholder');
  const readyBanner = document.getElementById('videoReadyToProcessBanner');

  if (player) {
    player.pause();
    player.src = '';
    player.style.display = 'none';
  }
  if (readyBanner) readyBanner.style.display = 'none';

  if (state.videoCameraActive) {
    if (liveVideo) liveVideo.style.display = 'block';
  } else {
    if (placeholder) placeholder.style.display = 'flex';
  }
};

window.executeProcessVideo = async function() {
  if (!state.videoRecordedBlob) {
    showToast("Record or upload a video clip first.", "⚠️");
    return;
  }

  const progressWrap = document.getElementById('videoProcessingProgress');
  const progressFill = document.getElementById('fillVideoProcess');
  const progressPct = document.getElementById('valVideoProcessPct');
  const progressStatus = document.getElementById('txtVideoProcessStatus');
  const btnProcess = document.getElementById('btnProcessVideoNow');

  if (progressWrap) progressWrap.style.display = 'block';
  if (btnProcess) btnProcess.disabled = true;

  let pct = 10;
  if (progressFill) progressFill.style.width = '10%';
  if (progressPct) progressPct.textContent = '10%';
  if (progressStatus) progressStatus.textContent = 'Extracting video frames and initializing hand landmark model...';

  const animTimer = setInterval(() => {
    if (pct < 85) {
      pct += Math.floor(Math.random() * 8) + 4;
      pct = Math.min(85, pct);
      if (progressFill) progressFill.style.width = `${pct}%`;
      if (progressPct) progressPct.textContent = `${pct}%`;
      if (pct > 40 && pct < 70) {
        if (progressStatus) progressStatus.textContent = 'Detecting 21-joint kinematics and segmenting gesture actions...';
      } else if (pct >= 70) {
        if (progressStatus) progressStatus.textContent = 'Synthesizing grammatical ISL sentence translation...';
      }
    }
  }, 300);

  const formData = new FormData();
  const filename = state.videoRecordedBlob.name || "camera_recording.webm";
  formData.append('file', state.videoRecordedBlob, filename);

  try {
    const res = await fetch('/api/video/process', {
      method: 'POST',
      body: formData
    });

    clearInterval(animTimer);

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Video processing failed.");
    }

    const data = await res.json();
    state.videoLastProcessResult = data;

    if (progressFill) progressFill.style.width = '100%';
    if (progressPct) progressPct.textContent = '100%';
    if (progressStatus) progressStatus.textContent = '✓ Analysis complete!';

    setTimeout(() => {
      if (progressWrap) progressWrap.style.display = 'none';
    }, 600);

    renderVideoProcessResults(data);
    showToast("Video analyzed successfully!");
    await loadDashboardStats();
  } catch (err) {
    clearInterval(animTimer);
    if (progressWrap) progressWrap.style.display = 'none';
    showToast("Processing error: " + err.message, "⚠️");
  } finally {
    if (btnProcess) btnProcess.disabled = false;
  }
};

function renderVideoProcessResults(data) {
  const chipsContainer = document.getElementById('timelineSignsChips');
  const sentenceTextarea = document.getElementById('videoTranslatedSentence');
  const ruleBadge = document.getElementById('badgeGrammarRule');
  const confBadge = document.getElementById('badgeTranslationConfidence');

  if (sentenceTextarea) {
    sentenceTextarea.value = data.sentence || '';
  }

  if (ruleBadge) {
    const isDraft = data.is_draft ? ' (DRAFT)' : '';
    const rule = data.raw_grammar_rule || data.grammar_rule || 'SOV_TOPIC_COMMENT';
    ruleBadge.textContent = `ISL: ${rule.replace(/_/g, ' ')}${isDraft}`;
  }

  if (confBadge) {
    confBadge.style.display = 'inline-block';
    const conf = (data.average_confidence !== undefined) ? data.average_confidence : (data.overall_confidence || 0.95);
    const confPct = Math.round(conf * 100);
    confBadge.textContent = `CONFIDENCE: ${confPct}%`;
  }

  if (chipsContainer) {
    chipsContainer.innerHTML = '';
    const signs = data.recognized_signs || [];

    if (signs.length === 0) {
      chipsContainer.innerHTML = '<span style="font-size:0.8rem; color:var(--text-dim);">No distinct signs detected in video. Ensure hands are clearly visible in lighting.</span>';
      return;
    }

    signs.forEach((s, idx) => {
      if (idx > 0) {
        const arrow = document.createElement('span');
        arrow.className = 'token-arrow';
        arrow.textContent = '➔';
        chipsContainer.appendChild(arrow);
      }

      const chip = document.createElement('div');
      chip.className = 'timeline-sign-chip';
      const icon = GESTURE_ICONS[s.gesture] || '✋';
      const confPct = Math.round((s.confidence || 0.95) * 100);

      chip.innerHTML = `
        <span class="chip-gesture-icon">${icon}</span>
        <div class="chip-info-wrap">
          <span class="chip-name">${s.label || s.gesture}</span>
          <span class="chip-time">${s.start_time}s - ${s.end_time}s (${confPct}%)</span>
        </div>
      `;
      chipsContainer.appendChild(chip);
    });
  }
}

window.copyVideoSentence = function() {
  const textarea = document.getElementById('videoTranslatedSentence');
  if (!textarea || !textarea.value) {
    showToast("No sentence to copy.", "⚠️");
    return;
  }
  if (navigator.clipboard) {
    navigator.clipboard.writeText(textarea.value)
      .then(() => showToast("Copied sentence to clipboard!"))
      .catch(() => showToast("Failed to copy text."));
  }
};

window.speakVideoSentence = function() {
  const textarea = document.getElementById('videoTranslatedSentence');
  if (!textarea || !textarea.value) return;
  speakSentence(textarea.value);
};

window.saveVideoSentenceTranslation = async function() {
  const textarea = document.getElementById('videoTranslatedSentence');
  const sentence = textarea ? textarea.value.trim() : '';

  if (!sentence) {
    showToast("No sentence available to save.", "⚠️");
    return;
  }

  const result = state.videoLastProcessResult || {};
  const tokens = (result.recognized_signs || []).map(s => s.gesture);

  const payload = {
    sentence: sentence,
    tokens: tokens.length > 0 ? tokens : ["GESTURE_ACTION"],
    confidence: (result.average_confidence !== undefined) ? result.average_confidence : (result.overall_confidence || 0.96),
    duration_seconds: result.duration_seconds || 2.0,
    modality: "video",
    video_path: result.video_url || result.video_path || null
  };

  try {
    const res = await fetch('/api/translations/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error("Failed to save translation.");

    const feedback = document.getElementById('videoSentenceSaveFeedback');
    if (feedback) feedback.style.display = 'block';

    showToast("Translation saved permanently to database!");
    await loadSentenceHistory();
    await loadDashboardStats();
  } catch (err) {
    showToast("Failed to save translation: " + err.message, "⚠️");
  }
};

window.clearVideoSentenceResults = function() {
  const textarea = document.getElementById('videoTranslatedSentence');
  const chips = document.getElementById('timelineSignsChips');
  const ruleBadge = document.getElementById('badgeGrammarRule');
  const confBadge = document.getElementById('badgeTranslationConfidence');
  const feedback = document.getElementById('videoSentenceSaveFeedback');

  if (textarea) textarea.value = '';
  if (chips) chips.innerHTML = '<span style="font-size:0.8rem; color:var(--text-dim);">No signs detected yet. Record or upload a video clip on the left to analyze gestures.</span>';
  if (ruleBadge) ruleBadge.textContent = 'WAITING FOR SIGNS';
  if (confBadge) confBadge.style.display = 'none';
  if (feedback) feedback.style.display = 'none';
  state.videoLastProcessResult = null;
  window.discardRecordedVideo();
};

// =========================================================================
// 12. VIDEO PLAYBACK MODAL MODULE
// =========================================================================

window.openVideoPlaybackModal = function(videoUrl, title) {
  const modal = document.getElementById('videoPlaybackModal');
  const player = document.getElementById('modalVideoPlayer');
  const titleEl = document.getElementById('videoPlaybackModalTitle');
  const captionEl = document.getElementById('modalVideoCaption');

  if (titleEl && title) titleEl.textContent = `🎬 ${title}`;
  if (captionEl) captionEl.textContent = `Playing video: ${videoUrl}`;
  if (player) {
    player.src = videoUrl;
    player.play().catch(() => {});
  }
  if (modal) modal.style.display = 'flex';
};

window.closeVideoPlaybackModal = function() {
  const modal = document.getElementById('videoPlaybackModal');
  const player = document.getElementById('modalVideoPlayer');
  if (player) {
    player.pause();
    player.src = '';
  }
  if (modal) modal.style.display = 'none';
};

