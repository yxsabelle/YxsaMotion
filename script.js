"use strict";
/* ==========================================================
   YxsaMotion — gesture recognition with Google Teachable Machine
   ========================================================== */

const MODEL_URL = "https://teachablemachine.withgoogle.com/models/K0dYLOXIr/";
const CONFIDENCE_THRESHOLD = 0.85;  // 0–1: minimum confidence to count a prediction
const STABLE_FRAMES = 5;            // consecutive matching predictions before animating
const NEUTRAL_DELAY_MS = 700;       // how long a gesture may vanish before returning to neutral
const PREDICT_INTERVAL_MS = 120;    // time between predictions
const FLIP_INPUT = true;            // Teachable Machine webcams train on mirrored frames; set false if yours wasn't

const LABEL_MAP = {
  heartsign: "heart", heart: "heart",
  fingerheartsign: "fingerHeart", fingerheart: "fingerHeart",
  peacesign: "peace", peace: "peace",
};
const GESTURES = {
  heart:       { name: "Heart Sign",        message: "Sending you love ♡" },
  fingerHeart: { name: "Finger Heart Sign", message: "A little love, just for you 🎀" },
  peace:       { name: "Peace Sign",        message: "Peace, love, and pink vibes ✌🏻" },
};

/* ---------- 2. UI STATE ---------- */
const $ = (id) => document.getElementById(id);
const el = {
  welcome: $("welcome"), app: $("app"), stage: $("stage"), video: $("video"), loader: $("loader"),
  loaderText: $("loader-text"), start: $("btn-start"), stop: $("btn-stop"), switchCam: $("btn-switch"),
  replay: $("btn-replay"), home: $("btn-home"), getStarted: $("btn-get-started"), retry: $("btn-retry"),
  statusIcon: $("status-icon"), statusText: $("status-text"), name: $("gesture-name"), meter: $("meter"),
  conf: $("conf-text"), message: $("message"), notice: $("notice"), noticeText: $("notice-text"),
};
const NEUTRAL_MSG = "Waiting for your gesture…";
const cache = new Map(); // avoids touching the DOM when text hasn't changed
function setText(node, value) {
  if (cache.get(node) !== value) { cache.set(node, value); node.textContent = value; }
}
function setStatus(icon, text) { setText(el.statusIcon, icon); setText(el.statusText, text); }
function showNotice(text, canRetry) {
  el.noticeText.textContent = text; el.retry.hidden = !canRetry; el.notice.hidden = false;
}
function hideNotice() { el.notice.hidden = true; }
function showLoader(text) { el.loaderText.textContent = text; el.loader.hidden = false; }
function hideLoader() { el.loader.hidden = true; }
function showView(name) {
  el.welcome.hidden = name !== "welcome"; el.app.hidden = name !== "app";
  window.scrollTo(0, 0);
}
function resetReadout() {
  setText(el.name, "—"); setText(el.conf, "Confidence 0%"); el.meter.style.transform = "scaleX(0)";
}

/* ---------- 3. MODEL LOADING ---------- */
let model = null, modelPromise = null;

function modelBase() {
  let url = MODEL_URL.trim();
  if (!url || url.includes("PASTE_YOUR")) throw new Error("NO_URL");
  url = url.replace(/(model|metadata)\.json$/, "");
  return url.endsWith("/") ? url : url + "/";
}
function loadModel() {
  if (modelPromise) return modelPromise;
  modelPromise = (async () => {
    if (!window.tmImage || !window.tf) throw new Error("NO_LIB");
    const base = modelBase();
    model = await tmImage.load(base + "model.json", base + "metadata.json");
    const labels = model.getClassLabels();
    console.info("[YxsaMotion] model classes:", labels);
    if (!labels.some((l) => LABEL_MAP[normalize(l)])) {
      console.warn("[YxsaMotion] none of your class names match LABEL_MAP. Edit LABEL_MAP in script.js.");
      showNotice("Your model loaded, but none of its class names match Heart Sign, Finger Heart Sign or Peace Sign. Update LABEL_MAP in script.js (classes found: " + labels.join(", ") + ").", false);
    }
  })();
  modelPromise.catch(() => { modelPromise = null; }); // allow retry
  return modelPromise;
}
function modelErrorMessage(err) {
  if (err && err.message === "NO_URL") return "The model link is missing. Open script.js and replace MODEL_URL with your Teachable Machine link.";
  if (err && err.message === "NO_LIB") return "The recognition libraries didn't load. Check your internet connection and reload the page.";
  return "The model couldn't be loaded. Check that MODEL_URL is correct, ends with a /, and that the model is uploaded and shared publicly.";
}
async function initModel() {
  hideNotice(); showLoader("Loading model…"); setStatus("◌", "Loading model");
  try {
    await loadModel();
    if (el.notice.hidden) hideNotice();
    setStatus("○", "Camera inactive");
  } catch (err) {
    console.error(err);
    setStatus("!", "Model loading failed"); showNotice(modelErrorMessage(err), true);
  } finally { hideLoader(); }
}

/* ---------- 4. CAMERA ---------- */
let stream = null, facing = "user", running = false, predictRaf = 0, busy = false, lastPredict = 0, errorCount = 0;

function cameraErrorMessage(err) {
  const n = err && err.name;
  if (n === "NotAllowedError" || n === "SecurityError") return "Camera access was blocked. Allow the camera from the lock or camera icon in your browser's address bar, then press Start Camera again.";
  if (n === "NotFoundError" || n === "OverconstrainedError") return "No camera was found on this device. Connect a camera and try again.";
  if (n === "NotReadableError" || n === "AbortError") return "The camera is busy or unavailable. Close other apps using it, then try again.";
  return "The camera couldn't be started. Please reload the page and try again.";
}
async function startCamera() {
  hideNotice();
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !window.isSecureContext) {
    setStatus("!", "Camera unavailable");
    showNotice("Camera access needs a modern browser and a secure page (HTTPS or localhost).", false);
    return;
  }
  el.start.disabled = true;
  try { showLoader("Loading model…"); await loadModel(); }
  catch (err) { setStatus("!", "Model loading failed"); showNotice(modelErrorMessage(err), true); hideLoader(); el.start.disabled = false; return; }
  hideLoader();
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: facing }, width: { ideal: 640 }, height: { ideal: 480 } }, audio: false,
    });
  } catch (err) {
    console.error(err);
    setStatus("!", err && err.name === "NotAllowedError" ? "Camera permission denied" : "Camera unavailable");
    showNotice(cameraErrorMessage(err), false); el.start.disabled = false; return;
  }
  const track = stream.getVideoTracks()[0];
  track.addEventListener("ended", onCameraEnded);
  const fm = track.getSettings().facingMode;
  el.video.classList.toggle("mirror", fm ? fm === "user" : facing === "user"); // preview only; model gets the raw frame
  el.video.srcObject = stream;
  try { await el.video.play(); } catch (e) { /* autoplay is muted; ignore */ }
  el.stage.classList.add("live");
  el.start.hidden = true; el.start.disabled = false; el.stop.hidden = false;
  navigator.mediaDevices.enumerateDevices().then((d) => {
    el.switchCam.hidden = d.filter((x) => x.kind === "videoinput").length < 2;
  }).catch(() => {});
  running = true; errorCount = 0; resetStability();
  setStatus("✿", "Camera active"); setText(el.message, "Show me a sign ♡");
  cancelAnimationFrame(predictRaf); predictRaf = requestAnimationFrame(predictLoop);
}
function releaseStream() {
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  el.video.srcObject = null;
}
function stopCamera(statusText) {
  running = false; cancelAnimationFrame(predictRaf);
  releaseStream();
  el.stage.classList.remove("live");
  el.start.hidden = false; el.start.disabled = false; el.stop.hidden = true; el.switchCam.hidden = true;
  resetStability(); FX.stop(); resetReadout(); setText(el.message, "Show me a sign ♡");
  setStatus("○", statusText || "Camera inactive");
}
function onCameraEnded() {
  if (!running) return;
  stopCamera("Camera interrupted");
  showNotice("The camera was interrupted. Press Start Camera to continue.", false);
}
async function switchCamera() {
  facing = facing === "user" ? "environment" : "user";
  running = false; cancelAnimationFrame(predictRaf); releaseStream();
  el.stage.classList.remove("live"); el.start.hidden = false; el.stop.hidden = true;
  await startCamera();
}

/* ---------- 5. PREDICTION, MAPPING & STABILITY ---------- */
let candidate = null, streak = 0, active = null, lastActive = null, neutralSince = 0;

function normalize(label) { return String(label).toLowerCase().replace(/[^a-z0-9]/g, ""); }
function resetStability() { candidate = null; streak = 0; active = null; neutralSince = 0; }

function predictLoop(t) {
  if (!running) return;
  predictRaf = requestAnimationFrame(predictLoop);
  if (busy || t - lastPredict < PREDICT_INTERVAL_MS || el.video.readyState < 2) return;
  lastPredict = t; predictOnce();
}
async function predictOnce() {
  busy = true;
  try {
    processPrediction(await model.predict(el.video, FLIP_INPUT));
    errorCount = 0;
  } catch (err) {
    console.error(err);
    if (++errorCount >= 5) { stopCamera("Recognition stopped"); showNotice("Something went wrong while recognizing gestures. Press Start Camera to try again.", false); }
  } finally { busy = false; }
}
function processPrediction(preds) {
  let best = preds[0];
  for (const p of preds) if (p.probability > best.probability) best = p;
  const key = LABEL_MAP[normalize(best.className)];
  const pct = Math.round(best.probability * 100);
  const now = performance.now();

  if (key && best.probability >= CONFIDENCE_THRESHOLD) {
    neutralSince = 0;
    streak = key === candidate ? streak + 1 : 1; candidate = key;
    setText(el.name, GESTURES[key].name);
    setText(el.conf, "Confidence " + pct + "%"); el.meter.style.transform = "scaleX(" + best.probability.toFixed(2) + ")";
    if (streak >= STABLE_FRAMES && key !== active) activate(key);
    else if (!active) setStatus("◌", "Recognizing gesture…");
  } else {
    candidate = null; streak = 0;
    if (!neutralSince) neutralSince = now;
    if (now - neutralSince > NEUTRAL_DELAY_MS) {
      if (active) deactivate();
      setText(el.name, "—"); setText(el.conf, "Confidence " + (key ? pct : 0) + "%");
      el.meter.style.transform = "scaleX(0)";
      setStatus("✿", "No confident gesture detected");
    }
  }
}
function activate(key) {
  active = lastActive = key;
  setText(el.message, GESTURES[key].message);
  setStatus("✓", "Gesture recognized");
  FX.start(key);
}
function deactivate() {
  active = null; FX.stop(); setText(el.message, NEUTRAL_MSG);
}
function replayAnimation() {
  const key = active || lastActive;
  if (!key) { setText(el.message, "Make a gesture first, then replay it."); return; }
  setText(el.message, GESTURES[key].message); FX.replay(key);
}

/* ---------- 6. ANIMATION SYSTEM (canvas) ----------
   One engine, three effect types: heart, fingerHeart, peace.
   Each type provides make() (creates a particle) and a draw function. */
const FX = (() => {
  const cv = $("fx"), ctx = cv.getContext("2d");
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
  const PINKS = ["#f8b9cb", "#f4a0b9", "#e8829f", "#fbd5e0", "#ffffff"];
  // fill / outline / shade colors taken from the hand-drawn bow style: dusty mauve, hot pink, baby pink
  const BOW_PALS = [
    { f: "#e8a9be", s: "#a8476a", d: "#c9758f", b: "#f6d3de" },
    { f: "#e4689a", s: "#9a3358", d: "#bf4474", b: "#f19bbd" },
    { f: "#f5bfd5", s: "#b0507a", d: "#e39ab9", b: "#fde3ec" },
  ];
  let W = 0, H = 0, parts = [], mode = null, acc = 0, last = 0, raf = 0, ro = null;
  const rnd = (a, b) => a + Math.random() * (b - a);
  const pick = (a) => a[(Math.random() * a.length) | 0];
  const maxParts = () => (reduced.matches ? 16 : 40);

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = cv.clientWidth; H = cv.clientHeight;
    cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  /* particle factories */
  const rise = (shape, extra) => Object.assign({ shape, x: rnd(0.08, 0.92) * W, y: H + 24, vy: -rnd(45, 95), sway: rnd(8, 28), ph: rnd(0, 6), size: rnd(9, 24), color: pick(PINKS), life: rnd(4, 6.5), rot: rnd(-0.3, 0.3) }, extra);
  const TYPES = {
    heart: { rate: 4.5, make: () => rise("heart") },
    fingerHeart: { rate: 1.1, make: () => Math.random() < 0.55
      ? rise("cuteBow", { size: rnd(15, 22), vy: -rnd(18, 36), life: 4.5, pop: true })
      : rise("bow", { size: rnd(15, 24), vy: -rnd(18, 38), pal: pick(BOW_PALS), life: 4.5, pop: true }) },
    peace: { rate: 3.2, make: () => Math.random() < 0.7
      ? rise("peace", { vy: -rnd(30, 70), size: rnd(14, 26), life: rnd(3, 4.5), pop: true })
      : rise("star", { vy: -rnd(30, 70), size: rnd(8, 14), life: rnd(3, 4.5), pop: true }) },
  };
  const sparkle = () => ({ shape: "sparkle", x: rnd(0, W), y: rnd(0, H), size: rnd(4, 9), life: rnd(1.2, 2), color: "#ffffff", vy: 0, sway: 0, ph: rnd(0, 6), rot: 0 });

  /* shape drawing (origin already translated to the particle) */
  function heartPath(s) {
    ctx.beginPath(); ctx.moveTo(0, s * 0.45);
    ctx.bezierCurveTo(-s * 1.1, -s * 0.1, -s * 0.55, -s * 0.9, 0, -s * 0.4);
    ctx.bezierCurveTo(s * 0.55, -s * 0.9, s * 1.1, -s * 0.1, 0, s * 0.45); ctx.closePath();
  }
  function star4(s) {
    ctx.beginPath();
    for (let i = 0; i < 8; i++) { const r = i % 2 ? s * 0.25 : s, a = (i * Math.PI) / 4; ctx.lineTo(Math.cos(a) * r, Math.sin(a) * r); }
    ctx.closePath();
  }
  function star5(s) {
    ctx.beginPath();
    for (let i = 0; i < 10; i++) { const r = i % 2 ? s * 0.45 : s, a = (i * Math.PI) / 5 - Math.PI / 2; ctx.lineTo(Math.cos(a) * r, Math.sin(a) * r); }
    ctx.closePath();
  }
  const DRAW = {
    heart(p) { heartPath(p.size); ctx.fillStyle = p.color; ctx.shadowColor = "rgba(244,160,185,.7)"; ctx.shadowBlur = 8; ctx.fill(); },
    sparkle(p) { star4(p.size); ctx.fillStyle = "#fff"; ctx.shadowColor = "#f4a0b9"; ctx.shadowBlur = 6; ctx.fill(); },
    star(p) { star5(p.size); ctx.fillStyle = p.color; ctx.fill(); },
    peace(p) {
      const s = p.size; ctx.lineWidth = Math.max(2, s * 0.14); ctx.lineCap = "round"; ctx.strokeStyle = p.color === "#ffffff" ? "#f4a0b9" : p.color;
      ctx.fillStyle = "rgba(255,255,255,.55)"; ctx.beginPath(); ctx.arc(0, 0, s, 0, 6.2832); ctx.fill(); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, -s); ctx.lineTo(0, s); ctx.moveTo(0, 0); ctx.lineTo(-s * 0.7, s * 0.7); ctx.moveTo(0, 0); ctx.lineTo(s * 0.7, s * 0.7); ctx.stroke();
    },
    cuteBow(p) { // glossy cartoon bow: dark outline, shiny highlights, round knot, notched tails
      const s = p.size, sw = Math.sin(p.age * 2 + p.ph) * 0.07, ink = "#3b1c28";
      ctx.lineJoin = "round"; ctx.lineCap = "round"; ctx.strokeStyle = ink; ctx.lineWidth = 1.5;
      for (const d of [-1, 1]) { // tails
        ctx.save(); ctx.rotate(-d * (0.3 + sw)); ctx.beginPath();
        ctx.moveTo(-s * 0.16, s * 0.1); ctx.lineTo(s * 0.16, s * 0.1);
        ctx.quadraticCurveTo(s * 0.3, s * 0.8, s * 0.34, s * 1.55);
        ctx.lineTo(0, s * 1.25); ctx.lineTo(-s * 0.3, s * 1.55);
        ctx.quadraticCurveTo(-s * 0.28, s * 0.8, -s * 0.16, s * 0.1); ctx.closePath();
        ctx.fillStyle = "#f8bcd0"; ctx.fill(); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(-s * 0.06, s * 0.3); ctx.lineTo(-s * 0.02, s * 1.0);
        ctx.strokeStyle = "rgba(255,255,255,.85)"; ctx.lineWidth = s * 0.1; ctx.stroke(); ctx.restore();
        ctx.strokeStyle = ink; ctx.lineWidth = 1.5;
      }
      for (const d of [-1, 1]) { // loops
        ctx.beginPath(); ctx.moveTo(d * s * 0.2, -s * 0.15);
        ctx.bezierCurveTo(d * s * 0.6, -s * 0.75, d * s * 1.2, -s * 1.0, d * s * 1.55, -s * 0.55);
        ctx.bezierCurveTo(d * s * 1.78, -s * 0.2, d * s * 1.62, s * 0.35, d * s * 1.4, s * 0.68);
        ctx.bezierCurveTo(d * s * 1.0, s * 0.62, d * s * 0.5, s * 0.45, d * s * 0.2, s * 0.15); ctx.closePath();
        ctx.fillStyle = "#f9c4d6"; ctx.fill(); ctx.stroke();
        ctx.beginPath(); ctx.ellipse(d * s * 0.95, s * 0.4, s * 0.45, s * 0.14, d * 0.15, 0, 6.2832); // soft underside
        ctx.fillStyle = "#f29bb9"; ctx.fill();
        ctx.beginPath(); // fold lines
        ctx.moveTo(d * s * 0.25, -s * 0.02); ctx.quadraticCurveTo(d * s * 0.8, -s * 0.28, d * s * 1.3, -s * 0.12);
        ctx.moveTo(d * s * 0.25, s * 0.08); ctx.quadraticCurveTo(d * s * 0.8, s * 0.1, d * s * 1.25, s * 0.3);
        ctx.strokeStyle = "#e8789f"; ctx.lineWidth = 1.1; ctx.stroke();
        ctx.beginPath(); ctx.ellipse(d * s * 0.9, -s * 0.5, s * 0.38, s * 0.13, d * -0.45, 0, 6.2832); // gloss
        ctx.fillStyle = "rgba(255,255,255,.9)"; ctx.fill();
        ctx.beginPath(); ctx.arc(d * s * 1.32, -s * 0.28, s * 0.07, 0, 6.2832); ctx.fill();
        ctx.strokeStyle = ink; ctx.lineWidth = 1.5;
      }
      ctx.beginPath(); ctx.arc(0, 0, s * 0.32, 0, 6.2832); // round knot
      ctx.fillStyle = "#fbc9da"; ctx.fill(); ctx.stroke();
      ctx.beginPath(); ctx.ellipse(-s * 0.1, -s * 0.1, s * 0.13, s * 0.08, -0.6, 0, 6.2832);
      ctx.fillStyle = "rgba(255,255,255,.95)"; ctx.fill();
    },
    bow(p) {
      const s = p.size, c = p.pal, sw = Math.sin(p.age * 2.2 + p.ph) * 0.1;
      ctx.lineJoin = "round"; ctx.lineWidth = 1.3; ctx.strokeStyle = c.s; ctx.fillStyle = c.f;
      for (const d of [-1, 1]) { // swallow-tail ribbon ends, swaying
        ctx.save(); ctx.rotate(d * (0.35 + sw)); ctx.beginPath();
        ctx.moveTo(-s * 0.16, s * 0.1); ctx.lineTo(s * 0.16, s * 0.1);
        ctx.quadraticCurveTo(s * 0.3, s * 0.9, s * 0.2, s * 1.6);
        ctx.lineTo(0, s * 1.35); ctx.lineTo(-s * 0.2, s * 1.6);
        ctx.quadraticCurveTo(-s * 0.3, s * 0.9, -s * 0.16, s * 0.1); ctx.closePath();
        ctx.fill(); ctx.stroke(); ctx.restore();
      }
      for (const d of [-1, 1]) { // puffy loops
        const loop = () => { ctx.beginPath(); ctx.moveTo(0, 0);
          ctx.bezierCurveTo(d * s * 0.5, -s * 1.0, d * s * 1.8, -s * 1.0, d * s * 1.8, -s * 0.05);
          ctx.bezierCurveTo(d * s * 1.8, s * 0.8, d * s * 0.6, s * 0.85, 0, 0); ctx.closePath(); };
        loop(); ctx.fillStyle = c.f; ctx.fill(); ctx.stroke();
        ctx.save(); ctx.translate(d * s * 0.28, 0); ctx.scale(0.62, 0.52); loop(); // inner shade
        ctx.fillStyle = c.d; ctx.globalAlpha *= 0.75; ctx.fill(); ctx.restore();
        ctx.save(); ctx.scale(0.88, 0.88); loop(); ctx.setLineDash([2, 3]); // stitching
        ctx.lineWidth = 0.8; ctx.strokeStyle = "rgba(255,255,255,.85)"; ctx.stroke(); ctx.restore();
      }
      ctx.beginPath(); ctx.ellipse(0, 0, s * 0.3, s * 0.38, 0, 0, 6.2832); // knot
      ctx.fillStyle = c.f; ctx.fill(); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(-s * 0.1, -s * 0.2); ctx.lineTo(-s * 0.1, s * 0.2);
      ctx.strokeStyle = "rgba(255,255,255,.8)"; ctx.lineWidth = 1; ctx.stroke();
    },
  };
  /* engine */
  function spawn() {
    if (parts.length >= maxParts()) return;
    parts.push(TYPES[mode].make());
    if (Math.random() < 0.35 && parts.length < maxParts()) parts.push(sparkle());
  }
  function tick(ts) {
    const dt = Math.min(0.05, (ts - last) / 1000 || 0.016); last = ts;
    const slow = reduced.matches ? 0.5 : 1;
    if (mode) { acc += dt * TYPES[mode].rate * (reduced.matches ? 0.4 : 1); while (acc >= 1) { acc -= 1; spawn(); } }
    ctx.clearRect(0, 0, W, H);
    parts = parts.filter((p) => {
      p.age = (p.age || 0) + dt;
      if (p.kill && p.dieAt === undefined) p.dieAt = p.age;
      const fadeOut = p.dieAt === undefined ? 1 : 1 - (p.age - p.dieAt) / 0.6;
      return p.age < p.life && fadeOut > 0;
    });
    for (const p of parts) {
      const fadeOut = p.dieAt === undefined ? 1 : 1 - (p.age - p.dieAt) / 0.6;
      const alpha = Math.max(0, Math.min(1, p.age / 0.5, (p.life - p.age) / 1.2, fadeOut));
      if (p.vy) p.y += p.vy * slow * dt;
      const x = p.x + Math.sin(p.age * 1.4 + p.ph) * p.sway * slow;
      const pop = p.pop ? Math.min(1, p.age / 0.35) : 1, sc = p.pop ? 1 + Math.sin(pop * Math.PI) * 0.35 : 1;
      const tw = p.shape === "sparkle" ? 0.6 + 0.4 * Math.sin(p.age * 8 + p.ph) : 1;
      ctx.save(); ctx.translate(x, p.y); ctx.rotate(p.rot + Math.sin(p.age + p.ph) * 0.15); ctx.scale(sc * tw, sc * tw);
      ctx.globalAlpha = alpha; DRAW[p.shape](p); ctx.restore();
    }
    raf = mode || parts.length ? requestAnimationFrame(tick) : 0;
  }
  function run() { if (!raf) { last = performance.now(); raf = requestAnimationFrame(tick); } }
  function killAll() { for (const p of parts) p.kill = true; }

  return {
    init() { resize(); ro = new ResizeObserver(resize); ro.observe(cv); },
    start(key) { try { killAll(); mode = key; acc = 1; run(); } catch (e) { console.error(e); } },
    stop() { mode = null; killAll(); },
    replay(key) { try { parts = []; mode = key; acc = 3; run(); } catch (e) { console.error(e); } },
    destroy() { mode = null; parts = []; cancelAnimationFrame(raf); raf = 0; if (ro) ro.disconnect(); ro = null; ctx.clearRect(0, 0, W, H); },
  };
})();

/* ---------- 7. NAVIGATION & EVENTS ---------- */
function enterExperience() {
  showView("app"); FX.init(); initModel(); // load the model when the experience starts
  el.start.focus({ preventScroll: true });
}
function goHome() {
  stopCamera(); FX.destroy(); hideNotice(); hideLoader(); showView("welcome");
  el.getStarted.focus({ preventScroll: true });
}
el.getStarted.addEventListener("click", enterExperience);
el.home.addEventListener("click", goHome);
el.start.addEventListener("click", startCamera);
el.stop.addEventListener("click", () => stopCamera());
el.switchCam.addEventListener("click", switchCamera);
el.replay.addEventListener("click", replayAnimation);
el.retry.addEventListener("click", initModel);
window.addEventListener("pagehide", () => { releaseStream(); });