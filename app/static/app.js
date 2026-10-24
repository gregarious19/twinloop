// TwinLoop viewer (SPEC §6.6): upload, poll, render a coloured coronary tree, list lesions.
import * as THREE from "three";
import { OrbitControls } from "./vendor/OrbitControls.js";
import { STOPS, SMALL_DIAMETER_MM, sampleColor } from "./color.js";

const STEP_LABELS = {
  queued: "Waiting in queue",
  loading: "Reading scan",
  segmenting: "Finding coronary arteries",
  extracting: "Measuring vessels",
  solving: "Computing FFR",
};
const RADIAL_SEGMENTS = 8;
const SIGNIFICANT_FFR = 0.8;
const POLL_MS = 2000;
const LIST_MS = 10000;
const RETRY_MS = 5000;

const $ = (id) => document.getElementById(id);
const els = {
  offline: $("offline"),
  form: $("upload-form"),
  image: $("image"),
  mask: $("mask"),
  pressure: $("pressure"),
  analyze: $("analyze"),
  progress: $("upload-progress"),
  bar: $("upload-bar"),
  progressText: $("upload-text"),
  formError: $("form-error"),
  caseList: $("case-list"),
  state: $("state"),
  canvasWrap: $("canvas-wrap"),
  legend: $("legend"),
  tooltip: $("tooltip"),
  resultsPanel: $("results-panel"),
  summary: $("summary"),
  lesionBody: document.querySelector("#lesion-table tbody"),
};

let cases = [];
let selectedId = null;
let pollTimer = null;
let retryTimer = null;
let offline = false;
let viewer = null;

// ---- API helper: every fetch goes through here so the offline banner is consistent ----------
function setOffline(value) {
  offline = value;
  els.offline.hidden = !value;
}

async function api(path, options) {
  try {
    const response = await fetch(path, options);
    if (offline) onReconnected();
    return response;
  } catch (err) {
    if (!offline) setOffline(true);
    scheduleRetry();
    throw err;
  }
}

function scheduleRetry() {
  if (retryTimer) return;
  retryTimer = setTimeout(async () => {
    retryTimer = null;
    try {
      await api("/api/health");
    } catch {
      /* api() already rescheduled */
    }
  }, RETRY_MS);
}

function onReconnected() {
  setOffline(false);
  refreshCases();
  if (selectedId) selectCase(selectedId);
}

// ---- Stage states ------------------------------------------------------------------------
function clearState() {
  els.state.replaceChildren();
}

function showStage(kind) {
  const canvas = kind === "result";
  els.canvasWrap.hidden = !canvas;
  els.resultsPanel.hidden = !canvas;
  els.state.hidden = canvas;
  if (!canvas) clearState();
}

function showEmpty() {
  showStage("state");
  const p = document.createElement("p");
  p.textContent = "Upload a CCTA scan (.nii.gz or zipped DICOM) to begin.";
  els.state.append(p);
}

function showProcessing(status) {
  showStage("state");
  const spinner = document.createElement("div");
  spinner.className = "spinner";
  const label = document.createElement("p");
  label.textContent = STEP_LABELS[status.status] || status.status;
  const file = document.createElement("p");
  file.textContent = status.filename;
  els.state.append(spinner, label, file);
}

function showFailed(status) {
  showStage("state");
  const box = document.createElement("div");
  box.className = "error-box";
  box.textContent = status.error || "Processing failed";
  const retry = document.createElement("button");
  retry.type = "button";
  retry.textContent = "Try another file";
  retry.addEventListener("click", () => {
    selectedId = null;
    renderCaseList();
    showEmpty();
    els.image.focus();
  });
  els.state.append(box, retry);
}

function showUploading(percent) {
  showStage("state");
  const p = document.createElement("p");
  p.textContent = `Uploading… ${percent}%`;
  els.state.append(p);
}

// ---- Case list ---------------------------------------------------------------------------
async function refreshCases() {
  try {
    const response = await api("/api/cases");
    if (!response.ok) return;
    cases = await response.json();
    renderCaseList();
    if (!selectedId && cases.length === 0 && !viewer?.hasResult) showEmpty();
  } catch {
    /* offline banner already shown */
  }
}

function renderCaseList() {
  els.caseList.replaceChildren();
  for (const c of cases) {
    const li = document.createElement("li");
    li.dataset.id = c.case_id;
    if (c.case_id === selectedId) li.classList.add("selected");
    const left = document.createElement("div");
    const name = document.createElement("div");
    name.className = "name";
    name.textContent = c.filename;
    const when = document.createElement("div");
    when.className = "when";
    when.textContent = new Date(c.created_at).toLocaleString();
    left.append(name, when);
    const badge = document.createElement("span");
    badge.className = `badge ${c.status}`;
    badge.textContent = c.status;
    li.append(left, badge);
    li.addEventListener("click", () => selectCase(c.case_id));
    els.caseList.append(li);
  }
}

// ---- Selecting / polling a case ----------------------------------------------------------
function stopPolling() {
  if (pollTimer) clearTimeout(pollTimer);
  pollTimer = null;
}

async function selectCase(id) {
  stopPolling();
  selectedId = id;
  renderCaseList();
  let status;
  try {
    const response = await api(`/api/cases/${id}`);
    if (!response.ok) return;
    status = await response.json();
  } catch {
    pollTimer = setTimeout(() => selectCase(id), RETRY_MS);
    return;
  }
  if (selectedId !== id) return;
  if (status.status === "done") {
    await loadResult(id);
  } else if (status.status === "failed") {
    showFailed(status);
  } else {
    showProcessing(status);
    pollTimer = setTimeout(() => selectCase(id), POLL_MS);
    return;
  }
  refreshCases();
}

async function loadResult(id) {
  try {
    const response = await api(`/api/cases/${id}/result`);
    if (!response.ok) {
      const detail = (await response.json().catch(() => ({}))).detail || "Result not ready";
      showFailed({ error: detail });
      return;
    }
    const result = await response.json();
    if (selectedId !== id) return;
    renderResult(result);
  } catch {
    pollTimer = setTimeout(() => loadResult(id), RETRY_MS);
  }
}

// ---- Upload ------------------------------------------------------------------------------
els.form.addEventListener("submit", (event) => {
  event.preventDefault();
  els.formError.textContent = "";
  if (!els.image.files.length) {
    els.formError.textContent = "Choose a scan file first.";
    return;
  }
  const data = new FormData();
  data.append("image", els.image.files[0]);
  if (els.mask.files.length) data.append("mask", els.mask.files[0]);
  data.append("aortic_pressure_mmhg", els.pressure.value);

  stopPolling();
  els.analyze.disabled = true;
  els.progress.hidden = false;
  setProgress(0);
  showUploading(0);

  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/cases");
  xhr.upload.onprogress = (e) => {
    if (!e.lengthComputable) return;
    const percent = Math.round((100 * e.loaded) / e.total);
    setProgress(percent);
    showUploading(percent);
  };
  const finish = () => {
    els.analyze.disabled = false;
    els.progress.hidden = true;
  };
  xhr.onload = async () => {
    finish();
    let body = {};
    try {
      body = JSON.parse(xhr.responseText);
    } catch {
      /* non-JSON error page */
    }
    if (xhr.status === 202) {
      await refreshCases();
      selectCase(body.case_id);
    } else {
      els.formError.textContent = body.detail || `Upload failed (HTTP ${xhr.status})`;
      if (selectedId) selectCase(selectedId);
      else showEmpty();
    }
  };
  xhr.onerror = () => {
    finish();
    setOffline(true);
    scheduleRetry();
    els.formError.textContent = "Upload failed: server not reachable.";
    if (selectedId) selectCase(selectedId);
    else showEmpty();
  };
  xhr.send(data);
});

function setProgress(percent) {
  els.bar.style.width = `${percent}%`;
  els.progressText.textContent = `Uploading… ${percent}%`;
}

// ---- Result rendering --------------------------------------------------------------------
function fmt(value, digits = 2) {
  return value === null || value === undefined ? "n/a" : Number(value).toFixed(digits);
}

function renderResult(result) {
  showStage("result");
  renderSummary(result);
  renderLesions(result);
  renderLegend();
  try {
    if (!viewer) viewer = createViewer();
    viewer.show(result);
  } catch (err) {
    showStage("state");
    const box = document.createElement("div");
    box.className = "error-box";
    box.textContent = `3D view unavailable: ${err.message}`;
    els.state.append(box);
    els.resultsPanel.hidden = false;
  }
}

function renderSummary(result) {
  const s = result.summary;
  els.summary.replaceChildren();
  const dl = document.createElement("dl");
  const rows = [
    ["Min FFR", s.min_ffr === null ? "n/a" : `${fmt(s.min_ffr, 2)} (${s.min_ffr_branch_id})`],
    ["Lesions", String(s.n_lesions)],
    ["Branches with FFR ≤ 0.80", String(s.n_significant_branches)],
    ["Total time", `${fmt(result.timings_s.total, 1)} s`],
    ["Trees", String(result.trees.length)],
  ];
  for (const [k, v] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = k;
    const dd = document.createElement("dd");
    dd.textContent = v;
    dl.append(dt, dd);
  }
  els.summary.append(dl);
  const note = document.createElement("p");
  note.className = "note";
  note.textContent = result.disclaimer;
  els.summary.append(note);
  for (const w of result.warnings) {
    const warn = document.createElement("p");
    warn.className = "note";
    warn.textContent = `Warning: ${w}`;
    els.summary.append(warn);
  }
}

function renderLesions(result) {
  els.lesionBody.replaceChildren();
  if (result.lesions.length === 0) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 7;
    td.textContent = "No lesions detected";
    tr.append(td);
    els.lesionBody.append(tr);
    return;
  }
  for (const les of result.lesions) {
    const tr = document.createElement("tr");
    if (les.ffr_distal <= SIGNIFICANT_FFR) tr.classList.add("significant");
    const cells = [
      les.lesion_id,
      les.branch_id,
      fmt(les.diameter_stenosis_pct, 1),
      fmt(les.length_mm, 1),
      fmt(les.ffr_proximal, 2),
      fmt(les.ffr_distal, 2),
      fmt(les.delta_ffr, 2),
    ];
    for (const text of cells) {
      const td = document.createElement("td");
      td.textContent = text;
      tr.append(td);
    }
    tr.addEventListener("click", () => viewer?.focusLesion(les.position_mm));
    els.lesionBody.append(tr);
  }
}

function renderLegend() {
  els.legend.hidden = false;
  els.legend.replaceChildren();
  const title = document.createElement("div");
  title.textContent = "CT-FFR (unvalidated)";
  const lo = 0.6;
  const hi = 1.0;
  const pct = (v) => ((v - lo) / (hi - lo)) * 100;
  const gradient = document.createElement("div");
  gradient.className = "gradient";
  const stops = STOPS.map(([v, c]) => `${c} ${pct(v)}%`).join(", ");
  gradient.style.background = `linear-gradient(to right, ${STOPS[0][1]} 0%, ${stops})`;
  const mark = document.createElement("div");
  mark.className = "mark";
  mark.style.left = `${pct(SIGNIFICANT_FFR)}%`;
  gradient.append(mark);
  const ticks = document.createElement("div");
  ticks.className = "ticks";
  for (const v of [0.7, 0.8, 0.9, 1.0]) {
    const span = document.createElement("span");
    span.style.left = `${pct(v)}%`;
    span.textContent = v.toFixed(2);
    ticks.append(span);
  }
  const small = document.createElement("div");
  const swatch = document.createElement("span");
  swatch.className = "swatch";
  small.append(swatch, `⌀ < ${SMALL_DIAMETER_MM} mm`);
  small.style.marginTop = "6px";
  els.legend.append(title, gradient, ticks, small);
}

// ---- three.js viewer ---------------------------------------------------------------------
function createViewer() {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  els.canvasWrap.prepend(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 5000);
  camera.up.set(0, 0, 1); // LPS: +z is superior
  scene.add(new THREE.AmbientLight(0xffffff, 0.8));
  const sun = new THREE.DirectionalLight(0xffffff, 1.4);
  sun.position.set(1, -1, 2);
  scene.add(sun);
  scene.add(new THREE.DirectionalLight(0xffffff, 0.5).translateX(-1).translateZ(-1));

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  const group = new THREE.Group();
  scene.add(group);
  const marker = new THREE.Mesh(
    new THREE.SphereGeometry(1.5, 16, 12),
    new THREE.MeshBasicMaterial({ color: 0xffffff, depthTest: false }), // visible through the vessel wall
  );
  marker.renderOrder = 10;
  marker.visible = false;
  scene.add(marker);

  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  const state = { center: new THREE.Vector3(), hasResult: false };

  function resize() {
    const { clientWidth: w, clientHeight: h } = els.canvasWrap;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(els.canvasWrap);

  renderer.setAnimationLoop(() => {
    controls.update();
    renderer.render(scene, camera);
  });

  function clearGroup() {
    for (const mesh of group.children) {
      mesh.geometry.dispose();
      mesh.material.dispose();
    }
    group.clear();
  }

  function buildBranch(branch, center) {
    const n = branch.points_mm.length;
    if (n < 2) return null;
    const points = branch.points_mm.map((p) => new THREE.Vector3(p[0], p[1], p[2]).sub(center));
    const curve = new THREE.CatmullRomCurve3(points);
    const tubular = Math.max(8, n);
    const meanRadius = branch.radius_mm.reduce((a, b) => a + b, 0) / n;
    const geometry = new THREE.TubeGeometry(curve, tubular, meanRadius, RADIAL_SEGMENTS, false);
    const colors = new Float32Array(geometry.attributes.position.count * 3);
    const color = new THREE.Color();
    const perRing = RADIAL_SEGMENTS + 1;
    for (let j = 0; j <= tubular; j++) {
      const idx = Math.round((j / tubular) * (n - 1));
      color.set(sampleColor(branch.ffr[idx], branch.radius_mm[idx]));
      for (let k = 0; k < perRing; k++) color.toArray(colors, (j * perRing + k) * 3);
    }
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    const material = new THREE.MeshStandardMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      roughness: 0.6,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.userData = { branch };
    return mesh;
  }

  function show(result) {
    clearGroup();
    marker.visible = false;
    els.tooltip.hidden = true;
    const all = result.trees.flatMap((t) => t.branches.flatMap((b) => b.points_mm));
    if (all.length === 0) return;
    const center = new THREE.Vector3();
    for (const p of all) center.add(new THREE.Vector3(p[0], p[1], p[2]));
    center.divideScalar(all.length);
    state.center = center;
    for (const tree of result.trees) {
      for (const branch of tree.branches) {
        const mesh = buildBranch(branch, center);
        if (mesh) group.add(mesh);
      }
    }
    const box = new THREE.Box3().setFromObject(group);
    const sphere = box.getBoundingSphere(new THREE.Sphere());
    const distance = (sphere.radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2))) * 1.15;
    controls.target.copy(sphere.center);
    camera.position.copy(sphere.center).add(new THREE.Vector3(0, -distance, distance * 0.25));
    camera.near = Math.max(0.1, distance / 200);
    camera.far = distance * 10;
    camera.updateProjectionMatrix();
    controls.update();
    state.hasResult = true;
    resize();
  }

  function focusLesion(position) {
    const target = new THREE.Vector3(position[0], position[1], position[2]).sub(state.center);
    marker.position.copy(target);
    marker.visible = true;
    const offset = camera.position.clone().sub(controls.target);
    offset.setLength(25);
    controls.target.copy(target);
    camera.position.copy(target).add(offset);
    controls.update();
  }

  // Click (not drag) -> tooltip with FFR, diameter and branch id.
  let down = null;
  renderer.domElement.addEventListener("pointerdown", (e) => {
    down = { x: e.clientX, y: e.clientY };
  });
  renderer.domElement.addEventListener("pointerup", (e) => {
    if (!down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 4) return;
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.set(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1,
    );
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObjects(group.children, false)[0];
    if (!hit) {
      els.tooltip.hidden = true;
      return;
    }
    const branch = hit.object.userData.branch;
    const idx = Math.max(
      0,
      Math.min(branch.ffr.length - 1, Math.round(hit.uv.x * (branch.ffr.length - 1))),
    );
    const diameter = 2 * branch.radius_mm[idx];
    els.tooltip.textContent = `FFR ${branch.ffr[idx].toFixed(2)} · ⌀ ${diameter.toFixed(1)} mm · ${branch.branch_id}`;
    els.tooltip.style.left = `${e.clientX - rect.left + 12}px`;
    els.tooltip.style.top = `${e.clientY - rect.top + 12}px`;
    els.tooltip.hidden = false;
  });

  return {
    show,
    focusLesion,
    get hasResult() {
      return state.hasResult;
    },
  };
}

// ---- Boot --------------------------------------------------------------------------------
showEmpty();
refreshCases();
setInterval(refreshCases, LIST_MS);
