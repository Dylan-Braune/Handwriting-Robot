/* app.js -- shared across all three pages: backend connection settings,
   a small fetch() wrapper, and the top nav/settings widget. */

const STORAGE_KEYS = {
  backendUrl: "hwrobot_backend_url",
  apiKey: "hwrobot_api_key",
};

function getBackendUrl() {
  return (localStorage.getItem(STORAGE_KEYS.backendUrl) || "http://localhost:5000").replace(/\/+$/, "");
}

function getApiKey() {
  return localStorage.getItem(STORAGE_KEYS.apiKey) || "";
}

function setBackendConfig(url, key) {
  localStorage.setItem(STORAGE_KEYS.backendUrl, url);
  localStorage.setItem(STORAGE_KEYS.apiKey, key);
}

/** fetch() wrapper that prefixes the configured backend URL and attaches
 * the API key header if one is set. Throws with a readable message on
 * any non-2xx response or network failure. */
async function apiFetch(path, options = {}) {
  const url = getBackendUrl() + path;
  const headers = options.headers || {};
  const key = getApiKey();
  if (key) headers["X-API-Key"] = key;
  let res;
  try {
    res = await fetch(url, { ...options, headers });
  } catch (e) {
    throw new Error(
      `Could not reach the backend at ${getBackendUrl()}. Is server.py running on your Odroid, ` +
      `and is the address in Settings (top right) correct? (${e.message})`
    );
  }
  if (!res.ok) {
    let detail = "";
    try {
      const j = await res.json();
      detail = j.error || JSON.stringify(j);
    } catch (_) {
      detail = res.statusText;
    }
    throw new Error(`Backend returned ${res.status}: ${detail}`);
  }
  const ct = res.headers.get("content-type") || "";
  return ct.includes("application/json") ? res.json() : res;
}

function absoluteUrl(path) {
  if (!path) return path;
  return path.startsWith("http") ? path : getBackendUrl() + path;
}

async function checkBackendStatus() {
  const dot = document.getElementById("statusDot");
  const label = document.getElementById("statusLabel");
  if (!dot) return;
  try {
    await apiFetch("/api/authors");
    dot.className = "status-dot ok";
    label.textContent = "Connected";
  } catch (e) {
    dot.className = "status-dot bad";
    label.textContent = "Not connected";
  }
}

/** Populates any <select data-author-select> element on the page with
 * the ten authors from the backend. Called once per page on load. */
async function populateAuthorDropdowns() {
  const selects = document.querySelectorAll("select[data-author-select]");
  if (!selects.length) return;
  try {
    const authors = await apiFetch("/api/authors");
    selects.forEach((sel) => {
      sel.innerHTML = "";
      authors.forEach((a) => {
        const opt = document.createElement("option");
        opt.value = a.id;
        opt.textContent = a.id;
        sel.appendChild(opt);
      });
    });
    return authors;
  } catch (e) {
    selects.forEach((sel) => {
      sel.innerHTML = '<option value="">(backend not reachable)</option>';
    });
  }
}

function renderTopbar(activePage) {
  const el = document.getElementById("topbar");
  if (!el) return;
  el.innerHTML = `
    <header class="topbar">
      <div class="brand"><span class="dot"></span> Handwriting Robot</div>
      <nav class="tabs">
        <a href="index.html" class="${activePage === "read" ? "active" : ""}">Read</a>
        <a href="write.html" class="${activePage === "write" ? "active" : ""}">Write</a>
        <a href="stats.html" class="${activePage === "stats" ? "active" : ""}">Model &amp; Style</a>
      </nav>
      <div class="settings-toggle" id="settingsToggle">
        <span class="status-dot" id="statusDot"></span>
        <span id="statusLabel">Checking...</span>
      </div>
    </header>
    <div class="settings-panel" id="settingsPanel">
      <label>Backend address (your Odroid)</label>
      <input type="text" id="backendUrlInput" placeholder="http://192.168.1.50:5000" />
      <label>API key (only if you set WEBAPP_API_KEY on the server)</label>
      <input type="text" id="apiKeyInput" placeholder="(optional)" />
      <div class="btn-row">
        <button id="saveSettingsBtn">Save &amp; reconnect</button>
      </div>
      <p class="hint">This is saved in your browser only. It never leaves your device except as the address requests are sent to.</p>
    </div>
  `;

  document.getElementById("backendUrlInput").value = getBackendUrl();
  document.getElementById("apiKeyInput").value = getApiKey();

  document.getElementById("settingsToggle").addEventListener("click", () => {
    document.getElementById("settingsPanel").classList.toggle("show");
  });
  document.getElementById("saveSettingsBtn").addEventListener("click", () => {
    setBackendConfig(
      document.getElementById("backendUrlInput").value.trim() || "http://localhost:5000",
      document.getElementById("apiKeyInput").value.trim()
    );
    document.getElementById("settingsPanel").classList.remove("show");
    checkBackendStatus();
    populateAuthorDropdowns();
  });

  checkBackendStatus();
}

function showError(boxId, message) {
  const box = document.getElementById(boxId);
  if (!box) return;
  box.textContent = message;
  box.classList.add("show");
}

function clearError(boxId) {
  const box = document.getElementById(boxId);
  if (!box) return;
  box.classList.remove("show");
  box.textContent = "";
}
