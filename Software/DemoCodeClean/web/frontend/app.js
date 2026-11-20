/* app.js -- shared across all three pages: backend connection settings,
   a small fetch() wrapper, and the top nav/settings widget. */

const STORAGE_KEYS = {
  backendUrl: "hwrobot_backend_url",
  apiKey: "hwrobot_api_key",
};

function getBackendUrl() {
  return (localStorage.getItem(STORAGE_KEYS.backendUrl) || "http://localhost:5000").replace(/\/+$/, "");    //any trailing slashes come off the saved address
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
  if (key) headers["X-API-Key"] = key;   //the key header goes on only when one is set
  let res;
  try {
    res = await fetch(url, { ...options, headers });   // the caller options with the headers merged in
  } catch (e) {
    throw new Error(`Could not reach the backend at ${getBackendUrl()}.`);
  }
  if (!res.ok) {
    let detail = "";
    try {
      const j = await res.json();
      detail = j.error || JSON.stringify(j);   // the error field of the body where there is one
    } catch (_) {
      detail = res.statusText;
    }
    throw new Error(`Backend returned ${res.status}: ${detail}`);
  }
  const ct = res.headers.get("content-type") || "";   //json is parsed and anything else handed back whole
  return ct.includes("application/json") ? res.json() : res;
}

function absoluteUrl(path) {
  if (!path) return path;
  return path.startsWith("http") ? path : getBackendUrl() + path;   // a relative path is prefixed with the backend address
}

async function checkBackendStatus() {
  const dot = document.getElementById("statusDot");
  const label = document.getElementById("statusLabel");
  if (!dot) return;
  try {
    await apiFetch("/api/authors");
    dot.className = "status-dot ok";   //the dot turns green once the authors call answers
    label.textContent = "Connected";
  } catch (e) {
    dot.className = "status-dot bad";
    label.textContent = "Not connected";
  }
}

/** Populates any <select data-author-select> element on the page with
 * the ten authors from the backend. Called once per page on load. */
async function populateAuthorDropdowns() {
  const selects = document.querySelectorAll("select[data-author-select]");   // every dropdown marked for the author list
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
        <a href="training.html" class="${activePage === "training" ? "active" : ""}">Training</a>
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

/* ---------------------------------------------------------------------
   Per-page state persistence -- so navigating Read -> Stats -> Read
   doesn't wipe out what you typed/generated. Saved to localStorage (this
   browser only), keyed per page. Call saveState() after anything the
   user would be annoyed to lose; call loadState() once on page load
   before wiring up event listeners; the page provides its own "⟳ Reset"
   button (see resetButtonHtml()/wireResetButton()) for explicitly
   clearing it, since persisting forever-by-default needs an escape hatch.
--------------------------------------------------------------------- */
function savePageState(pageKey, state) {
  try {
    localStorage.setItem("hwrobot_state_" + pageKey, JSON.stringify(state));   //one saved state per page keyed by its name
  } catch (_) {
    // localStorage can throw (quota, private-mode Safari) -- losing
    // persistence silently is fine, it's a convenience, not correctness.
  }
}

function loadPageState(pageKey) {
  try {
    const raw = localStorage.getItem("hwrobot_state_" + pageKey);
    return raw ? JSON.parse(raw) : null;   //nothing saved yet gives null
  } catch (_) {
    return null;
  }
}

function clearPageState(pageKey) {
  try {
    localStorage.removeItem("hwrobot_state_" + pageKey);
  } catch (_) {}
}

/** Wires a page's "⟳ Reset" button (must already be in the DOM with this
 * id) to clear this page's saved state and then run the caller's own
 * onReset() to blank out the visible form/result. */
function wireResetButton(buttonId, pageKey, onReset) {
  const btn = document.getElementById(buttonId);
  if (!btn) return;
  btn.addEventListener("click", () => {
    clearPageState(pageKey);
    onReset();
  });
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

// ---------------------------------------------------------------------------
// Timing panel.  Every endpoint that does real work returns a `timing` object;
// this draws it as a table of stages with their share of the total and their
// cost per character, which is the figure the requirements are written in.
// ---------------------------------------------------------------------------
function renderTiming(containerId, timing) {
  const el = document.getElementById(containerId);
  if (!el) return;
  if (!timing) { el.innerHTML = ""; return; }

  const rows = (timing.stages || []).map((s) => {
    const share = timing.totalSeconds ? (100 * s.seconds) / timing.totalSeconds : 0;   //each stage as a percentage of the total
    return `<tr><td>${s.stage}</td><td class="num secs">${s.seconds.toFixed(3)}</td>
            <td class="num">${share.toFixed(1)}%</td></tr>`;
  });

  // what it took beside what was expected, and nothing else: no allowance
  // formula, no share of the allowance, no pass or fail colour.  a server
  // that sends no limit simply leaves the phrase out
  const expected = timing.limitSeconds   //the allowance is shown only when the server sends one
    ? `&middot; expected ${timing.limitSeconds.toFixed(2)} s` : "";

  el.innerHTML = `
    <h3>Timing</h3>
    <p><strong>${timing.totalSeconds.toFixed(2)} s</strong> total
       ${expected}
       &middot; ${timing.characters} characters
       ${timing.lines ? "&middot; " + timing.lines + " line(s)" : ""}</p>
    <table class="timing-table">
      <thead><tr><th>stage</th><th class="num">seconds</th>
                 <th class="num">share</th></tr></thead>
      <tbody>${rows.join("")}</tbody>
    </table>`;
}
