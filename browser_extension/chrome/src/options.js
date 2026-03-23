/**
 * SUM Exocortex — Options Controller
 *
 * Settings: API URL, JWT Token
 * Connection testing against /state endpoint
 */

const DEFAULT_SETTINGS = {
    apiUrl: "http://localhost:8000/api/v1/quantum",
    jwtToken: ""
};

const elements = {};

document.addEventListener("DOMContentLoaded", () => {
    // Bind elements
    elements.apiUrl = document.getElementById("apiUrl");
    elements.jwtToken = document.getElementById("jwtToken");
    elements.testConnectionBtn = document.getElementById("testConnectionBtn");
    elements.connectionResult = document.getElementById("connectionResult");
    elements.configureShortcutsBtn = document.getElementById("configureShortcutsBtn");
    elements.crystallizeCount = document.getElementById("crystallizeCount");
    elements.connStatus = document.getElementById("connStatus");
    elements.exportDataBtn = document.getElementById("exportDataBtn");
    elements.saveBtn = document.getElementById("saveBtn");
    elements.resetBtn = document.getElementById("resetBtn");

    loadSettings();
    setupEventListeners();
});

async function loadSettings() {
    const settings = await chrome.storage.sync.get(Object.keys(DEFAULT_SETTINGS));
    elements.apiUrl.value = settings.apiUrl || DEFAULT_SETTINGS.apiUrl;
    elements.jwtToken.value = settings.jwtToken || DEFAULT_SETTINGS.jwtToken;

    // Load crystallization count
    const local = await chrome.storage.local.get(["crystallizeCount"]);
    elements.crystallizeCount.textContent = local.crystallizeCount || 0;
}

function setupEventListeners() {
    elements.testConnectionBtn.addEventListener("click", testConnection);

    elements.configureShortcutsBtn.addEventListener("click", () => {
        chrome.tabs.create({ url: "chrome://extensions/shortcuts" });
    });

    elements.exportDataBtn.addEventListener("click", exportData);
    elements.saveBtn.addEventListener("click", saveSettings);

    elements.resetBtn.addEventListener("click", async () => {
        if (confirm("Reset all settings to defaults?")) {
            await chrome.storage.sync.clear();
            await loadSettings();
            showMessage("Settings reset to defaults");
        }
    });
}

async function testConnection() {
    const apiUrl = elements.apiUrl.value.trim().replace(/\/+$/, "");
    if (!apiUrl) {
        showConnectionResult("error", "Please enter an API URL");
        return;
    }

    elements.testConnectionBtn.disabled = true;
    elements.testConnectionBtn.textContent = "Testing…";

    try {
        const headers = { "Content-Type": "application/json" };
        const token = elements.jwtToken.value.trim();
        if (token) headers["Authorization"] = `Bearer ${token}`;

        const res = await fetch(`${apiUrl}/state`, {
            method: "GET",
            headers,
            signal: AbortSignal.timeout(5000)
        });

        if (res.ok) {
            const data = await res.json();
            const axioms = data.axiom_count ?? data.total_axioms ?? "?";
            showConnectionResult("success",
                `✓ Connected! Axioms: ${axioms}, Tick: ${data.tick ?? data.current_tick ?? "?"}`);
            elements.connStatus.textContent = "Online";
            elements.connStatus.style.color = "#10b981";
        } else {
            showConnectionResult("error", `Connection failed: HTTP ${res.status}`);
            elements.connStatus.textContent = "Error";
            elements.connStatus.style.color = "#ef4444";
        }
    } catch (error) {
        showConnectionResult("error", `Connection failed: ${error.message}`);
        elements.connStatus.textContent = "Offline";
        elements.connStatus.style.color = "#ef4444";
    } finally {
        elements.testConnectionBtn.disabled = false;
        elements.testConnectionBtn.textContent = "Test Connection";
    }
}

function showConnectionResult(type, message) {
    elements.connectionResult.className = `connection-result ${type}`;
    elements.connectionResult.textContent = message;
    setTimeout(() => {
        elements.connectionResult.className = "connection-result";
    }, 5000);
}

async function saveSettings() {
    const settings = {
        apiUrl: elements.apiUrl.value.trim(),
        jwtToken: elements.jwtToken.value.trim()
    };

    try {
        await chrome.storage.sync.set(settings);
        showMessage("Settings saved successfully");
    } catch (error) {
        showMessage("Failed to save settings", "error");
    }
}

async function exportData() {
    try {
        const data = {
            settings: await chrome.storage.sync.get(),
            exportDate: new Date().toISOString()
        };
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `sum-exocortex-config-${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
        showMessage("Config exported successfully");
    } catch (error) {
        showMessage("Failed to export config", "error");
    }
}

function showMessage(text, type = "success") {
    const msg = document.createElement("div");
    msg.className = "success-message";
    msg.textContent = text;
    if (type === "error") msg.style.background = "#dc2626";
    document.body.appendChild(msg);
    setTimeout(() => msg.remove(), 3000);
}