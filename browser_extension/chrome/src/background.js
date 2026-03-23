/**
 * SUM Exocortex — Background Service Worker
 *
 * Horizon I: Continuous Human Digitization
 *
 * Context menu actions:
 *   1. "Crystallize to SUM"     → POST /ingest  (NLP triplet extraction)
 *   2. "Crystallize (Math Only)" → POST /ingest/math  (zero-LLM path)
 *
 * Reads API URL and JWT from chrome.storage.sync.
 * Shows badge with axiom count on successful crystallization.
 */

const DEFAULT_API = "http://localhost:8000/api/v1/quantum";

// ─── Helpers ─────────────────────────────────────────────────────────

async function getConfig() {
    const cfg = await chrome.storage.sync.get(["apiUrl", "jwtToken"]);
    return {
        apiUrl: (cfg.apiUrl || DEFAULT_API).replace(/\/+$/, ""),
        jwtToken: cfg.jwtToken || null
    };
}

function authHeaders(token) {
    const h = { "Content-Type": "application/json" };
    if (token) h["Authorization"] = `Bearer ${token}`;
    return h;
}

// ─── Context Menus ───────────────────────────────────────────────────

chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: "crystallize-full",
        title: "🧠 Crystallize to SUM",
        contexts: ["selection"]
    });
    chrome.contextMenus.create({
        id: "crystallize-math",
        title: "⚡ Crystallize (Math Only — No LLM)",
        contexts: ["selection"]
    });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
    const text = info.selectionText;
    if (!text) return;

    const { apiUrl, jwtToken } = await getConfig();
    const endpoint = info.menuItemId === "crystallize-math"
        ? `${apiUrl}/ingest/math`
        : `${apiUrl}/ingest`;

    try {
        const res = await fetch(endpoint, {
            method: "POST",
            headers: authHeaders(jwtToken),
            body: JSON.stringify({ text, branch: "main" })
        });

        if (!res.ok) throw new Error(`API returned ${res.status}`);

        const data = await res.json();
        const count = data.axioms_ingested || data.axioms_count || "✓";

        // Set badge
        chrome.action.setBadgeText({ text: String(count), tabId: tab.id });
        chrome.action.setBadgeBackgroundColor({ color: "#10b981", tabId: tab.id });

        // Show toast via content script
        chrome.scripting.executeScript({
            target: { tabId: tab.id },
            func: (msg) => {
                const badge = document.createElement("div");
                badge.innerText = msg;
                badge.style.cssText =
                    "position:fixed;bottom:20px;right:20px;background:#10b981;color:#000;" +
                    "padding:12px 18px;border-radius:10px;font-family:monospace;z-index:999999;" +
                    "font-weight:bold;box-shadow:0 4px 12px rgba(0,0,0,0.4);font-size:13px;" +
                    "backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,0.15);";
                document.body.appendChild(badge);
                setTimeout(() => badge.remove(), 3000);
            },
            args: [`🧠 Crystallized ${count} axioms into Gödel State`]
        });

        // Clear badge after 5s
        setTimeout(() => {
            chrome.action.setBadgeText({ text: "", tabId: tab.id });
        }, 5000);

    } catch (error) {
        console.error("SUM Crystallization failed:", error);
        chrome.action.setBadgeText({ text: "!", tabId: tab.id });
        chrome.action.setBadgeBackgroundColor({ color: "#ef4444", tabId: tab.id });

        chrome.scripting.executeScript({
            target: { tabId: tab.id },
            func: (err) => {
                const badge = document.createElement("div");
                badge.innerText = `❌ SUM: ${err}`;
                badge.style.cssText =
                    "position:fixed;bottom:20px;right:20px;background:#ef4444;color:#fff;" +
                    "padding:12px 18px;border-radius:10px;font-family:monospace;z-index:999999;" +
                    "font-weight:bold;box-shadow:0 4px 12px rgba(0,0,0,0.4);font-size:13px;";
                document.body.appendChild(badge);
                setTimeout(() => badge.remove(), 4000);
            },
            args: [error.message]
        });
    }
});

// ─── Message Handler (from popup/content) ────────────────────────────

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg.action === "getConfig") {
        getConfig().then(sendResponse);
        return true; // async
    }
});