/**
 * SUM Exocortex — Content Script
 *
 * Injected into every page. Provides:
 *   1. Floating "Crystallize" button on text selection (>50 chars)
 *   2. Crystallization result overlay (axioms ingested, state)
 *   3. Message handler for background script communication
 */

// ─── State ───────────────────────────────────────────────────────────

let selectionButton = null;
let resultOverlay = null;

// ─── Init ────────────────────────────────────────────────────────────

(function () {
    // Message handler
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
        switch (request.action) {
            case "getSelection":
                sendResponse({ text: window.getSelection().toString() });
                break;
            case "getPageContent":
                sendResponse({ content: extractPageContent() });
                break;
            case "showResult":
                showCrystallizationResult(request.data);
                break;
            case "showError":
                showErrorToast(request.error);
                break;
        }
    });

    // Floating crystallize button on selection
    document.addEventListener("mouseup", (e) => {
        const text = window.getSelection().toString().trim();
        if (text.length > 50) {
            showSelectionButton(e.pageX, e.pageY, text);
        } else {
            hideSelectionButton();
        }
    });

    document.addEventListener("mousedown", (e) => {
        if (!e.target.closest(".sum-crystallize-btn")) {
            hideSelectionButton();
        }
    });
})();

// ─── Page Content Extraction ─────────────────────────────────────────

function extractPageContent() {
    const selectors = [
        "main", "article", '[role="main"]',
        ".content", "#content", ".post",
        ".article-body", ".entry-content"
    ];

    let content = "";
    for (const selector of selectors) {
        const el = document.querySelector(selector);
        if (el) {
            content = el.innerText;
            break;
        }
    }

    if (!content) content = document.body.innerText;
    content = content.replace(/\s+/g, " ").trim();

    const title = document.title;
    if (title) content = `${title}\n\n${content}`;

    return content;
}

// ─── Floating Crystallize Button ─────────────────────────────────────

function showSelectionButton(x, y, text) {
    if (!selectionButton) {
        selectionButton = document.createElement("div");
        selectionButton.className = "sum-crystallize-btn";
        selectionButton.innerHTML = `
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
            </svg>
            <span>Crystallize</span>
        `;
        selectionButton.style.cssText =
            "position:absolute;display:none;align-items:center;gap:6px;" +
            "background:#2563eb;color:#fff;padding:6px 12px;border-radius:8px;" +
            "font-family:-apple-system,system-ui,sans-serif;font-size:12px;font-weight:600;" +
            "cursor:pointer;z-index:999999;box-shadow:0 4px 12px rgba(37,99,235,0.4);" +
            "border:1px solid rgba(255,255,255,0.15);transition:transform 0.15s;" +
            "backdrop-filter:blur(4px);";
        document.body.appendChild(selectionButton);

        selectionButton.addEventListener("mouseenter", () => {
            selectionButton.style.transform = "scale(1.05)";
        });
        selectionButton.addEventListener("mouseleave", () => {
            selectionButton.style.transform = "scale(1)";
        });

        selectionButton.addEventListener("click", () => {
            hideSelectionButton();
            chrome.runtime.sendMessage({
                action: "summarize",   // handled by background
                text: text,
                type: "selection"
            });
        });
    }

    selectionButton.style.left = `${x + 10}px`;
    selectionButton.style.top = `${y - 40}px`;
    selectionButton.style.display = "flex";
}

function hideSelectionButton() {
    if (selectionButton) {
        selectionButton.style.display = "none";
    }
}

// ─── Crystallization Result Overlay ──────────────────────────────────

function showCrystallizationResult(data) {
    if (resultOverlay) resultOverlay.remove();

    resultOverlay = document.createElement("div");
    resultOverlay.style.cssText =
        "position:fixed;top:20px;right:20px;width:320px;background:#0f172a;" +
        "border:1px solid #1e293b;border-radius:12px;padding:16px;z-index:999999;" +
        "font-family:-apple-system,system-ui,sans-serif;color:#e2e8f0;" +
        "box-shadow:0 8px 24px rgba(0,0,0,0.5);backdrop-filter:blur(12px);";

    resultOverlay.innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
            <span style="font-size:14px;font-weight:700;background:linear-gradient(135deg,#38bdf8,#818cf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                🧠 Crystallized
            </span>
            <span id="sum-close" style="cursor:pointer;color:#64748b;font-size:18px;">&times;</span>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px;">
            <div style="background:#1e293b;padding:8px;border-radius:8px;text-align:center;">
                <div style="font-size:20px;font-weight:700;color:#10b981;">${data.axioms_ingested ?? "?"}</div>
                <div style="font-size:9px;color:#64748b;text-transform:uppercase;">Axioms</div>
            </div>
            <div style="background:#1e293b;padding:8px;border-radius:8px;text-align:center;">
                <div style="font-size:20px;font-weight:700;color:#38bdf8;">${data.new_axioms ?? "?"}</div>
                <div style="font-size:9px;color:#64748b;text-transform:uppercase;">New</div>
            </div>
        </div>
        <div style="font-size:10px;color:#64748b;font-family:monospace;word-break:break-all;">
            State: ${String(data.new_global_state || "").substring(0, 40)}…
        </div>
    `;

    document.body.appendChild(resultOverlay);

    resultOverlay.querySelector("#sum-close").addEventListener("click", () => {
        resultOverlay.remove();
        resultOverlay = null;
    });

    setTimeout(() => {
        if (resultOverlay) {
            resultOverlay.remove();
            resultOverlay = null;
        }
    }, 8000);
}

// ─── Error Toast ─────────────────────────────────────────────────────

function showErrorToast(error) {
    const toast = document.createElement("div");
    toast.style.cssText =
        "position:fixed;bottom:20px;right:20px;background:#ef4444;color:#fff;" +
        "padding:12px 18px;border-radius:10px;font-family:monospace;z-index:999999;" +
        "font-size:12px;box-shadow:0 4px 12px rgba(0,0,0,0.4);";
    toast.innerText = `❌ SUM: ${error}`;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}