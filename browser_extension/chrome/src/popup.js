/**
 * SUM Exocortex — Popup Controller
 *
 * Gödel State Dashboard:
 *   - Live state integer (truncated BigInt)
 *   - Axiom count, bit length, tick number
 *   - Machine discoveries feed
 *   - Crystallize Active Tab button
 *   - Sync State button (browser → server merge)
 */

const DEFAULT_API = "http://localhost:8000/api/v1/quantum";

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

// ─── State Fetcher ───────────────────────────────────────────────────

async function fetchState() {
    const dot = document.getElementById("status-dot");
    try {
        const { apiUrl, jwtToken } = await getConfig();

        const res = await fetch(`${apiUrl}/state`, {
            headers: authHeaders(jwtToken)
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        // State integer
        const stateStr = String(data.global_state_integer || "1");
        const el = document.getElementById("godel-state");
        el.innerText = stateStr.length > 60
            ? `${stateStr.substring(0, 30)}…${stateStr.substring(stateStr.length - 20)}`
            : stateStr;

        // Metrics
        document.getElementById("axiom-count").innerText =
            data.axiom_count ?? data.total_axioms ?? "—";
        document.getElementById("bit-length").innerText =
            data.bit_length ?? (stateStr === "1" ? "0" : stateStr.length) ?? "—";
        document.getElementById("tick").innerText =
            data.tick ?? data.current_tick ?? "—";

        // Online
        dot.classList.remove("offline");
        dot.classList.add("online");
        dot.title = "Connected";
        document.getElementById("ingest-btn").disabled = false;
        document.getElementById("sync-btn").disabled = false;

        // Fetch discoveries
        fetchDiscoveries(apiUrl, jwtToken);

    } catch (e) {
        document.getElementById("godel-state").innerText =
            "Neural Link Offline";
        dot.classList.remove("online");
        dot.classList.add("offline");
        dot.title = e.message;
        document.getElementById("ingest-btn").disabled = true;
        document.getElementById("sync-btn").disabled = true;
    }
}

// ─── Discoveries ─────────────────────────────────────────────────────

async function fetchDiscoveries(apiUrl, token) {
    try {
        const res = await fetch(`${apiUrl}/discoveries`, {
            headers: authHeaders(token)
        });
        if (!res.ok) return;
        const data = await res.json();
        const items = data.recent || [];

        if (items.length === 0) return;

        const panel = document.getElementById("discoveries-panel");
        const list = document.getElementById("discoveries-list");
        panel.style.display = "block";
        list.innerHTML = items.slice(0, 5).map(d =>
            `<div class="discovery-item">${d.subject} <span class="pred">${d.predicate}</span> ${d.object}</div>`
        ).join("");
    } catch (e) {
        // Silent — discoveries are optional
    }
}

// ─── Crystallize Active Tab ──────────────────────────────────────────

document.getElementById("ingest-btn").addEventListener("click", async () => {
    const btn = document.getElementById("ingest-btn");
    const status = document.getElementById("status");
    btn.disabled = true;
    btn.innerText = "Extracting…";
    status.innerText = "Reading page content…";
    status.className = "";

    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        chrome.scripting.executeScript({
            target: { tabId: tabs[0].id },
            func: () => {
                // Smart content extraction
                const selectors = ["main", "article", '[role="main"]', ".content", "#content"];
                for (const s of selectors) {
                    const el = document.querySelector(s);
                    if (el) return `${document.title}\n\n${el.innerText}`;
                }
                return `${document.title}\n\n${document.body.innerText}`;
            }
        }, async (results) => {
            if (!results || !results[0]) {
                status.innerText = "Failed to read page.";
                status.className = "status-error";
                btn.disabled = false;
                btn.innerText = "Crystallize Tab";
                return;
            }

            const pageText = results[0].result;
            status.innerText = `Ingesting ${pageText.length.toLocaleString()} chars…`;

            try {
                const { apiUrl, jwtToken } = await getConfig();
                const res = await fetch(`${apiUrl}/ingest`, {
                    method: "POST",
                    headers: authHeaders(jwtToken),
                    body: JSON.stringify({ text: pageText, branch: "main" })
                });

                if (!res.ok) throw new Error(`API Error ${res.status}`);
                const data = await res.json();

                status.innerText = `✓ Crystallized ${data.axioms_ingested ?? "?"} axioms`;
                status.className = "status-success";
                fetchState();
            } catch (e) {
                status.innerText = `Ingestion failed: ${e.message}`;
                status.className = "status-error";
            } finally {
                setTimeout(() => {
                    btn.disabled = false;
                    btn.innerText = "Crystallize Tab";
                }, 2500);
            }
        });
    });
});

// ─── Sync State ──────────────────────────────────────────────────────

document.getElementById("sync-btn").addEventListener("click", async () => {
    const btn = document.getElementById("sync-btn");
    const status = document.getElementById("status");
    btn.disabled = true;
    btn.innerText = "Syncing…";

    try {
        const { apiUrl, jwtToken } = await getConfig();
        const stateRes = await fetch(`${apiUrl}/state`, {
            headers: authHeaders(jwtToken)
        });
        if (!stateRes.ok) throw new Error("Could not fetch state");
        const stateData = await stateRes.json();

        const syncRes = await fetch(`${apiUrl}/sync/state`, {
            method: "POST",
            headers: authHeaders(jwtToken),
            body: JSON.stringify({
                peer_state_integer: String(stateData.global_state_integer)
            })
        });

        if (!syncRes.ok) throw new Error(`Sync failed: ${syncRes.status}`);
        const syncData = await syncRes.json();

        status.innerText = `✓ Synced — ${syncData.status || "merged"}`;
        status.className = "status-success";
        fetchState();
    } catch (e) {
        status.innerText = `Sync failed: ${e.message}`;
        status.className = "status-error";
    } finally {
        setTimeout(() => {
            btn.disabled = false;
            btn.innerText = "Sync State";
        }, 2500);
    }
});

// ─── Boot ────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", fetchState);