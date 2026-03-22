/**
 * SUM Exocortex — Popup Controller
 *
 * Displays the live Gödel state from the local KOS instance and
 * provides one-click crystallization of the active browser tab
 * into the Quantum Knowledge OS.
 */

const API_URL = "http://localhost:8000/api/v1/quantum";

async function fetchState() {
    try {
        const res = await fetch(`${API_URL}/state`);
        if (!res.ok) throw new Error("Offline");
        const data = await res.json();
        const stateStr = data.global_state_integer;
        document.getElementById('godel-state').innerText =
            `Gödel State: ${stateStr.substring(0, 40)}${stateStr.length > 40 ? '...' : ''}`;
        document.getElementById('godel-state').style.color = "#10b981";
        document.getElementById('ingest-btn').disabled = false;
    } catch (e) {
        document.getElementById('godel-state').innerText =
            "Neural Link Offline (Is KOS running on port 8000?)";
        document.getElementById('godel-state').style.color = "#ef4444";
        document.getElementById('ingest-btn').disabled = true;
    }
}

document.getElementById('ingest-btn').addEventListener('click', async () => {
    const btn = document.getElementById('ingest-btn');
    const status = document.getElementById('status');

    btn.disabled = true;
    btn.innerText = "Extracting Tab...";
    status.innerText = "Reading page content...";

    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
        chrome.scripting.executeScript({
            target: {tabId: tabs[0].id},
            func: () => document.body.innerText
        }, async (results) => {
            if (!results || !results[0]) {
                status.innerText = "Failed to read page.";
                status.style.color = "#ef4444";
                btn.disabled = false;
                btn.innerText = "Crystallize Active Tab";
                return;
            }

            const pageText = results[0].result;
            status.innerText = `Ingesting ${pageText.length} characters...`;

            try {
                const res = await fetch(`${API_URL}/ingest`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: pageText, branch: "main" })
                });

                if (!res.ok) throw new Error("API Error");

                const data = await res.json();
                status.innerText = `✓ Crystallized ${data.axioms_ingested} axioms into Gödel State`;
                status.style.color = "#10b981";
                fetchState();
            } catch (e) {
                status.innerText = "Ingestion failed.";
                status.style.color = "#ef4444";
            } finally {
                setTimeout(() => {
                    btn.disabled = false;
                    btn.innerText = "Crystallize Active Tab";
                    status.innerText = "";
                }, 3000);
            }
        });
    });
});

document.addEventListener('DOMContentLoaded', fetchState);