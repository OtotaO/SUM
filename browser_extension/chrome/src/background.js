/**
 * SUM Exocortex — Background Service Worker
 *
 * Right-click context menu: "Crystallize to SUM"
 * Highlights any selected text on a webpage and POSTs it to the
 * local Quantum Knowledge OS for immediate Gödel State ingestion.
 */

chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: "crystallize-to-sum",
        title: "Crystallize to SUM (Quantum OS)",
        contexts: ["selection"]
    });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
    if (info.menuItemId === "crystallize-to-sum") {
        const text = info.selectionText;
        if (!text) return;

        try {
            const response = await fetch('http://localhost:8000/api/v1/quantum/ingest', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text, branch: "main" })
            });

            if (response.ok) {
                const data = await response.json();
                console.log("Successfully crystallized into Gödel State:", data.new_global_state);

                chrome.scripting.executeScript({
                    target: { tabId: tab.id },
                    func: () => {
                        const badge = document.createElement('div');
                        badge.innerText = '🧠 Crystallized to SUM';
                        badge.style.cssText = 'position:fixed; bottom:20px; right:20px; background:#10b981; color:#000; padding:10px 15px; border-radius:8px; font-family:monospace; z-index:999999; font-weight:bold; box-shadow: 0 4px 6px rgba(0,0,0,0.3);';
                        document.body.appendChild(badge);
                        setTimeout(() => badge.remove(), 2500);
                    }
                });
            } else {
                throw new Error("KOS returned non-200 status.");
            }
        } catch (error) {
            console.error("Failed to connect to Quantum Knowledge OS:", error);
            chrome.scripting.executeScript({
                target: { tabId: tab.id },
                func: () => {
                    alert("SUM Connection Failed. Is the Quantum Knowledge OS running on localhost:8000?");
                }
            });
        }
    }
});