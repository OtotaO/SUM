/* User-triggered, local source capture. No background process or network calls. */
'use strict';
const extension = globalThis.browser || globalThis.chrome;
const source = document.getElementById('source');
const status = document.getElementById('status');
const MAX_SOURCE_CHARS = 100000;

document.getElementById('capture').addEventListener('click', async () => {
  status.textContent = 'Reading your selection...';
  try {
    const [tab] = await extension.tabs.query({ active: true, currentWindow: true });
    if (!Number.isInteger(tab?.id)) throw new Error('No active page');
    const results = await extension.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => globalThis.getSelection()?.toString() || '',
    });
    const text = results[0]?.result;
    if (typeof text !== 'string' || !text.trim()) {
      status.textContent = 'Select text on the page first, or paste it here.';
      return;
    }
    if (text.length > MAX_SOURCE_CHARS) {
      status.textContent = 'Selection exceeds 100,000 characters. Choose a smaller passage; nothing was truncated.';
      return;
    }
    source.value = text;
    status.textContent = 'Selection captured locally. Copy it, then open the workbench.';
  } catch {
    status.textContent = 'This page does not allow capture. Copy the text yourself and paste it here.';
  }
});

document.getElementById('copy').addEventListener('click', async () => {
  if (!source.value.trim()) {
    status.textContent = 'Capture or paste a source first.';
    return;
  }
  try {
    await navigator.clipboard.writeText(source.value);
    status.textContent = 'Source copied. Open the workbench and paste it into Source.';
  } catch {
    source.focus();
    source.select();
    status.textContent = 'Clipboard access failed. Use your keyboard to copy the selected source.';
  }
});

document.getElementById('open').addEventListener('click', async () => {
  try {
    await extension.tabs.create({ url: 'https://sum-demo.ototao.workers.dev/' });
  } catch {
    status.textContent = 'Could not open the workbench. Visit sum-demo.ototao.workers.dev.';
  }
});
