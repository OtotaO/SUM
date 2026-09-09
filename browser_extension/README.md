# SUM source capture

Capture selected text locally, copy it, then paste it into the [SUM workbench](https://sum-demo.ototao.workers.dev/) for review. This client does not summarize or judge text itself.

## Use

1. Select text on a normal web page and open the extension popup.
2. Click **Capture selected text**. Inspect or edit the captured text.
3. Click **Copy source**, then **Open SUM workbench**, and paste into Source.

Capture is explicit. There is no background collection, content script on every page, page-history storage, API key, or request carrying your text. Closing the popup discards its content. Copying writes to the system clipboard; the workbench handles text under its own displayed generation and review controls. Browser-internal pages, some embedded frames, and browser stores may deny capture; manual copy/paste remains available. Selections over 100,000 characters are rejected without truncation.

## Install from this checkout

Chrome: open `chrome://extensions`, enable Developer mode, choose **Load unpacked**, and select `browser_extension/chrome`. Edge uses the same steps at `edge://extensions` with `browser_extension/edge`.

Firefox: open `about:debugging`, choose **This Firefox > Load Temporary Add-on**, then select `browser_extension/firefox/manifest.json`. This is a development package, not a signed store release. Cross-browser package generation and mocked API tests do not establish a browser-store approval or a manual test on every browser.

## Maintain

`src/` is the single maintained popup source. Regenerate packages with `python browser_extension/build.py`; CI checks them using `--check`. Run behavior tests with `node --test browser_extension/test_capture.mjs`.

Permissions are limited to `activeTab`, `scripting`, and `clipboardWrite`. Selection reads use temporary page access after the user invokes the extension; see the [Chrome scripting contract](https://developer.chrome.com/docs/extensions/reference/api/scripting) and [Firefox scripting contract](https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/API/scripting/executeScript).

The earlier Exocortex research API client and incomplete instant-summarization manifests are preserved in git history before this change. The current extension requires no local server and does not promote the internal quantum API to a shipped service.

Apache-2.0; see the repository LICENSE.
