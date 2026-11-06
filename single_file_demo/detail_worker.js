// Runs the detail checks off the page's main thread, so the page can stop them at a deadline whatever the
// input. handle() is exported for the offline Node tests; in a worker it answers each message.
import { detailFlags } from './detail_flags.js';

export function handle({ id, source, output }) {
  try { return { id, ok: true, result: detailFlags(source, output) }; }
  catch (e) { return { id, ok: false, message: /up to 300/.test(e.message) ? e.message : null }; }
}

if (typeof WorkerGlobalScope !== 'undefined' && self instanceof WorkerGlobalScope) {
  self.onmessage = event => self.postMessage(handle(event.data));
}
