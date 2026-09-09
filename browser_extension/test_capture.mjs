import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import fs from 'node:fs';

const code = fs.readFileSync(new URL('./src/popup.js', import.meta.url), 'utf8');
function popup({ selection = 'Alice may cancel the lease.', deny = false, clipboardFail = false } = {}) {
  const elements = Object.fromEntries(['source', 'status', 'capture', 'copy', 'open'].map(id => [id, {
    value: '', textContent: '', focused: false, selected: false,
    addEventListener(type, callback) { this[type] = callback; },
    focus() { this.focused = true; }, select() { this.selected = true; },
  }]));
  const calls = [];
  vm.runInNewContext(code, {
    document: { getElementById: id => elements[id] },
    chrome: {
      tabs: {
        query: async () => { calls.push('query'); return [{ id: 7 }]; },
        create: async value => { calls.push(['open', value.url]); },
      },
      scripting: { executeScript: async options => {
        calls.push(['capture', options.target.tabId]);
        if (deny) throw new Error('restricted');
        return [{ result: selection }];
      } },
    },
    navigator: { clipboard: { writeText: async value => {
      if (clipboardFail) throw new Error('denied');
      calls.push(['copy', value]);
    } } },
  });
  return { elements, calls };
}

test('opening popup does not read or send page text', () => {
  assert.deepEqual(popup().calls, []);
});
test('capture preserves exact text; copy and opening are separate explicit actions', async () => {
  const selected = '  Alice may NOT cancel.\nAmount: $1,800. <script>literal</script>  ';
  const { elements: e, calls } = popup({ selection: selected });
  await e.capture.click();
  assert.equal(e.source.value, selected);
  assert.equal(calls.length, 2);
  await e.copy.click();
  assert.deepEqual(calls.at(-1), ['copy', selected]);
  await e.open.click();
  assert.deepEqual(calls.at(-1), ['open', 'https://sum-demo.ototao.workers.dev/']);
});
test('oversized and empty selections preserve existing source without silent truncation', async () => {
  for (const selection of ['', 'x'.repeat(100001)]) {
    const { elements: e } = popup({ selection });
    e.source.value = 'previous';
    await e.capture.click();
    assert.equal(e.source.value, 'previous');
  }
});
test('restricted page and denied clipboard have usable manual fallback', async () => {
  const { elements: e } = popup({ deny: true, clipboardFail: true });
  await e.capture.click();
  assert.match(e.status.textContent, /does not allow capture/);
  e.source.value = 'manual source';
  await e.copy.click();
  assert.equal(e.source.focused && e.source.selected, true);
});
test('all generated packages reference real files and request only explicit capture permissions', () => {
  for (const browser of ['chrome', 'edge', 'firefox']) {
    const root = new URL(`./${browser}/`, import.meta.url);
    const m = JSON.parse(fs.readFileSync(new URL('manifest.json', root)));
    assert.deepEqual(m.permissions, ['activeTab', 'scripting', 'clipboardWrite']);
    assert.equal(m.host_permissions, undefined);
    assert.equal(m.content_scripts, undefined);
    assert.equal(m.background, undefined);
    assert.ok(fs.existsSync(new URL(m.action.default_popup, root)));
    for (const asset of ['popup.js', 'popup.css']) assert.ok(fs.existsSync(new URL(`src/${asset}`, root)));
  }
});
