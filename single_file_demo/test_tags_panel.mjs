import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { JSDOM, VirtualConsole } from 'jsdom';

// The "Tags and back" card: loads index.html with a fetch stub that serves the committed panel JSON (or a
// modified copy) and checks the card renders, the slider switches rows, and data reaches the page as text.
const html = await readFile(new URL('./index.html', import.meta.url), 'utf8');
const panelJson = JSON.parse(await readFile(new URL('./tags_roundtrip_panel.json', import.meta.url), 'utf8'));

async function until(predicate) {
  for (let i = 0; i < 500; i++) { if (predicate()) return; await new Promise(r => setTimeout(r, 2)); }
  assert.fail('Timed out waiting for the tags card');
}

function page(data) {
  const fetchStub = async url => {
    if (String(url).includes('tags_roundtrip_panel.json')) {
      return data === null ? new Response('{}', { status: 404 }) : new Response(JSON.stringify(data), { status: 200 });
    }
    return new Response('{}', { status: 404 });
  };
  const console = new VirtualConsole();
  const dom = new JSDOM(html, { url: 'https://tags.example', runScripts: 'dangerously', virtualConsole: console,
    beforeParse(window) {
      window.fetch = fetchStub;
      window.HTMLElement.prototype.scrollIntoView = () => {};
    } });
  const $ = id => dom.window.document.getElementById(id);
  return { $, window: dom.window, close() { dom.window.close(); } };
}

test('the card renders the recorded run and the slider switches rows', async () => {
  const p = page(panelJson);
  try {
    await until(() => p.$('tags-panel').style.display === '');
    const rows = panelJson.rows;
    assert.equal(p.$('tags-slider').max, String(rows.length - 1));
    assert.equal(p.$('tags-ticks').children.length, rows.length);
    const start = rows[Number(p.$('tags-slider').value)];
    assert.equal(p.$('tags-rewrite').textContent, start.shown_sample.rewrite);
    assert.equal(p.$('tags-used').children.length, start.tags.length);
    assert.equal(p.$('tags-source-text').textContent, panelJson.source.text);
    assert.equal(p.$('tags-source-line').querySelector('a').href, panelJson.source.url);

    const s = start.summary.gist.recall;
    assert.match(p.$('tags-stats').textContent,
      new RegExp(`gist recall ${s.mean.toFixed(2)} \\[${s.min.toFixed(2)}-${s.max.toFixed(2)}\\]`));

    p.$('tags-slider').value = '0';
    p.$('tags-slider').dispatchEvent(new p.window.Event('input'));
    assert.equal(p.$('tags-row-name').textContent, 'control: no tags');
    assert.equal(p.$('tags-rewrite').textContent, rows[0].shown_sample.rewrite);
    assert.match(p.$('tags-used').textContent, /written from no tags/);

    const last = rows.length - 1;
    p.$('tags-slider').value = String(last);
    p.$('tags-slider').dispatchEvent(new p.window.Event('input'));
    assert.equal(p.$('tags-rewrite').textContent, rows[last].shown_sample.rewrite);
    const schools = [...p.$('tags-used').children].map(c => c.firstChild.textContent);
    assert.deepEqual(schools, rows[last].tags.map(t => t.school));
    assert.ok(new Set(schools).size === 3, 'the mixed row labels tags by all three schools');

    // Lists are capped at 6 with a "more" line; the judge select switches between the two judges' lists.
    const gistFirst = p.$('tags-lists').querySelector('li').textContent;
    assert.equal(gistFirst, '✗ ' + rows[last].shown_sample.scores.gist.not_recovered[0]);
    assert.ok(p.$('tags-lists').textContent.includes('more (full list in tags_roundtrip_panel.json)'));
    p.$('tags-judge').value = 'exact';
    p.$('tags-judge').dispatchEvent(new p.window.Event('change'));
    assert.equal(p.$('tags-lists').querySelector('li').textContent,
      '✗ ' + rows[last].shown_sample.scores.exact.not_recovered[0]);
  } finally { p.close(); }
});

test('data strings are inserted as text, never as markup', async () => {
  const evil = '<img src=x onerror="window.__pwned=1">';
  const data = structuredClone(panelJson);
  const i = Number(4);
  data.rows[i].tags[0].tag = evil;
  data.rows[i].tags[0].school = evil;
  data.rows[i].shown_sample.rewrite = evil;
  data.rows[i].shown_sample.scores.gist.not_recovered[0] = evil;
  data.source.text = evil;
  data.source.title = evil;
  data.source.url = 'javascript:window.__pwned=1';
  data.scope = evil;
  const p = page(data);
  try {
    await until(() => p.$('tags-panel').style.display === '');
    p.$('tags-slider').value = String(i);
    p.$('tags-slider').dispatchEvent(new p.window.Event('input'));
    assert.equal(p.$('tags-panel').querySelectorAll('img').length, 0);
    assert.ok(p.$('tags-used').textContent.includes(evil));
    assert.equal(p.$('tags-rewrite').textContent, evil);
    assert.ok(p.$('tags-lists').textContent.includes(evil));
    assert.equal(p.$('tags-source-text').textContent, evil);
    assert.ok(p.$('tags-scope').textContent.includes(evil));
    assert.equal(p.$('tags-source-line').querySelector('a').hasAttribute('href'), false, 'a non-https link is not set');
    assert.equal(p.window.__pwned, undefined);
  } finally { p.close(); }
});

test('the card stays hidden when the data does not load', async () => {
  const p = page(null);
  try {
    await new Promise(r => setTimeout(r, 30));
    assert.equal(p.$('tags-panel').style.display, 'none');
  } finally { p.close(); }
});
