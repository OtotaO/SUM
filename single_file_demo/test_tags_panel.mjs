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
    const text = panelJson.texts[0];
    const rows = text.rows;
    assert.equal(p.$('tags-text').children.length, panelJson.texts.length);
    assert.equal(p.$('tags-slider').max, String(rows.length - 1));
    assert.equal(p.$('tags-ticks').children.length, rows.length);
    const start = rows[Number(p.$('tags-slider').value)];
    assert.equal(p.$('tags-rewrite').textContent, start.shown_sample.rewrite);
    assert.equal(p.$('tags-used').children.length, start.tags.length);
    assert.equal(p.$('tags-source-text').textContent, text.source.text);
    assert.equal(p.$('tags-source-line').querySelector('a').href, text.source.url);
    assert.equal(p.$('tags-cut').style.display, 'none', 'the first text has no cut');

    const s = start.summary.gist.recall;
    assert.match(p.$('tags-stats').textContent,
      new RegExp(`gist recall ${s.mean.toFixed(2)} \\[${s.min.toFixed(2)}-${s.max.toFixed(2)}\\]`));
    // Row 4 is the 32 topic tags: the tags-alone figures appear there and nowhere else.
    assert.deepEqual([start.condition, start.tag_count], ['topic', 32]);
    assert.ok(p.$('tags-stats').textContent.includes('tags alone: gist recall ' + text.tags_alone.gist_recall.toFixed(2)));

    p.$('tags-slider').value = '0';
    p.$('tags-slider').dispatchEvent(new p.window.Event('input'));
    assert.equal(p.$('tags-row-name').textContent, 'control: no tags');
    assert.equal(p.$('tags-rewrite').textContent, rows[0].shown_sample.rewrite);
    assert.match(p.$('tags-used').textContent, /written from no tags/);
    assert.ok(!p.$('tags-stats').textContent.includes('tags alone'));

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

test('the second text shows its own rows, notes and the cut', async () => {
  const p = page(panelJson);
  try {
    await until(() => p.$('tags-panel').style.display === '');
    const text = panelJson.texts[1];
    const rows = text.rows;
    p.$('tags-text').value = '1';
    p.$('tags-text').dispatchEvent(new p.window.Event('change'));
    assert.equal(p.$('tags-source-text').textContent, text.source.text);
    assert.equal(p.$('tags-source-line').querySelector('a').href, text.source.url);
    assert.ok(p.$('tags-text-notes').textContent.includes('played no part in choosing the judge settings'));
    const tw = rows.find(r => r.condition === 'topic' && r.tag_count === 32).tag_words;
    const mw = rows.find(r => r.condition === 'mixed' && r.tag_count === 32).tag_words;
    assert.ok(p.$('tags-text-notes').textContent.includes(`mixed tags carry ${mw} words against ${tw}`));
    assert.equal(p.$('tags-cut-note').textContent, text.cut.note);

    // The 32 topic tags: cut shown, and its exact recall is above the full rewrite's, so the card says so.
    const i = rows.findIndex(r => r.condition === 'topic' && r.tag_count === 32);
    p.$('tags-slider').value = String(i);
    p.$('tags-slider').dispatchEvent(new p.window.Event('input'));
    const r = rows[i];
    assert.equal(p.$('tags-cut').style.display, '');
    assert.equal(p.$('tags-cut-text').textContent, r.cut.shown.text);
    assert.ok(p.$('tags-cut-stats').textContent.includes(`seed 7 kept ${r.cut.shown.kept_sentences} of ${r.cut.shown.rewrite_sentences} sentences`));
    assert.ok(r.cut.summary.exact.recall.mean > r.summary.exact.recall.mean);
    assert.match(p.$('tags-cut-warn').textContent, /should not do/);
    assert.ok(p.$('tags-stats').textContent.includes('tags alone: gist recall ' + text.tags_alone.gist_recall.toFixed(2)));

    // A row whose seed 7 cut kept nothing says so, and the control row has no cut.
    const empty = rows.findIndex(x => x.cut && x.cut.shown.kept_sentences === 0);
    assert.ok(empty > 0);
    p.$('tags-slider').value = String(empty);
    p.$('tags-slider').dispatchEvent(new p.window.Event('input'));
    assert.equal(p.$('tags-cut-text').textContent, '(the seed 7 cut kept no sentence)');
    assert.equal(p.$('tags-cut-warn').textContent, '');
    p.$('tags-slider').value = '0';
    p.$('tags-slider').dispatchEvent(new p.window.Event('input'));
    assert.equal(p.$('tags-cut').style.display, 'none');

    // Back to the first text: its rows and no cut.
    p.$('tags-text').value = '0';
    p.$('tags-text').dispatchEvent(new p.window.Event('change'));
    assert.equal(p.$('tags-source-text').textContent, panelJson.texts[0].source.text);
    assert.ok(p.$('tags-text-notes').textContent.includes('This run did not score the cut.'));
    assert.equal(p.$('tags-cut').style.display, 'none');
  } finally { p.close(); }
});

test('data strings are inserted as text, never as markup', async () => {
  const evil = '<img src=x onerror="window.__pwned=1">';
  const data = structuredClone(panelJson);
  const i = Number(4);
  const t = data.texts[0];
  t.rows[i].tags[0].tag = evil;
  t.rows[i].tags[0].school = evil;
  t.rows[i].shown_sample.rewrite = evil;
  t.rows[i].shown_sample.scores.gist.not_recovered[0] = evil;
  t.source.text = evil;
  t.source.title = evil;
  t.source.url = 'javascript:window.__pwned=1';
  t.scope = evil;
  t.notes = [evil];
  t.cut = { note: evil };
  t.rows[i].cut = { summary: { nonempty: 1, gist: {}, exact: {} }, shown: { text: evil, kept_sentences: 1, rewrite_sentences: 2, scores: null } };
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
    assert.ok(p.$('tags-text-notes').textContent.includes(evil));
    assert.equal(p.$('tags-cut-text').textContent, evil);
    assert.equal(p.$('tags-cut-note').textContent, evil);
    assert.ok(p.$('tags-text').textContent.includes(evil));
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
