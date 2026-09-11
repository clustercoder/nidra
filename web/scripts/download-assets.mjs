// Downloads all discovered wiz.io page assets into public/.
// Usage: node scripts/download-assets.mjs
import { readFileSync, mkdirSync, writeFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';

const root = new URL('..', import.meta.url).pathname;
const discovery = JSON.parse(readFileSync(join(root, 'docs/research/wiz-assets-discovery2.json'), 'utf8'));

const urls = new Set();
for (const img of discovery.images) {
  const src = img.src || '';
  if (!src.startsWith('http')) continue;
  if (src.includes('cookielaw.org')) continue; // consent widget, not cloned
  urls.add(src.split('?')[0] + (src.includes('?') ? '?' + src.split('?')[1] : ''));
}
urls.add('https://www.wiz.io/favicon.png');

function localPath(url) {
  const u = new URL(url);
  const base = u.pathname.split('/').pop().split('?')[0];
  if (u.hostname === 'www.datocms-assets.com') return join('public/dato', base);
  if (u.pathname === '/favicon.png') return join('public/seo', base);
  return join('public/media', base);
}

const list = [...urls];
console.log(`Downloading ${list.length} assets...`);
const results = { ok: [], fail: [] };

async function fetchOne(url) {
  const rel = localPath(url);
  const abs = join(root, rel);
  if (existsSync(abs)) { results.ok.push({ url, rel, cached: true }); return; }
  try {
    const res = await fetch(url, { headers: { 'user-agent': 'Mozilla/5.0' } });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const buf = Buffer.from(await res.arrayBuffer());
    mkdirSync(dirname(abs), { recursive: true });
    writeFileSync(abs, buf);
    results.ok.push({ url, rel, bytes: buf.length });
  } catch (e) {
    results.fail.push({ url, error: String(e.message || e) });
  }
}

for (let i = 0; i < list.length; i += 4) {
  await Promise.all(list.slice(i, i + 4).map(fetchOne));
  if (i % 40 === 0) console.log(`  ${Math.min(i + 4, list.length)}/${list.length}`);
}

writeFileSync(join(root, 'docs/research/asset-map.json'), JSON.stringify(results, null, 1));
console.log(`Done. ok=${results.ok.length} fail=${results.fail.length}`);
if (results.fail.length) console.log(results.fail.slice(0, 10));
