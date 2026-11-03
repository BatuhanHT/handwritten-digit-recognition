// src/predict.py içindeki "# [adım N] başlık" … "# [/adım N]" işaretlerini okur.
// Demo, her adımın çıktısından önce o adımı yapan gerçek satırları (satır numaralarıyla) gösterir.
// Bir adım birden fazla bölge içerebilir (ör. adım 4: forward_steps ve run içindeki softmax).
const START = /^\s*# \[adım (\d+)\]\s*(.*)$/;
const END = /^\s*# \[\/adım (\d+)\]\s*$/;

function dedent(lines) {
  const indents = lines.filter((l) => l.text.trim()).map((l) => l.text.match(/^ */)[0].length);
  const cut = indents.length ? Math.min(...indents) : 0;
  return lines.map((l) => ({ no: l.no, text: l.text.slice(cut) }));
}

function parseCodeCells(source) {
  const cells = {};
  const open = new Map(); // adım → { start, title }
  source.split(/\r?\n/).forEach((text, i) => {
    const no = i + 1;
    const s = text.match(START);
    const e = text.match(END);
    if (s) {
      if (open.has(s[1])) throw new Error(`adım ${s[1]} kapanmadan yeniden açıldı (satır ${no})`);
      open.set(s[1], { start: no, title: s[2].trim() });
      cells[s[1]] ??= { title: s[2].trim(), regions: [] };
      if (!cells[s[1]].title) cells[s[1]].title = s[2].trim();
    } else if (e) {
      const o = open.get(e[1]);
      if (!o) throw new Error(`adım ${e[1]} açılmadan kapatıldı (satır ${no})`);
      open.delete(e[1]);
      cells[e[1]].regions.push({ from: o.start + 1, to: no - 1, lines: [] });
    }
  });
  if (open.size) throw new Error(`kapanmamış adım: ${[...open.keys()].join(', ')}`);

  const all = source.split(/\r?\n/);
  for (const cell of Object.values(cells)) {
    for (const r of cell.regions) {
      const raw = all.slice(r.from - 1, r.to).map((text, k) => ({ no: r.from + k, text }));
      r.lines = dedent(raw.filter((l) => l.text.trim() !== ''));
    }
  }
  return cells;
}

module.exports = { parseCodeCells };
