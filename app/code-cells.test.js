// Çalıştırma: cd app && npm test   (Node'un yerleşik test aracı)
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { parseCodeCells } = require('./code-cells');

test('predict.py içinde 4 adımın da kod hücresi var ve boş değil', () => {
  const src = fs.readFileSync(path.join(__dirname, '..', 'src', 'predict.py'), 'utf8');
  const cells = parseCodeCells(src);
  assert.deepEqual(Object.keys(cells).sort(), ['1', '2', '3', '4']);
  for (const [n, cell] of Object.entries(cells)) {
    assert.ok(cell.title, `adım ${n} başlıksız`);
    assert.ok(cell.regions.length >= 1);
    for (const r of cell.regions) assert.ok(r.lines.length > 0, `adım ${n} bölgesi boş`);
  }
  assert.equal(cells['4'].regions.length, 2); // forward_steps + run içindeki softmax
  // satır numaraları dosyadaki gerçek satırları göstermeli (girintisi kaldırılmış hâli birebir)
  const lines = src.split(/\r?\n/);
  for (const [n, cell] of Object.entries(cells)) {
    for (const r of cell.regions) {
      assert.match(lines[r.from - 2], new RegExp(`# \\[adım ${n}\\]`)); // başlık satırı = işaret satırı
      for (const l of r.lines) assert.equal(lines[l.no - 1].trimStart(), l.text.trimStart());
    }
  }
});

test('Windows satır sonlarıyla (CRLF) da aynı sonuç çıkar', () => {
  const src = fs.readFileSync(path.join(__dirname, '..', 'src', 'predict.py'), 'utf8').replace(/\r?\n/g, '\n');
  assert.deepEqual(parseCodeCells(src.replace(/\n/g, '\r\n')), parseCodeCells(src));
});

test('bölge girintisi kaldırılır, boş satırlar atlanır, numaralar korunur', () => {
  const src = ['def f():', '    # [adım 1] Başlık', '    a = 1', '', '        b = 2', '    # [/adım 1]'].join('\n');
  const { 1: cell } = parseCodeCells(src);
  assert.equal(cell.title, 'Başlık');
  assert.deepEqual(cell.regions[0].lines, [{ no: 3, text: 'a = 1' }, { no: 5, text: '    b = 2' }]);
});

test('hatalı işaretler anlaşılır hata verir', () => {
  assert.throws(() => parseCodeCells('# [adım 1] a\nx = 1'), /kapanmamış adım: 1/);
  assert.throws(() => parseCodeCells('x = 1\n# [/adım 2]'), /adım 2 açılmadan kapatıldı/);
});
