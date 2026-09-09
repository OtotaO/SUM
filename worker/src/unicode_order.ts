// Python compares Unicode code points; JS < compares UTF-16 code units. These
// differ between supplementary characters and BMP characters above U+DFFF.
export function compareUnicodeCodepoints(a: string, b: string): number {
  let ai = 0, bi = 0;
  while (ai < a.length && bi < b.length) {
    const ac = a.codePointAt(ai)!;
    const bc = b.codePointAt(bi)!;
    if (ac !== bc) return ac < bc ? -1 : 1;
    ai += ac > 0xffff ? 2 : 1;
    bi += bc > 0xffff ? 2 : 1;
  }
  return ai < a.length ? 1 : bi < b.length ? -1 : 0;
}

export function compareTriples(a: [string, string, string], b: [string, string, string]): number {
  for (let i = 0; i < 3; i++) {
    const order = compareUnicodeCodepoints(a[i], b[i]);
    if (order) return order;
  }
  return 0;
}
