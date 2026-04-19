/**
 * RFC 8785 JSON Canonicalization Scheme (JCS) — JavaScript port.
 *
 * Byte-identical to ``internal/infrastructure/jcs.py``. The two
 * implementations are cross-verified in ``scripts/verify_cross_runtime.py``
 * against a fixture corpus: every input must produce the same UTF-8 bytes
 * from both runtimes. Any divergence is a correctness regression and is
 * gated in CI before merge.
 *
 * Scope matches the Python side exactly:
 *   object (plain object, string keys)
 *   array
 *   string
 *   integer (finite, !isFloat)
 *   boolean
 *   null
 *
 * Floats are intentionally UNSUPPORTED — the Python impl throws ValueError
 * on them, this impl throws TypeError with the same reasoning: the VC 2.0 +
 * eddsa-jcs-2022 path inside SUM never emits floats (integers and
 * ISO-8601 strings only), so shipping an ES6-ToString serializer for a
 * type we do not use would be untested weight.
 *
 * Key-sort rule (RFC 8785 §3.2.3): UTF-16 code-unit sequence.
 * JavaScript strings ARE UTF-16 natively, so the default lexicographic
 * comparison (Array.prototype.sort with no compareFn) produces exactly
 * the UTF-16-code-unit order. The Python side uses
 * ``s.encode("utf-16-be")`` as its sort key to achieve the same thing on
 * its UTF-8-backed strings. The two approaches yield identical
 * ordering for every Unicode input, verified by the supplementary-
 * character test case (U+1F600 sorts before U+E000).
 *
 * String escaping (RFC 8785 §3.2.2):
 *   "    → \"
 *   \    → \\
 *   \b \t \n \f \r   short-escape forms
 *   other <0x20        \u00XX (LOWERCASE hex)
 *   /                  NOT escaped (RFC 8785 diverges from 8259 here)
 *   everything else    passed through as UTF-8
 *
 * No dependencies. Pure ES2015+. Works in Node ≥ 14 and every modern
 * browser. Intended as the canonicalization substrate for the
 * single-file SUM demo artifact (paste-into-Claude) — provenance IDs
 * and any future signed bundle use it.
 */
'use strict';

/**
 * Canonicalize a JSON-serializable value per RFC 8785.
 *
 * @param {*} obj  supported: object, array, string, integer, boolean, null
 * @returns {Uint8Array}  UTF-8 byte sequence, byte-identical to the
 *                        Python ``canonicalize(obj)`` return.
 */
function canonicalize(obj) {
  const encoder = new TextEncoder();
  return encoder.encode(canonicalizeToStr(obj));
}

/**
 * Canonicalize to a JS string (UTF-16 by default in the engine, but the
 * sequence of code units matches the target canonical form byte-for-byte
 * after UTF-8 re-encoding via canonicalize()).
 *
 * @param {*} obj
 * @returns {string}
 */
function canonicalizeToStr(obj) {
  return _encode(obj);
}

function _encode(obj) {
  if (obj === true) return 'true';
  if (obj === false) return 'false';
  if (obj === null || obj === undefined) return 'null';
  const t = typeof obj;
  if (t === 'string') return _encodeString(obj);
  if (t === 'number') {
    if (!Number.isFinite(obj)) {
      throw new TypeError(
        'JCS: non-finite numbers (NaN, Infinity) are not valid JSON'
      );
    }
    if (!Number.isInteger(obj)) {
      throw new TypeError(
        'JCS: floating-point values are not supported by this implementation ' +
        '(use integer or ISO-8601 string representations instead)'
      );
    }
    return String(obj);
  }
  if (t === 'bigint') {
    return obj.toString();
  }
  if (Array.isArray(obj)) {
    const parts = [];
    for (const item of obj) parts.push(_encode(item));
    return '[' + parts.join(',') + ']';
  }
  if (t === 'object') {
    return _encodeObject(obj);
  }
  throw new TypeError(`JCS: unsupported type ${t}`);
}

function _encodeObject(obj) {
  const keys = Object.keys(obj);
  for (const k of keys) {
    if (typeof k !== 'string') {
      throw new TypeError(`JCS: object keys must be str, got ${typeof k}`);
    }
  }
  // Default string sort in JS is UTF-16 code-unit ordering, which is
  // exactly what RFC 8785 §3.2.3 specifies. No custom compareFn needed.
  keys.sort();
  const parts = [];
  for (const k of keys) {
    parts.push(_encodeString(k) + ':' + _encode(obj[k]));
  }
  return '{' + parts.join(',') + '}';
}

function _encodeString(s) {
  let out = '"';
  for (let i = 0; i < s.length; i++) {
    const cp = s.charCodeAt(i);
    if (cp === 0x22) out += '\\"';
    else if (cp === 0x5C) out += '\\\\';
    else if (cp === 0x08) out += '\\b';
    else if (cp === 0x09) out += '\\t';
    else if (cp === 0x0A) out += '\\n';
    else if (cp === 0x0C) out += '\\f';
    else if (cp === 0x0D) out += '\\r';
    else if (cp < 0x20) {
      out += '\\u' + cp.toString(16).padStart(4, '0');
    } else {
      out += s[i];
    }
  }
  out += '"';
  return out;
}

module.exports = {
  canonicalize,
  canonicalizeToStr,
};
