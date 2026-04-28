// scripts/vendor/build.js
//
// Regenerates the vendored `single_file_demo/vendor/sum-verify-deps.js`
// bundle from this directory's package-lock.json-pinned versions of
// `jose` and `canonicalize`. Two callers:
//
//   - Human:  `npm install && npm run build` after bumping deps.
//             Then `git add single_file_demo/vendor/ && commit`.
//   - CI:     vendor-byte-equivalence job in quantum-ci.yml. Runs
//             `npm install` from this directory's package-lock.json,
//             rebuilds the bundle, sha256-compares against the
//             committed bytes. Drift fails the gate with an explicit
//             remediation pointer (mirrors the zig-core .wasm gate
//             pattern).
//
// Output is one ESM file (`sum-verify-deps.js`) re-exporting the
// minimal verifier surface from both libraries. A second LICENSE
// file is emitted alongside it carrying both upstream LICENSE bodies
// concatenated, per Apache-2.0 + MIT attribution requirements.

import esbuild from "esbuild";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, "..", "..");
const outDir = resolve(repoRoot, "single_file_demo", "vendor");
mkdirSync(outDir, { recursive: true });

// Read pinned versions from this dir's package.json so the banner
// stamps the actual versions consumed.
const pkg = JSON.parse(
  readFileSync(resolve(__dirname, "package.json"), "utf8"),
);
const joseVersion = pkg.dependencies.jose;
const canonicalizeVersion = pkg.dependencies.canonicalize;

const banner = [
  "// Vendored bundle for sum_engine_internal browser receipt verifier (Phase E.1 v0.9.B).",
  "// Regenerated via `scripts/vendor/build.js`; CI verifies byte-equivalence.",
  "// DO NOT EDIT BY HAND.",
  `// jose@${joseVersion}        — MIT (panva). Filip Skokan and contributors.`,
  `// canonicalize@${canonicalizeVersion} — Apache-2.0 (Erdtman). Anders Rundgren and contributors.`,
  "// Full LICENSE bodies in `single_file_demo/vendor/LICENSE.txt`.",
].join("\n");

await esbuild.build({
  entryPoints: [resolve(__dirname, "entry-sum-verify-deps.js")],
  bundle: true,
  format: "esm",
  target: "es2020",
  platform: "browser",
  // Mangle private/internal names but keep public API intact. We
  // want a small but readable bundle; full minification with
  // identifier shortening would make CI byte-comparison brittle
  // across esbuild versions and obscure debugging in DevTools.
  minify: false,
  legalComments: "none",
  banner: { js: banner },
  outfile: resolve(outDir, "sum-verify-deps.js"),
});

// Concatenate upstream LICENSE files for attribution. Both are short.
const joseLicense = readFileSync(
  resolve(__dirname, "node_modules", "jose", "LICENSE.md"),
  "utf8",
);
const canonicalizeLicense = readFileSync(
  resolve(__dirname, "node_modules", "canonicalize", "LICENSE"),
  "utf8",
);
const combined = [
  "Vendored upstream LICENSE attributions for single_file_demo/vendor/sum-verify-deps.js",
  "=====================================================================================",
  "",
  `jose@${joseVersion} — https://github.com/panva/jose`,
  "-------------------------------------------------------------------------------------",
  "",
  joseLicense.trim(),
  "",
  "",
  `canonicalize@${canonicalizeVersion} — https://github.com/erdtman/canonicalize`,
  "-------------------------------------------------------------------------------------",
  "",
  canonicalizeLicense.trim(),
  "",
].join("\n");

writeFileSync(resolve(outDir, "LICENSE.txt"), combined);

console.log(`vendored: ${resolve(outDir, "sum-verify-deps.js")}`);
console.log(`vendored: ${resolve(outDir, "LICENSE.txt")}`);
console.log(`  jose@${joseVersion}, canonicalize@${canonicalizeVersion}`);
