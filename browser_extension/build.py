"""Generate unpacked browser packages from one source; --check detects drift."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSETS = ("popup.html", "popup.css", "popup.js")


def package_files(browser: str) -> dict[str, bytes]:
    manifest = {
        "manifest_version": 3,
        "name": "SUM source capture",
        "version": "3.0.0",
        "description": "Capture selected text locally, then copy it into the SUM review workbench.",
        "permissions": ["activeTab", "scripting", "clipboardWrite"],
        "action": {"default_popup": "src/popup.html"},
    }
    if browser == "firefox":
        manifest["browser_specific_settings"] = {
            "gecko": {"id": "sum-capture@ototao.github.io", "strict_min_version": "109.0"}
        }
    return {
        "manifest.json": (json.dumps(manifest, indent=2) + "\n").encode(),
        **{f"src/{name}": (ROOT / "src" / name).read_bytes() for name in ASSETS},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    drift = []
    for browser in ("chrome", "edge", "firefox"):
        for name, content in package_files(browser).items():
            path = ROOT / browser / name
            if args.check:
                if not path.exists() or path.read_bytes() != content:
                    drift.append(str(path.relative_to(ROOT)))
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
    if drift:
        print("Regenerate browser packages: " + ", ".join(drift))
        return 1
    print("Browser packages are current." if args.check else "Generated Chrome, Edge and Firefox packages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
