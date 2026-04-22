# Contributing to SUM

First off, **THANK YOU** for considering contributing to SUM! This project exists because of amazing people like you who believe knowledge tools should be free for everyone.

## 🌟 Core Principles

1. **Free Forever**: No paywalls, ever
2. **Privacy First**: User data never leaves their machine
3. **Universal Access**: Works for everyone, everywhere
4. **Beautiful Code**: Clean, readable, and well-documented
5. **Radical Innovation**: No idea is too wild

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/OtotaO/SUM.git
cd SUM

# Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies (or: `make install` does all of this in one shot)
pip install -e '.[sieve,dev]'
python -m spacy download en_core_web_sm

# One-time: enable the PORTFOLIO.md warn-hook for local commits
make install-hooks

# Run the test suite (1000+ tests; see Makefile for shortcuts)
make test
# Fast inner loop (CLI + codec + VC only):
make test-cli

# Run the 21-check Fortress gate
python scripts/verify_fortress.py --json
```

### Optional: Build the Zig Core (Bare-Metal Mode)

```bash
# Install Zig (macOS)
brew install zig

# Build the shared library
cd core-zig && zig build -Doptimize=ReleaseFast && cd ..

# Verify: the Python tests will now route through Zig C-ABI
python -m pytest Tests/ -v  # Look for "⚡ BARE-METAL ZIG CORE ENGAGED ⚡"
```

## 🛡️ Verification Gates

**All PRs must pass before merge:**

| Gate | Command | Expected |
|------|---------|----------|
| Test Suite | `make test` (or `python -m pytest Tests/ -v`) | 1000+ passed |
| Fortress | `make fortress` | 21/21 |
| Zig Tests | `cd core-zig && zig build test` | All pass |
| Cross-Runtime Harness | `make xruntime` | K1/K1-mw/K2/K3/K4 all PASS |
| PORTFOLIO contract | `make portfolio` | OK, every metric row labelled |
| pip install smoke | `make wheel && make smoke` | attest\|verify round-trip ✓ |

## 📁 Project Structure

The codebase follows a clean architecture:

```
internal/
├── algorithms/     # Core math (semantic_arithmetic, syntactic_sieve)
├── ensemble/       # Higher-order systems (arbiter, ouroboros, triggers)
├── infrastructure/ # Persistence, FFI, transport (akashic, zig_bridge, codec)
core-zig/           # Bare-metal Zig core (C-ABI exports)
sum_cli/            # `sum` CLI (attest / verify / resolve)
standalone_verifier/ # Node.js verifier (zero npm deps)
single_file_demo/   # Browser demo (SubtleCrypto Ed25519 verify)
Tests/              # 1000+ tests; see Makefile for fast subsets
scripts/            # Fortress gate, swarm launchers
```

> **Note:** Legacy files are in `legacy_archive/`. Do not add new code to root level.

## 💻 Code Style

```python
# We prefer clarity over cleverness
def good_function(data: List[str]) -> Dict[str, Any]:
    """
    Clear description of what this does.

    Args:
        data: What data represents

    Returns:
        What the function returns
    """
    # Comments for complex logic
    result = process_data(data)
    return result
```

### Strangler Fig Pattern (Zig)

When adding new Zig C-ABI exports, follow the existing pattern:

```python
# Every Zig-accelerated call site must have a Python fallback
zig = _get_zig_engine()
if zig is not None:
    result = zig.bigint_lcm(a, b)
    if result is not None:
        return result
# Legacy Python fallback
return math.lcm(a, b)
```

## 💡 Feature Requests

Have an idea? Open an issue with your vision!

## 🐛 Bug Reports

Found a bug? Let us know with clear steps to reproduce!

## 🎉 Your First Contribution

Look for issues labeled `good first issue` or `help wanted`

**Welcome to the revolution! 🚀**
