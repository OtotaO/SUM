#!/usr/bin/env python3
"""
SUM CLI — Command-Line Interface for the Quantum Knowledge OS

Provides batch operations for the Gödel-State Engine without
running the full FastAPI server.  Operates directly on the local
Akashic Ledger SQLite database.

Usage:
    python scripts/sum_cli.py status
    python scripts/sum_cli.py ingest <file_or_text>
    python scripts/sum_cli.py ask "What is quantum gravity?"
    python scripts/sum_cli.py export --format json
    python scripts/sum_cli.py diff --tick-a 10 --tick-b 50
    python scripts/sum_cli.py provenance "earth||is_a||planet"

Author: ototao
License: Apache License 2.0
"""

import sys
import os
import math
import json
import asyncio

# Uncap BigInt string conversion
sys.set_int_max_str_digits(0)

# Ensure project root is on sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import click

from internal.algorithms.semantic_arithmetic import GodelStateAlgebra
from internal.infrastructure.akashic_ledger import AkashicLedger
from internal.ensemble.confidence_calibrator import ConfidenceCalibrator


# ─── Shared boot context ─────────────────────────────────────────────

def _get_db_path():
    """Resolve the Akashic DB path from env or default."""
    return os.getenv("AKASHIC_DB", "production_akashic.db")


def _boot_sync():
    """Boot the algebra and ledger, replay state — returns (algebra, ledger, state)."""
    algebra = GodelStateAlgebra()
    ledger = AkashicLedger(_get_db_path())

    async def _replay():
        return await ledger.rebuild_state(algebra)

    state = asyncio.run(_replay())
    return algebra, ledger, state


# ─── CLI Group ────────────────────────────────────────────────────────

@click.group()
@click.version_option(version="0.24.0", prog_name="sum")
def cli():
    """SUM — Quantum Knowledge OS CLI

    Operate on the Gödel-State Engine from the command line.
    """
    pass


# ─── status ───────────────────────────────────────────────────────────

@cli.command()
def status():
    """Show current state summary."""
    algebra, ledger, state = _boot_sync()
    tick = asyncio.run(ledger.get_latest_tick())
    active = algebra.get_active_axioms(state)
    paradoxes = algebra.detect_curvature_paradoxes(state)

    click.echo("╔══════════════════════════════════════════╗")
    click.echo("║     SUM — Quantum Knowledge OS           ║")
    click.echo("╠══════════════════════════════════════════╣")
    click.echo(f"║  DB Path:        {_get_db_path():<23} ║")
    click.echo(f"║  Ledger Tick:    {tick:<23} ║")
    click.echo(f"║  Active Axioms:  {len(active):<23} ║")
    click.echo(f"║  Known Primes:   {len(algebra.prime_to_axiom):<23} ║")
    click.echo(f"║  State Bits:     {state.bit_length():<23} ║")
    click.echo(f"║  Paradoxes:      {len(paradoxes):<23} ║")
    click.echo("╚══════════════════════════════════════════╝")

    if paradoxes:
        click.echo("\n⚠ Paradoxes detected:")
        for p in paradoxes:
            click.echo(f"  • {p}")


# ─── ingest ───────────────────────────────────────────────────────────

@cli.command()
@click.argument("source")
@click.option("--source-url", default="", help="Provenance URL for this content")
@click.option("--confidence", default=None, type=float, help="Manual confidence [0.0–1.0]. Omit for auto-calibration.")
def ingest(source, source_url, confidence):
    """Ingest triplets from a file or inline text.

    SOURCE can be a .json file path containing triplets
    [[subject, predicate, object], ...] or inline text
    like "earth;orbits;sun | mars;has;moons".
    """
    algebra, ledger, state = _boot_sync()

    # Parse triplets
    triplets = []
    if os.path.isfile(source):
        with open(source, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                triplets = [t for t in data if isinstance(t, list) and len(t) == 3]
        if not source_url:
            source_url = f"file://{os.path.abspath(source)}"
    else:
        # Inline: "subj;pred;obj | subj;pred;obj"
        for part in source.split("|"):
            fields = [x.strip() for x in part.strip().split(";")]
            if len(fields) == 3:
                triplets.append(fields)

    if not triplets:
        click.echo("❌ No valid triplets found.", err=True)
        raise SystemExit(1)

    # Ingest
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    added = 0

    async def _ingest():
        nonlocal state, added
        calibrator = ConfidenceCalibrator()
        for s, p, o in triplets:
            s, p, o = s.strip().lower(), p.strip().lower(), o.strip().lower()
            prime = algebra.get_or_mint_prime(s, p, o)
            axiom = f"{s}||{algebra._canonicalize_predicate(p)}||{o}"
            if state % prime != 0:
                # Phase 24: auto-calibrate or use manual
                conf = await calibrator.calibrate(
                    axiom_key=axiom,
                    source_url=source_url,
                    current_state=state,
                    algebra=algebra,
                    ledger=ledger,
                    manual_confidence=confidence,
                )
                state = math.lcm(state, prime)
                await ledger.append_event(
                    "MINT", prime, axiom,
                    source_url=source_url,
                    confidence=conf,
                    ingested_at=now,
                )
                await ledger.append_event("MUL", prime)
                added += 1

    asyncio.run(_ingest())
    click.echo(f"✅ Ingested {added} axioms ({len(triplets)} provided, {len(triplets) - added} duplicates)")
    click.echo(f"   State bit-length: {state.bit_length()}")


# ─── ask ──────────────────────────────────────────────────────────────

@cli.command()
@click.argument("question")
@click.option("--top-k", default=10, type=int, help="Max results to return")
def ask(question, top_k):
    """Query knowledge in the Gödel state.

    Uses keyword matching against all active axioms.
    """
    algebra, ledger, state = _boot_sync()
    active = algebra.get_active_axioms(state)

    if not active:
        click.echo("No knowledge ingested yet.")
        return

    # Keyword matching
    question_lower = question.lower()
    stop_words = {
        "the", "and", "for", "are", "was", "were", "has", "have",
        "this", "that", "with", "from", "what", "who", "how",
        "why", "when", "where", "does", "did", "can", "could",
        "will", "would", "should", "about", "which",
    }
    keywords = [
        w for w in question_lower.replace("?", "").replace(".", "").split()
        if len(w) > 2 and w not in stop_words
    ]

    scored = []
    for ax in active:
        parts = ax.split("||")
        if len(parts) == 3:
            s, p, o = parts
            ax_text = f"{s} {p} {o}".lower()
            score = sum(1 for kw in keywords if kw in ax_text)
            if score > 0:
                scored.append((score, s, p, o, ax))

    # Provenance lookup
    if scored:
        axiom_keys = [item[4] for item in scored]
        prov_map = asyncio.run(ledger.get_provenance_batch(axiom_keys))
        from datetime import datetime as _dt, timezone as _tz
        _now = _dt.now(_tz.utc)

        weighted = []
        for score, s, p, o, ax in scored:
            prov = prov_map.get(ax, {})
            conf = prov.get("confidence", 0.5)
            ingested = prov.get("ingested_at", "")
            recency = 1.0
            if ingested:
                try:
                    ts = _dt.fromisoformat(ingested)
                    age_days = max((_now - ts).days, 0)
                    recency = 0.5 ** (age_days / 30.0)
                except (ValueError, TypeError):
                    pass
            weighted.append((score * conf * recency, s, p, o, conf, prov.get("source_url", "")))
        weighted.sort(reverse=True)
    else:
        weighted = []

    if not weighted:
        click.echo("No matching knowledge found.")
        return

    click.echo(f"\n🔍 Results for: \"{question}\" ({len(weighted)} matches)\n")
    for i, (wscore, s, p, o, conf, src) in enumerate(weighted[:top_k], 1):
        click.echo(f"  {i}. {s} → {p} → {o}")
        click.echo(f"     score={wscore:.3f}  confidence={conf:.2f}  source={src or '—'}")
    click.echo()


# ─── export ───────────────────────────────────────────────────────────

@cli.command("export")
@click.option("--format", "fmt", type=click.Choice(["json", "tsv", "triplets"]),
              default="json", help="Output format")
@click.option("--output", "-o", default="-", help="Output file (- for stdout)")
def export_state(fmt, output):
    """Export the current knowledge state."""
    algebra, ledger, state = _boot_sync()
    active = algebra.get_active_axioms(state)
    tick = asyncio.run(ledger.get_latest_tick())

    triplets = []
    for ax in active:
        parts = ax.split("||")
        if len(parts) == 3:
            triplets.append(parts)

    if fmt == "json":
        data = {
            "version": "0.23.0",
            "tick": tick,
            "state_bit_length": state.bit_length(),
            "axiom_count": len(triplets),
            "axioms": [
                {"subject": s, "predicate": p, "object": o}
                for s, p, o in triplets
            ],
        }
        content = json.dumps(data, indent=2)
    elif fmt == "tsv":
        lines = ["subject\tpredicate\tobject"]
        for s, p, o in triplets:
            lines.append(f"{s}\t{p}\t{o}")
        content = "\n".join(lines)
    else:  # triplets
        content = json.dumps(triplets, indent=2)

    if output == "-":
        click.echo(content)
    else:
        with open(output, "w") as f:
            f.write(content)
        click.echo(f"✅ Exported {len(triplets)} axioms to {output}")


# ─── diff ─────────────────────────────────────────────────────────────

@cli.command()
@click.option("--tick-a", type=int, required=True, help="Earlier tick")
@click.option("--tick-b", type=int, required=True, help="Later tick")
def diff(tick_a, tick_b):
    """Compare state between two historical ticks."""
    async def _diff():
        alg_a = GodelStateAlgebra()
        alg_b = GodelStateAlgebra()
        ledger = AkashicLedger(_get_db_path())

        state_a = await ledger.rebuild_state(alg_a, max_seq_id=tick_a)
        state_b = await ledger.rebuild_state(alg_b, max_seq_id=tick_b)

        axioms_a = set(alg_a.get_active_axioms(state_a))
        axioms_b = set(alg_b.get_active_axioms(state_b))

        added = axioms_b - axioms_a
        removed = axioms_a - axioms_b
        return axioms_a, axioms_b, added, removed

    axioms_a, axioms_b, added, removed = asyncio.run(_diff())

    click.echo(f"\n📊 State diff: tick {tick_a} → tick {tick_b}")
    click.echo(f"   Axioms at tick {tick_a}: {len(axioms_a)}")
    click.echo(f"   Axioms at tick {tick_b}: {len(axioms_b)}")
    click.echo(f"   Added:   +{len(added)}")
    click.echo(f"   Removed: -{len(removed)}")

    if added:
        click.echo("\n  ✅ Added:")
        for ax in sorted(added)[:20]:
            s, p, o = ax.split("||") if "||" in ax else (ax, "", "")
            click.echo(f"    + {s} → {p} → {o}")
        if len(added) > 20:
            click.echo(f"    ... and {len(added) - 20} more")

    if removed:
        click.echo("\n  ❌ Removed:")
        for ax in sorted(removed)[:20]:
            s, p, o = ax.split("||") if "||" in ax else (ax, "", "")
            click.echo(f"    - {s} → {p} → {o}")
        if len(removed) > 20:
            click.echo(f"    ... and {len(removed) - 20} more")
    click.echo()


# ─── provenance ───────────────────────────────────────────────────────

@cli.command()
@click.argument("axiom_key")
def provenance(axiom_key):
    """Show provenance chain for an axiom.

    AXIOM_KEY should be in "subject||predicate||object" format.
    """
    algebra, ledger, state = _boot_sync()
    chain = asyncio.run(ledger.get_axiom_provenance(axiom_key))
    prime = algebra.axiom_to_prime.get(axiom_key)

    if not chain and not prime:
        click.echo(f"❌ Axiom '{axiom_key}' not found.")
        return

    click.echo(f"\n🔗 Provenance for: {axiom_key}")
    if prime:
        click.echo(f"   Prime: {prime}")
        is_active = state % prime == 0 if state > 1 else False
        click.echo(f"   Active: {'✅ yes' if is_active else '❌ no'}")

    if chain:
        click.echo(f"   Sources: {len(chain)}\n")
        for i, entry in enumerate(chain, 1):
            click.echo(f"   {i}. tick={entry['seq_id']}")
            click.echo(f"      source: {entry['source_url'] or '—'}")
            click.echo(f"      confidence: {entry['confidence']:.2f}")
            click.echo(f"      ingested: {entry['ingested_at'] or '—'}")
    else:
        click.echo("   No provenance records found.")
    click.echo()


# ─── Entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
