#!/bin/bash
# ─── SUM Planetary Swarm Launcher ───────────────────────────────────
# Spins up 3 isolated Knowledge OS nodes on ports 8000/8001/8002.
# Each node gets its own Akashic Ledger (SQLite) via AKASHIC_DB env var.
# Press Ctrl+C to cleanly shut down all nodes.

echo "🚀 Igniting the Planetary Swarm..."

mkdir -p logs
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-math-only-mode}"

AKASHIC_DB=alpha_akashic.db uvicorn quantum_main:app --port 8000 > logs/alpha.log 2>&1 &
P1=$!
echo "🟢 Node Alpha running on port 8000 (PID: $P1)"

AKASHIC_DB=beta_akashic.db uvicorn quantum_main:app --port 8001 > logs/beta.log 2>&1 &
P2=$!
echo "🔵 Node Beta running on port 8001 (PID: $P2)"

AKASHIC_DB=gamma_akashic.db uvicorn quantum_main:app --port 8002 > logs/gamma.log 2>&1 &
P3=$!
echo "🟣 Node Gamma running on port 8002 (PID: $P3)"

trap "echo '🛑 Shutting down swarm...'; kill $P1 $P2 $P3 2>/dev/null; exit" SIGINT SIGTERM

echo ""
echo "🌍 Swarm is live. Press Ctrl+C to stop all nodes."
echo "   Alpha UI: http://localhost:8000"
echo "   Beta  UI: http://localhost:8001"
echo "   Gamma UI: http://localhost:8002"
echo ""
echo "Next steps:"
echo "   1. Wait ~5s for boot, then: python scripts/ignite_mesh.py"
echo "   2. Start harvesting:        python scripts/babel_harvester.py"
echo ""
wait
