#!/bin/bash
# cosmic_shutdown.sh - Gracefully shutdown the cosmic elevator

echo "🌙 COSMIC SHUTDOWN SEQUENCE"
echo "=========================="
echo ""

# Shutdown services
if [ -d pids ]; then
    for pidfile in pids/*.pid; do
        if [ -f "$pidfile" ]; then
            service=$(basename "$pidfile" .pid)
            pid=$(cat "$pidfile")
            
            if ps -p $pid > /dev/null 2>&1; then
                echo "🛑 Stopping $service (PID: $pid)..."
                kill $pid 2>/dev/null
                rm "$pidfile"
            else
                echo "⚠️  $service already stopped"
                rm "$pidfile"
            fi
        fi
    done
fi

echo ""
echo "✅ All cosmic services stopped"
echo "🌟 Until next time, cosmic traveler!"