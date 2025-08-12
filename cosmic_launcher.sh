#!/bin/bash
# cosmic_launcher.sh - Launch the entire cosmic elevator with one command!

echo "🌌 COSMIC ELEVATOR LAUNCHER"
echo "=========================="
echo ""
echo "🚀 Initializing all dimensions of consciousness..."
echo ""

# Check if Redis is running
if ! pgrep -x "redis-server" > /dev/null; then
    echo "⚡ Starting Redis (for cosmic memory)..."
    if command -v redis-server &> /dev/null; then
        redis-server --daemonize yes
    else
        echo "📦 Starting Redis in Docker..."
        docker run -d -p 6379:6379 --name cosmic-redis redis:7-alpine 2>/dev/null || echo "Redis already running"
    fi
fi

# Function to start a service in background
start_service() {
    local name=$1
    local script=$2
    local port=$3
    
    echo "🌟 Starting $name (Port $port)..."
    python $script > logs/${name}.log 2>&1 &
    echo $! > pids/${name}.pid
    sleep 2
}

# Create directories for logs and pids
mkdir -p logs pids

# Start all services
start_service "Simple-Dimension" "sum_simple.py" "3000"
start_service "Quantum-Dimension" "quantum_summary_engine.py" "3002"
start_service "Akashic-Records" "akashic_records.py" "3003"
start_service "Consciousness-Stream" "cosmic_consciousness_stream.py" "8765"

echo ""
echo "✅ All cosmic services launched!"
echo ""
echo "🌐 Services running at:"
echo "   - Simple API: http://localhost:3000"
echo "   - Quantum API: http://localhost:3002"
echo "   - Akashic API: http://localhost:3003"
echo "   - Consciousness Stream: ws://localhost:8765"
echo ""
echo "📊 Checking service health..."
sleep 3

# Check if services are running
for service in "Simple-Dimension" "Quantum-Dimension" "Akashic-Records" "Consciousness-Stream"; do
    if [ -f pids/${service}.pid ]; then
        pid=$(cat pids/${service}.pid)
        if ps -p $pid > /dev/null; then
            echo "   ✅ $service is running (PID: $pid)"
        else
            echo "   ❌ $service failed to start"
        fi
    fi
done

echo ""
echo "🎯 Ready for cosmic integration!"
echo ""
echo "Try these commands:"
echo "   1. python cosmic_integration.py     # Full cosmic experience"
echo "   2. python cosmic_consciousness_client.py  # Connect to consciousness stream"
echo "   3. python test_simple.py            # Test basic functionality"
echo ""
echo "To stop all services: ./cosmic_shutdown.sh"
echo ""
echo "🌟 Welcome to the Cosmic Elevator! 🌟"