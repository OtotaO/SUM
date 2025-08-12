#!/bin/bash
# quickstart.sh - Get SUM v2 running in 60 seconds

echo "🚀 SUM v2 Quick Start - Simplicity in Action!"
echo "=============================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed!${NC}"
    echo "Please install Docker from https://docker.com"
    exit 1
fi

# Function to check if port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        return 0
    else
        return 1
    fi
}

echo -e "${YELLOW}📦 Starting Redis (FREE and open source!)...${NC}"
# Stop any existing Redis container
docker stop sum-redis-quick 2>/dev/null || true
docker rm sum-redis-quick 2>/dev/null || true

# Start Redis
docker run -d \
    --name sum-redis-quick \
    -p 6379:6379 \
    redis:7-alpine \
    redis-server --appendonly yes

echo -e "${GREEN}✓ Redis started!${NC}"

echo -e "${YELLOW}🐘 Starting PostgreSQL...${NC}"
# Stop any existing PostgreSQL container
docker stop sum-postgres-quick 2>/dev/null || true
docker rm sum-postgres-quick 2>/dev/null || true

# Start PostgreSQL
docker run -d \
    --name sum-postgres-quick \
    -p 5432:5432 \
    -e POSTGRES_USER=sum \
    -e POSTGRES_PASSWORD=sum123 \
    -e POSTGRES_DB=sum \
    postgres:16-alpine

echo -e "${GREEN}✓ PostgreSQL started!${NC}"

# Wait for services to be ready
echo -e "${YELLOW}⏳ Waiting for services to be ready...${NC}"
sleep 5

# Check if services are running
echo -e "${YELLOW}🔍 Checking services...${NC}"

# Check Redis
if docker exec sum-redis-quick redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Redis is ready!${NC}"
else
    echo -e "${RED}❌ Redis failed to start${NC}"
    exit 1
fi

# Check PostgreSQL
if docker exec sum-postgres-quick pg_isready -U sum > /dev/null 2>&1; then
    echo -e "${GREEN}✓ PostgreSQL is ready!${NC}"
else
    echo -e "${RED}❌ PostgreSQL failed to start${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}🎉 All services are running!${NC}"
echo ""
echo "You can now run SUM v2:"
echo "----------------------"
echo ""
echo "1. Simple API (core features):"
echo "   python sum_simple.py"
echo ""
echo "2. Intelligence API (with patterns & memory):"
echo "   python sum_intelligence.py"
echo ""
echo "3. Or use Docker Compose for the full stack:"
echo "   docker-compose -f docker-compose-simple.yml up"
echo ""
echo "Test the API:"
echo "-------------"
echo "curl -X POST localhost:3000/summarize \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{\"text\": \"Your text to summarize here\"}'"
echo ""
echo -e "${YELLOW}To stop the services later:${NC}"
echo "docker stop sum-redis-quick sum-postgres-quick"
echo "docker rm sum-redis-quick sum-postgres-quick"
echo ""
echo -e "${GREEN}✨ Happy summarizing with simplicity!${NC}"