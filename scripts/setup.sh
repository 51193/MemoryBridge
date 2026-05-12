#!/usr/bin/env bash
# One-click setup for MemoryBridge.
# Downloads Qdrant, creates .env template, generates initial API token.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

echo "=== MemoryBridge Setup ==="
echo ""

# Step 1: setup (Qdrant + .env)
echo "[1/2] Running --setup ..."
uv run python -m memory_bridge.host_manager --setup

# Step 2: init token
echo ""
echo "[2/2] Creating initial API token ..."
uv run python scripts/token_admin.py create --label "admin"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit .env with your API keys"
echo "  2. Start the service: uv run python -m memory_bridge.host_manager"
