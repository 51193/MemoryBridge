#!/usr/bin/env bash
# One-click setup for MemoryBridge.
# Downloads Qdrant, creates .env template, initializes token database.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

echo "=== MemoryBridge Setup ==="
echo ""

uv run python -m memory_bridge.host_manager --init
