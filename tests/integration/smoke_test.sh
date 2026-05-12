#!/usr/bin/env bash
# Integration smoke test for MemoryBridge.
# Requires MemoryBridge server running on localhost:8000.
# Usage: bash tests/integration/smoke_test.sh [--base-url URL] [--quiet]
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
uv run python tests/integration/smoke_test.py "$@"
