#!/usr/bin/env bash
# Token administration for MemoryBridge.
# Usage: bash scripts/token_admin.sh create --label "my-agent"
#        bash scripts/token_admin.sh list
#        bash scripts/token_admin.sh delete TOKEN
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
uv run python scripts/token_admin.py "$@"
