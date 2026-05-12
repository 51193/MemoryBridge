#!/usr/bin/env bash
# Clean up test data from integration smoke tests.
# Usage: bash tests/integration/cleanup.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
uv run python tests/integration/cleanup.py
