#!/usr/bin/env bash
set -euo pipefail

# ── Usage ──────────────────────────────────────────────────────────────────
#   bash scripts/build.sh              # full: lint → test → build
#   bash scripts/build.sh --check      # lint + test only
#   bash scripts/build.sh --build      # build pyz only (skip checks)
#   bash scripts/build.sh --run        # build then start service to smoke-test

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

ok() { echo -e "${GREEN}✓${NC} $*"; }
die() { echo -e "${RED}✗${NC} $*"; exit 1; }

MODE="${1:-}"

# ── Checks ─────────────────────────────────────────────────────────────────

if [ "$MODE" != "--build" ]; then
    echo "=== mypy =="
    uv run mypy src/ || die "mypy failed"

    echo ""
    echo "=== ruff =="
    uv run ruff check src/ tests/ || die "ruff failed"

    echo ""
    echo "=== pytest =="
    uv run pytest -v || die "tests failed"

    ok "all checks passed"
fi

# ── Build ───────────────────────────────────────────────────────────────────

if [ "$MODE" != "--check" ]; then
    echo ""
    echo "=== shiv build ==="
    mkdir -p dist

    if ! uv pip show shiv &>/dev/null; then
        uv pip install shiv
    fi

    PYTHON_VERSION="$(uv run python --version 2>&1)"
    echo "python: $PYTHON_VERSION"

    uv run shiv \
        --compile-pyc \
        --console-script host-manager \
        --output-file dist/memorybridge.pyz \
        .

    SIZE="$(du -h dist/memorybridge.pyz | cut -f1)"
    ok "built dist/memorybridge.pyz ($SIZE)"

    sha256sum dist/memorybridge.pyz > dist/memorybridge.pyz.sha256
    ok "checksum: dist/memorybridge.pyz.sha256"
fi

# ── Run / smoke ─────────────────────────────────────────────────────────────

if [ "$MODE" = "--run" ]; then
    echo ""
    echo "=== starting service ==="
    python3 dist/memorybridge.pyz
fi
