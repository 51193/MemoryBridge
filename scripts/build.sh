#!/usr/bin/env bash
set -euo pipefail

# ── Usage ──────────────────────────────────────────────────────────────────
#   bash scripts/build.sh                    # full: deps → checks → build
#   bash scripts/build.sh --check            # deps + checks only (no build)
#   bash scripts/build.sh --build            # build pyz only (skip deps & checks)
#   bash scripts/build.sh --build --version 1.2.3   # build with version injection
#   bash scripts/build.sh --run              # build then start service to smoke-test
#
# Environment:
#   PYTHON_VERSION           Python version for uv sync (default: 3.13)
#   AUTO_INSTALL_PYTHON      Set to "true" to auto-install Python via uv
#                            (used by CI; local devs manage Python themselves)
#
# CI alignment:
#   CI test.yml    →  bash scripts/build.sh --check
#   CI release.yml →  bash scripts/build.sh --build --version $VERSION

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()  { echo -e "${GREEN}✓${NC} $*"; }
warn(){ echo -e "${YELLOW}⚠${NC} $*"; }
die() { echo -e "${RED}✗${NC} $*"; exit 1; }

# ── Parse args ──────────────────────────────────────────────────────────────

DO_CHECK=true
DO_BUILD=true
DO_RUN=false
VERSION=""
PYTHON_VER="${PYTHON_VERSION:-3.13}"
INSTALL_PYTHON="${AUTO_INSTALL_PYTHON:-false}"
OLD_VERSION=""
RESTORE_VERSION=false

i=1
while [ $i -le $# ]; do
    arg="${!i}"
    case "$arg" in
        --check) DO_BUILD=false ;;
        --build) DO_CHECK=false ;;
        --run)   DO_RUN=true ;;
        --version)
            i=$((i + 1))
            if [ $i -gt $# ]; then die "--version requires a value (e.g. --version 1.2.3)"; fi
            VERSION="${!i}"
            ;;
        -h|--help)
            echo "Usage: bash scripts/build.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --check            Run checks only (mypy + ruff + pytest), skip build"
            echo "  --build            Build .pyz only, skip checks"
            echo "  --run              Build then start service (smoke test)"
            echo "  --version X.Y.Z    Inject version into pyproject.toml before build"
            echo "  --help             Show this help"
            echo ""
            echo "Environment:"
            echo "  PYTHON_VERSION           Python version for uv sync (default: 3.13)"
            echo "  AUTO_INSTALL_PYTHON      Set to 'true' for CI: auto-install Python via uv"
            exit 0
            ;;
        *) die "unknown option: $arg (use --help)" ;;
    esac
    i=$((i + 1))
done

# ── Version injection (before uv sync — matches CI release.yml order) ─────
# Only inject if building AND a version was provided.
# In --check mode this is a no-op regardless.

if [ "$DO_BUILD" = true ] && [ -n "$VERSION" ]; then
    OLD_VERSION="$(grep '^version = ' pyproject.toml | head -1 | sed 's/version = "\(.*\)"/\1/')"
    echo "injecting version: $VERSION (was: $OLD_VERSION)"
    sed -i "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml
    RESTORE_VERSION=true
fi

# ── Python install (CI-only; local devs manage Python themselves) ────────────

if [ "${INSTALL_PYTHON}" = "true" ]; then
    echo "=== installing python ${PYTHON_VER} ==="
    uv python install "${PYTHON_VER}" || die "uv python install failed"
fi

# ── Deps ─────────────────────────────────────────────────────────────────────

if [ "$DO_CHECK" = true ]; then
    echo "=== install deps (python ${PYTHON_VER}) ==="
    uv sync --extra dev --python "${PYTHON_VER}" || die "uv sync failed"
elif [ "$DO_BUILD" = true ]; then
    # Build-only mode: runtime deps only (matches CI release.yml)
    echo "=== install deps (python ${PYTHON_VER}, runtime only) ==="
    uv sync --python "${PYTHON_VER}" || die "uv sync failed"
fi

# Ensure shiv is available for the build step
if [ "$DO_BUILD" = true ] && ! uv pip show shiv &>/dev/null; then
    uv pip install shiv
fi

# ── Checks ───────────────────────────────────────────────────────────────────

if [ "$DO_CHECK" = true ]; then
    echo ""
    echo "=== mypy ==="
    uv run mypy src/ || die "mypy failed"

    echo ""
    echo "=== ruff ==="
    uv run ruff check src/ tests/ || die "ruff failed"

    echo ""
    echo "=== pytest ==="
    uv run pytest -v || die "tests failed"

    ok "all checks passed"
fi

# ── Build ─────────────────────────────────────────────────────────────────────

if [ "$DO_BUILD" = true ]; then
    echo ""
    echo "=== shiv build ==="

    PYTHON_VER_OUT="$("${PYTHON3:-python3}" --version 2>&1)"
    echo "python: $PYTHON_VER_OUT"

    mkdir -p dist
    uv run shiv \
        --compile-pyc \
        --console-script host-manager \
        --output-file dist/memorybridge.pyz \
        .

    SIZE="$(du -h dist/memorybridge.pyz | cut -f1)"
    ok "built dist/memorybridge.pyz ($SIZE)"

    # ── Checksum (matches CI release.yml: cd dist && sha256sum) ──────────────
    (cd dist && sha256sum memorybridge.pyz > memorybridge.pyz.sha256)
    ok "checksum: dist/memorybridge.pyz.sha256"
fi

# ── Restore original version ─────────────────────────────────────────────────

if [ "$RESTORE_VERSION" = true ] && [ -n "$OLD_VERSION" ]; then
    sed -i "s/^version = \".*\"/version = \"$OLD_VERSION\"/" pyproject.toml
    echo "restored version: $OLD_VERSION"
fi

# ── Run / smoke ───────────────────────────────────────────────────────────────

if [ "$DO_RUN" = true ]; then
    echo ""
    echo "=== starting service ==="
    python3 dist/memorybridge.pyz
fi
