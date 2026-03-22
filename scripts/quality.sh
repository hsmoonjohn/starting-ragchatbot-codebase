#!/bin/bash
# Run all code quality checks for the project.
# Usage: ./scripts/quality.sh [--fix]
#
# Options:
#   --fix    Auto-fix formatting and import order issues (black + ruff --fix)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

FIX=false
for arg in "$@"; do
  case $arg in
    --fix) FIX=true ;;
  esac
done

PASS=0
FAIL=0

run_check() {
  local name="$1"
  shift
  echo ""
  echo "── $name ──────────────────────────────────"
  if "$@"; then
    echo "✓ $name passed"
    ((PASS++))
  else
    echo "✗ $name failed"
    ((FAIL++))
  fi
}

echo "================================================"
echo " Code Quality Checks"
echo "================================================"

if $FIX; then
  echo ""
  echo "── Auto-fix mode ──────────────────────────────"
  echo "Running black (format)..."
  uv run black backend/
  echo "Running ruff (fix imports + lint)..."
  uv run ruff check --fix backend/
  echo ""
  echo "Auto-fix complete. Running checks to confirm..."
fi

run_check "black (formatting)" \
  uv run black --check backend/

run_check "ruff (linting)" \
  uv run ruff check backend/

run_check "pytest (tests)" \
  uv run pytest backend/tests/ -v

echo ""
echo "================================================"
echo " Results: ${PASS} passed, ${FAIL} failed"
echo "================================================"

if [ "$FAIL" -gt 0 ]; then
  echo ""
  echo "Tip: run './scripts/quality.sh --fix' to auto-fix formatting issues."
  exit 1
fi
