#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CODE_RELEASE_PYTHON="${CODE_RELEASE_PYTHON:-python3}"

"$CODE_RELEASE_PYTHON" "$REPO_ROOT/tests/synthetic_smoke.py"
