#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
tracker="${1:-tgtrack_b256}"
vot evaluate "$tracker"
vot analysis "$tracker" --nocache
