#!/usr/bin/env bash
if ${USE_CONDA:-}; then
    . tools/venv/bin/activate
fi

set -euo pipefail

"$(dirname $0)"/test_black.sh
"$(dirname $0)"/test_flake8.sh
pycodestyle -r espnet test utils setup.py
LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$(pwd)/tools/chainer_ctc/ext/warp-ctc/build" pytest
