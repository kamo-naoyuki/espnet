#!/usr/bin/env bash

if ${USE_CONDA:-}; then
    . tools/venv/bin/activate
fi
# The black default is 88
linewidth=120

targets="espnet test doc utils setup.py"
black --line-length ${linewidth} --check ${targets}
if [ $? != 0 ]; then
    echo "Please apply "black" style as following:"
    echo "% pip install black"
    echo "% black --line-length ${linewidth} ${targets}"
    exit 1
fi
