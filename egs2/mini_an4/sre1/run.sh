#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

./sre.sh \
    --lang en \
    --train_set train_nodev \
    --valid_set train_dev \
    --test_sets "test" \
    --sre_args "--preprocess_conf n_fft=8 --preprocess_conf hop_length=2" \
    "$@"
