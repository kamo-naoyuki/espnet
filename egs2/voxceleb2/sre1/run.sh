#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail
audio_format=wav

./sre.sh \
    --lang en \
    --fs 16000 \
    --audio_format "${audio_format}" \
    --local_data_opts "--audio_format ${audio_format}" \
    --train_set train \
    --valid_set test \
    --test_sets "test" \
    "$@"
