#!/bin/bash
set -euo pipefail
SECONDS=0
log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=0
stop_stage=10000000
user=
password=
audio_format=flac

cmd=run.pl

log "$0 $*"
. ./utils/parse_options.sh || exit 1;

if [ $# -ne 0 ]; then
    log "Usage $0"
    exit 1
fi

. ./db.sh    # Specify the corpora directory
. ./path.sh  # Setup the environment


if [ -z "${VOXCELEB1}" ]; then
    log "Error: \$VOXCELEB1 is not set. See db.sh"
    exit 1
fi
if [ -z "${VOXCELEB2}" ]; then
    log "Error: \$VOXCELEB2 is not set. See db.sh"
    exit 1
fi


if [ $stage -le 1 ]; then
    log "stage 1: Download and conver"
    if [ -z "${user}" ]; then
        log "Error: give --user and --password"
        exit 1
    fi
    if [ -z "${password}" ]; then
        log "Error: give --user and --password"
        exit 1
    fi
    local/dataprep.py --user "${user}" --password "${password}" --audio_format "${audio_format}" --download --extract --convert
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Format datadir"
    mkdir -p data/train data/test
    <${VOXCELEB2}/train_list.txt awk '{ split($2, array, "/"); split(array[3], array, "."); print($1 "_" array[1],"'${VOXCELEB2}'/dev/aac/" $2) }' | LC_ALL=C sort > data/train/wav.scp
    <${VOXCELEB2}/train_list.txt awk '{ split($2, array, "/"); split(array[3], array, "."); print($1 "_" array[1],$1) }' | LC_ALL=C sort > data/train/utt2spk
    <data/train/utt2spk utils/utt2spk_to_spk2utt.pl >data/train/spk2utt

    <${VOXCELEB2}/veri_test.txt awk '{ printf("pair_%05d '${VOXCELEB1}'/test/wav/%s\n",NR,$2) }' > data/test/wav.scp
    <${VOXCELEB2}/veri_test.txt awk '{ printf("pair_%05d '${VOXCELEB1}'/test/wav/%s\n",NR,$3) }' > data/test/ref.scp
    <${VOXCELEB2}/veri_test.txt awk '{ printf("pair_%05d %d\n",NR,$1) }' > data/test/label
    <${VOXCELEB2}/veri_test.txt awk '{ printf("pair_%05d dummy\n",NR) }' > data/test/utt2spk
    <data/test/utt2spk utils/utt2spk_to_spk2utt.pl >data/test/spk2utt

    # TODO(kamo): define dev set

fi

log "Successfully finished. [elapsed=${SECONDS}s]"
