#!/bin/bash
# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -euo pipefail

# general configuration
stage=0        # start from 0 if you need to start from data preparation
stop_stage=100
cmd=run.pl
nlsyms=

dict=data/lang_1char/units.txt

echo "$0 $*"  # Print the command line for logging
. utils/parse_options.sh

if [ $# -lt 3 ]; then
    cat << EOF
Usage: $0 <train_set> <dev> <eval_set> [eval_set ...]
e.g. $0 --dict data/lang_1char/units.txt data/train data/dev data/test1 data/test2
EOF
    exit 1
fi

. ./path.sh

train_set=$1
dev_set=$2
eval_sets="${@:3}"

wordlmdatadir=data/local/wordlm_train
lmdatadir=data/local/lm_train

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "$1: stage 3: Prepare texts for lm training"
    if [ ! -z "${nlsyms}" ]; then
        nlsymls_opt="-l ${nlsyms}"
    fi

    mkdir -p  ${wordlmdatadir}  ${lmdatadir}
    # 1. Prepare text for wordlm
    cut -f 2- -d" " data/${train_set}/text > ${wordlmdatadir}/train.txt
    if [ -e ${wordlmdatadir}/train_others.txt ]; then
        cat ${wordlmdatadir}/train_others.txt >> ${wordlmdatadir}/train.txt
    fi
    cut -f 2- -d" " data/${dev_set}/text > ${wordlmdatadir}/dev.txt
    for x in ${eval_sets}; do
        cat data/${x}/text
    done | cut -f 2- -d" " > ${wordlmdatadir}/test.txt


    # 2. Prepare text for charlm
    text2token.py -s 1 -n 1 ${nlsyms_opts} data/${train_set}/text \
        | cut -f 2- -d" " > ${lmdatadir}/train.txt
    if [ -e ${wordlmdatadir}/train_others.txt ]; then
        # TODO(kamo): Check
        <${wordlmdatadir}/train_others.txt \
            text2token.py -n 1 | cut -f 2- -d" " >> ${lmdatadir}/train.txt
    fi
    text2token.py -s 1 -n 1 ${nlsyms_opts} data/${dev_set}/text \
        | cut -f 2- -d" " > ${lmdatadir}/dev.txt
    for x in ${eval_sets}; do
        cat data/${x}/text
    done | text2token.py -s 1 -n 1 ${nlsyms_opts} | cut -f 2- -d" " > ${lmdatadir}/test.text
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "$0: stage 2: Feature Generation"
    for x in ${train_set} ${dev_set} ${eval_sets}; do
        utils/copy_data_dir.sh data/${x} feats/${x}
    done

    # Generate the fbank features
    for x in ${train_set} ${dev_set} ${eval_sets}; do
        make_fbank.sh --cmd "${cmd}" --nj 10 \
            feats/${x} exp/make_fbank/${x} feats/${x}/data
        utils/fix_data_dir.sh feats/${x}
    done

    # compute global CMVN
    compute-cmvn-stats.py scp:feats/${train_set}/feats.scp feats/${train_set}/cmvn.ark
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "$0: stage 3: Make json"
    if [ ! -z "${nlsyms}" ]; then
        nlsymls_opt="--nlsyms ${nlsyms}"
    fi

    for x in ${train_set} ${dev_set} ${eval_sets}; do
        data2json.sh --feat feats/${x}/feats.scp \
            ${nlsyms_otps} feats/${x} ${dict} > feats/${x}/data.json
    done
fi
