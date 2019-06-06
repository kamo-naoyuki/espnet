#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -euo pipefail

# general configuration
stage=0        # start from 0 if you need to start from data preparation
stop_stage=100

. utils/parse_options.sh
. ./path.sh
. ./cmd.sh


wsj0=/export/corpora5/LDC/LDC93S6B
wsj1=/export/corpora5/LDC/LDC94S13B
train_set=train_si284
dev_set=test_dev93
eval_sets="test_eval92 "

dict=data/lang_1char/units.txt
nlsyms=data/lang_1char/non_lang_syms.txt

wordlmdatadir=data/local/wordlm_train
wordlmdict=${lmdatadir}/wordlist_${lm_vocabsize}.txt
lmdatadir=data/local/lm_train
lmdict=${dict}

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    local/wsj_data_prep.sh ${wsj0}/??-{?,??}.? ${wsj1}/??-{?,??}.?
    local/wsj_format_data.sh

    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list"
    cut -f 2- data/${train_set}/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
    cat ${nlsyms}

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    zcat ${wsj1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z \
            | grep -v "<" | tr "[:lower:]" "[:upper:]" > ${wordlmdatadir}/train_others.txt
fi



if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then

    mkdir -p  ${wordlmdatadir}  ${lmdatadir}
    # 1. Prepare text for wordlm
    cut -f 2- -d" " data/${train_set}/text > ${wordlmdatadir}/train.txt
    if [ -e ${wordlmdatadir}/train_others.txt ]; then
        cat ${wordlmdatadir}/train_others.txt >> ${wordlmdatadir}/train.txt
    fi
    cut -f 2- -d" " data/${dev_set}/text > ${wordlmdatadir}/dev.txt
    for rtask in ${eval_sets}; do
        cat data/${rtask}/text
    done | cut -f 2- -d" " > ${wordlmdatadir}/test.txt


    # 2. Prepare text for charlm
    text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}/text \
        | cut -f 2- -d" " > ${lmdatadir}/train.txt
    if [ -e ${wordlmdatadir}/train_others.txt ]; then
        # TODO(kamo): Check
        <${wordlmdatadir}/train_others.txt \
            text2token.py -n 1 | cut -f 2- -d" " >> ${lmdatadir}/train.txt
    fi
    text2token.py -s 1 -n 1 -l ${nlsyms} data/${dev_set}/text \
        | cut -f 2- -d" " > ${lmdatadir}/dev.txt
    for rtask in ${eval_sets}; do
        cat data/${rtask}/text
    done | text2token.py -s 1 -n 1 -l ${nlsyms} | cut -f 2- -d" " > ${lmdatadir}/test.text

fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature Generation"
    for x in ${train_set} ${dev_set} ${eval_sets}; do
        utils/copy_data_dir.sh data/${x} feats/${x}
    done

    # Generate the fbank features
    for x in ${train_set} ${dev_set} ${eval_sets}; do
        make_fbank.sh --cmd "${train_cmd}" --nj 10 \
            feats/${x} exp/make_fbank/${x} feats/${x}/data
        utils/fix_data_dir.sh feats/${x}
    done

    # compute global CMVN
    compute-cmvn-stats.py scp:feats/${train_set}/feats.scp feats/${train_set}/cmvn.ark

    echo "make json files"
    for rtask in ${train_set} ${dev_set} ${eval_sets}; do
        data2json.sh --feat feats/${rtask}/feats.scp \
            --nlsyms ${nlsyms} feats/${rtask} ${dict} > feats/${rtask}/data.json
    done
fi
