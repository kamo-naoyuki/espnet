#!/bin/bash
# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -euo pipefail

cmd=run.pl
nj=32

backend=pytorch
ngpu=0         # number of gpus ("0" uses cpu, otherwise use gpu)

decode_config=conf/decode.yaml
nlsyms=data/lang_1char/non_lang_syms.txt
dict=data/lang_1char/units.txt

# ${lmexpdir}/rnnlm.model.best
lm=
wordlm=

wer=true  # calc wer or not

echo "$0 $*"  # Print the command line for logging
. utils/parse_options.sh
. ./path.sh

expdir=$1
recog_model=$2
recog_set="${@:3}"


pids=() # initialize pids
for rtask in ${recog_set}; do
(
    decode_dir=decode_${rtask}_$(basename ${decode_config%.*})
    if [ ! -z ${lm} ]; then
        recog_opts="--word-rnnlm ${lm}"
    elif [ ! -z ${wordlm} ]; then
        recog_opts="--rnnlm ${wordlm}"
    fi

    # split data
    splitjson.py --parts ${nj} feats/${rtask}/data.json

    ${cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
        asr_recog.py \
        --config ${decode_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --recog-json feats/${rtask}/split${nj}utt/data.JOB.json \
        --result-label ${expdir}/${decode_dir}/data.JOB.json \
        --model ${recog_model}  \
        ${recog_opts}

    score_sclite.sh --wer ${wer} --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}

) &
pids+=($!) # store background pids
done
i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
[ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
