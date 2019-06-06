#!/bin/bash
# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -euo pipefail

stage=0        # start from 0 if you need to start from data preparation
stop_stage=100

cmd=run.pl
backend=pytorch
ngpu=1
dict=data/lang_1char/units.txt

lm_config=conf/lm.yaml
use_wordlm=true     # false means to train/use a character LM
lm_vocabsize=65000  # effective only for word LMs
lm_resume=          # specify a snapshot file to resume LM training
lmtag=              # tag for managing LMs

echo "$0 $*"  # Print the command line for logging
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    echo "Usage: $0"
    exit 1
fi

. ./path.sh


# It takes about one day. If you just want to do end-to-end ASR without LM,
# you can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    if [ ${use_wordlm} = true ]; then
        lmtag=${lmtag}_word${lm_vocabsize}
    fi
fi
lmexpname=train_rnnlm_${backend}_${lmtag}
lmexpdir=exp/${lmexpname}
if [ ${use_wordlm} = true ]; then
    lmdatadir=data/local/wordlm_train
    lmdict=${lmexpdir}/wordlist_${lm_vocabsize}.txt
else
    lmdatadir=data/local/lm_train
    [ -z "${dict}" ] && echo "$0: Error: --dict is required if --use_wordlm=false" && exit 1
    lmdict=${dict}
fi


for f in ${lmdatadir}/train.txt ${lmdatadir}/valid.txt ${lmdatadir}/test.txt ${lmdict}; do
    [ -e "${f}" ] && echo "$0: Error: ${f} doesn't exist" && exit 1
done
mkdir -p ${lmexpdir}

if [ ${use_wordlm} = true ]; then
    text2vocabulary.py -s ${lm_vocabsize} -o ${lmdict} ${lmdatadir}/train.txt
fi

# use only 1 gpu
if [ ${ngpu} -gt 1 ]; then
    echo "LM training does not support multi-gpu. signle gpu will be used."
fi

${cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
    lm_train.py \
    --config ${lm_config} \
    --ngpu ${ngpu} \
    --backend ${backend} \
    --verbose 1 \
    --outdir ${lmexpdir} \
    --tensorboard-dir tensorboard/${lmexpname} \
    --train-label ${lmdatadir}/train.txt \
    --valid-label ${lmdatadir}/valid.txt \
    --test-label ${lmdatadir}/test.txt \
    --resume ${lm_resume} \
    --dict ${lmdict}
