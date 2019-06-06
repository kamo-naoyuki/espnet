#!/bin/bash
# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -euo pipefail

cmd=run.pl

backend=pytorch
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot
seed=1

# config files
preprocess_config=  # use conf/specaug.yaml for data augmentation
train_config=conf/train.yaml

dict=data/lang_1char/units.txt
n_average=10 # use 1 for RNN models

# exp tag
tag="" # tag for managing experiments.

echo "$0 $*"  # Print the command line for logging
. utils/parse_options.sh || exit 1;

if [ $# -ne 2 ]; then
    cat << EOF
Usage: $0 <train_set> <dev>
e.g. $0 --dict data/lang_1char/units.txt data/train data/dev
EOF
    exit 1
fi

. ./path.sh

train_set=$1
dev_set=$2


if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})_$(basename ${preprocess_config%.*})
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

opts=
if [ ! -z ${preprocess_yaml} ]; then
    opts+="--preprocess-conf ${preprocess_config} "
fi


${cmd} --gpu ${ngpu} ${expdir}/train.log \
    asr_train.py \
    ${opts} \
    --config ${train_config} \
    --ngpu ${ngpu} \
    --backend ${backend} \
    --outdir ${expdir}/results \
    --tensorboard-dir tensorboard/${expname} \
    --debugmode ${debugmode} \
    --debugdir ${expdir} \
    --minibatches ${N} \
    --verbose ${verbose} \
    --resume ${resume} \
    --seed ${seed} \
    --dict ${dict} \
    --train-json ${train_set}/data.json \
    --valid-json ${dev_set}/data.json


if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
    recog_model=model.last${n_average}.avg.best
    average_checkpoints.py --backend ${backend} \
                           --snapshots ${expdir}/results/snapshot.ep.* \
                           --out ${expdir}/results/${recog_model} \
                           --num ${n_average}
fi
