#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}
SECONDS=0

# General configuration
stage=1          # Processes starts from the specified stage.
stop_stage=12    # Processes is stopped at the specified stage.
ngpu=1           # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1      # The number of nodes
dumpdir=dump     # Directory to dump features.
expdir=exp       # Directory to save experiments.

mnist=false  # Set true if using mnist data

# model related
num_channel=1 # Number of channels
dcgan_tag=    # Suffix to the result dir for asr model training.
dcgan_config= # Config for asr model training.
dcgan_args=   # Arguments for asr model training, e.g., "--max_epoch 10".
            # Note that it will overwrite args in asr config.

decode_model=valid.loss.best.pth # Model path for decoding e.g.,
                                 # decode_model=train.loss.best.pth
                                 # decode_model=3epoch.pth
                                 # decode_model=valid.acc.best.pth
                                 # decode_model=valid.loss.ave.pth

# Data preparation related
local_data_opts= # The options given to local/data.sh.

help_message=$(cat << EOF
Usage: $0
EOF
)

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh


# Set tag for naming of model directory
if [ -z "${dcgan_tag}" ]; then
    if [ -n "${dcgan_config}" ]; then
        dcgan_tag="$(basename "${dcgan_config}" .yaml)"
    else
        dcgan_tag="train"
    fi
    # Add overwritten arg's info
    if [ -n "${dcgan_args}" ]; then
        dcgan_tag+="$(echo "${dcgan_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
fi

# The directory used for training commands
dcgan_exp="${expdir}/dcgan_${dcgan_tag}"


# ========================== Main stages start from here. ==========================

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Data preparation"
    # [Task dependent] Need to create data.sh for new corpus
    local/data.sh ${local_data_opts}
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: DCGAN Training: train_set=${dumpdir}"
    if "${mnist}"; then
        train_type=mnist_train_64x64
        valid_type=mnist_test_64x64
    else
        # Transform the image to 64x64
        train_type=imagefolder_64x64
        valid_type=imagefolder_64x64
    fi

    log "ASR training started... log: '${dcgan_exp}/train.log'"
    # shellcheck disable=SC2086
    python3 -m espnet2.bin.launch \
        --cmd "${cuda_cmd} --name ${dcgan_exp}/train.log" \
        --log "${dcgan_exp}"/train.log \
        --ngpu "${ngpu}" \
        --num_nodes "${num_nodes}" \
        --init_file_prefix "${dcgan_exp}"/.dist_init_ \
        --multiprocessing_distributed true -- \
        python3 -m espnet2.bin.dcgan_train \
            --train_data_path_and_name_and_type "${dumpdir},image,${train_type}" \
            --valid_data_path_and_name_and_type "${dumpdir},image,${valid_type}" \
            --resume true \
            --num_channel "${num_channel}" \
            --output_dir "${dcgan_exp}" \
            --config "${dcgan_config}" ${dcgan_args}
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Decoding: training_dir=${dcgan_exp}"

    python3 -m espnet2.bin.dcgan_inference \
        --num_plots 10 \
        --output_dir "${dcgan_exp}"/decode \
        --model_file "${dcgan_exp}"/"${decode_model}" \
        --train_config "${dcgan_exp}"/config.yaml
fi
