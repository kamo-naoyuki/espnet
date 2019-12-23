#!/bin/bash

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

# General configuration
stage=1          # Processes starts from the specified stage.
stop_stage=100   # Processes is stopped at the specified stage.
ngpu=0           # The number of gpus ("0" uses cpu, otherwise use gpu).
nj=50            # The number of parallel jobs.
decode_nj=50     # The number of parallel jobs in decoding.
gpu_decode=false # Whether to perform gpu decoding.
dumpdir=dump     # Directory to dump features.
expdir=exp       # Directory to save experiments.

# Data preparation related
local_data_opts= # Options to be passed to local/data.sh.

# Feature extraction related
audio_format=flac # Audio format
fs=16000          # Sampling rate.
oov="<unk>"         # Out of vocabrary symbol.
blank="<blank>"     # CTC blank symbol
sos_eos="<sos/eos>" # sos and eos symbole

# Training related
train_config= # Config for training.
train_args=   # Arguments for training, e.g., "--max_epoch 1".
              # Note that it will overwrite args in train config.
tag=""        # Suffix for training directory.

# Decoding related
decode_config= # Config for decoding.
decode_args=   # Arguments for decoding, e.g., "--threshold 0.75".
               # Note that it will overwrite args in decode config.
decode_tag=""  # Suffix for decoding directory.
decode_model=eval.loss.best.pth # Model path for decoding e.g.,
                                # decode_model=train.loss.best.pt
                                # decode_model=3epoch/model.pt
                                # decode_model=eval.acc.best.pt
                                # decode_model=eval.loss.ave.pt
griffin_lim_iters=4 # the number of iterations of Griffin-Lim.

# [Task dependent] Set the datadir name created by local/data.sh
train_set=      # Name of training set.
dev_set=        # Name of development set.
eval_sets=      # Names of evaluation sets. Multiple items can be specified.
srctexts=       # Texts to create token list. Multiple items can be specified.
nlsyms_txt=none # Non-linguistic symbol list (needed if existing).
trans_type=char # Transcription type.

help_message=$(cat << EOF
Usage: $0 --train-set "<train_set_name>" --dev-set "<dev_set_name>" --eval_sets "<eval_set_names>" --srctexts "<srctexts>"

Options:
    # General configuration
    --stage      # Processes starts from the specified stage (default="${stage}").
    --stop_stage # Processes is stopped at the specified stage (default="${stop_stage}").
    --ngpu       # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --nj         # The number of parallel jobs (default="${nj}").
    --decode_nj  # The number of parallel jobs in decoding (default="${decode_nj}").
    --gpu_decode # Whether to perform gpu decoding (default="${gpu_decode}").
    --dumpdir    # Directory to dump features (default="${dumpdir}").
    --expdir     # Directory to save experiments (default="${expdir}").

    # Data prep related
    --local_data_opts # Options to be passed to local/data.sh (default="${local_data_opts}").

    # Feature extraction related
    --audio_format # Audio format (default="${audio_format}").
    --fs           # Sampling rate (default="${fs}").
    --oov          # Out of vocabrary symbol (default="${oov}").
    --blank        # CTC blank symbol (default="${blank}").
    --sos_eos=     # sos and eos symbole (default="${sos_eos}").

    # Training related
    --train_config # Config for training (default="${train_config}").
    --train_args   # Arguments for training, e.g., "--max_epoch 1" (default="${train_args}").
                   # Note that it will overwrite args in train config.
    --tag          # Suffix for training directory (default="${tag}").

    # Decoding related
    --decode_config     # Config for decoding (default="${decode_config}").
    --decode_args       # Arguments for decoding, e.g., "--threshold 0.75" (default="${decode_args}").
                        # Note that it will overwrite args in decode config.
    --decode_tag        # Suffix for decoding directory (default="${decode_tag}").
    --decode_model      # Model path for decoding (default=${decode_model}).
    --griffin_lim_iters # The number of iterations of Griffin-Lim (default=${griffin_lim_iters}).

    # [Task dependent] Set the datadir name created by local/data.sh.
    --train_set  # Name of training set (required).
    --dev_set    # Name of development set (required).
    --eval_sets  # Names of evaluation sets (required).
                 # Note that multiple items can be specified.
    --srctexts   # Texts to create token list (required).
                 # Note that multiple items can be specified.
    --nlsyms_txt # Non-linguistic symbol list (default="${nlsyms_txt}").
    --trans_type # Transcription type (default="${trans_type}").
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

data_feats="${dumpdir}/raw"

# Set tag for naming of model directory
if [ -z "${tag}" ]; then
    if [ -n "${train_config}" ]; then
        tag="$(basename "${train_config}" .yaml)"
    else
        tag="train"
    fi
    # Add overwritten arg's info
    if [ -n "${train_args}" ]; then
        tag+="$(echo "${train_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
fi
if [ -z "${decode_tag}" ]; then
    if [ -n "${decode_config}" ]; then
        decode_tag="$(basename "${decode_config}" .yaml)"
    else
        decode_tag=decode
    fi
    # Add overwritten arg's info
    if [ -n "${decode_args}" ]; then
        decode_tag+="$(echo "${decode_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
    decode_tag+="_$(echo "${decode_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
fi

# ========================== Main stages start from here. ==========================

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Data preparation for data/${train_set}, data/${dev_set}, etc."
    # [Task dependent] Need to create data.sh for new corpus
    local/data.sh ${local_data_opts}
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # TODO(kamo): Change kaldi-ark to npy or HDF5?
    log "Stage 2: Format wav.scp: data/ -> ${data_feats}/"
    # ====== Recreating "wav.scp" ======
    # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
    # shouldn't be used in training process.
    # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
    # and also it can also change the audio-format and sampling rate.
    # If nothing is need, then format_wav_scp.sh does nothing:
    # i.e. the input file format and rate is same as the output.

    for dset in "${train_set}" "${dev_set}" ${eval_sets}; do
        utils/copy_data_dir.sh data/"${dset}" "${data_feats}/${dset}"
        scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
            --audio-format "${audio_format}" --fs "${fs}" \
            "data/${dset}/wav.scp" "${data_feats}/${dset}"
    done

fi


token_list="data/token_list/${trans_type}/tokens.txt"
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Generate character level token_list from ${srctexts}"
    mkdir -p "$(dirname ${token_list})"
    # "nlsyms_txt" should be generated by local/data.sh if need

    { echo "${blank}"
      echo "${oov}"
      # shellcheck disable=SC2002
      cat ${srctexts} | cut -f 2- -d" " \
          | python -m espnet2.bin.tokenize \
                --token_type char --input - --output - \
                --non_language_symbols ${nlsyms_txt} \
                --write_vocabulary true
      echo "${sos_eos}"
    } > "${token_list}"
fi

# ========================== Data preparation is done here. ==========================



tts_exp="${expdir}/tts_${tag}"
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    _train_dir="${data_feats}/${train_set}"
    _dev_dir="${data_feats}/${dev_set}"
    log "Stage 4: TTS collect stats: train_set=${_train_dir}, dev_set=${_dev_dir}"

    _opts=
    if [ -n "${train_config}" ]; then
        # To generate the config file: e.g.
        #   % python -m espnet2.bin.tts_train --print_config --optim adam
        _opts+="--config ${train_config} "
    fi

    _scp=wav.scp
    # "sound" supports "wav", "flac", etc.
    _type=sound
    # FIXME(kamo): max_length is confusing name. How about fold_length?
    _max_length=80000

    # 1. Split the key file
    _logdir="${tts_exp}/stats/logdir"
    mkdir -p "${_logdir}"
    key_file="${_train_dir}/${_scp}"
    split_scps=""
    _nj=$((decode_nj<$(<${key_file} wc -l)?decode_nj:$(<${key_file} wc -l)))
    for n in $(seq ${_nj}); do
        split_scps+=" ${_logdir}/train.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_scps}

    key_file="${_dev_dir}/${_scp}"
    split_scps=""
    _nj=$((decode_nj<$(<${key_file} wc -l)?decode_nj:$(<${key_file} wc -l)))
    for n in $(seq ${_nj}); do
        split_scps+=" ${_logdir}/dev.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_scps}

    # 2. Submit jobs
    log "TTS collect_stats started... log: '${tts_exp}/train.log'"
    # shellcheck disable=SC2086
    ${train_cmd} JOB=1:"${_nj}" "${tts_exp}"/stats.JOB.log \
        python3 -m espnet2.bin.tts_train \
            --collect_stats true \
            --use_preprocessor true \
            --token_type char \
            --token_list "${token_list}" \
            --non_language_symbols "${nlsyms_txt}" \
            --normalize none \
            --batch_type const \
            --sort_in_batch none \
            --train_data_path_and_name_and_type "${_train_dir}/text,text,text" \
            --train_data_path_and_name_and_type "${_train_dir}/wav.scp,speech,${_type}" \
            --eval_data_path_and_name_and_type "${_dev_dir}/text,text,text" \
            --eval_data_path_and_name_and_type "${_dev_dir}/wav.scp,speech,${_type}" \
            --train_shape_file "${_logdir}/train.JOB.scp" \
            --eval_shape_file "${_logdir}/dev.JOB.scp" \
            --output_dir "${_logdir}/stats.JOB" \
            ${_opts} ${train_args}

    # 3. Aggregate shape files
    _opts=
    for i in $(seq "${_nj}"); do
        _opts+="--input_dir ${_logdir}/stats.${i} "
    done
    python -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${tts_exp}/stats"
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    _train_dir="${data_feats}/${train_set}"
    _dev_dir="${data_feats}/${dev_set}"
    log "Stage 4: TTS Training: train_set=${_train_dir}, dev_set=${_dev_dir}"

    _opts=
    if [ -n "${train_config}" ]; then
        # To generate the config file: e.g.
        #   % python -m espnet2.bin.tts_train --print_config --optim adam
        _opts+="--config ${train_config} "
    fi

    # "sound" supports "wav", "flac", etc.
    _type=sound
    # FIXME(kamo): max_length is confusing name. How about fold_length?
    _max_length=80000

    log "TTS training started... log: '${tts_exp}/train.log'"
    # shellcheck disable=SC2086
    ${cuda_cmd} --gpu "${ngpu}" "${tts_exp}"/train.log \
        python3 -m espnet2.bin.tts_train \
            --ngpu "${ngpu}" \
            --token_list "${_train_dir}/tokens.txt" \
            --use_preprocessor true \
            --token_type char \
            --token_list "${token_list}" \
            --non_language_symbols "${nlsyms_txt}" \
            --normalize global_mvn \
            --normalize_conf stats_file=${tts_exp}/stats/train/feats_stats.npz \
            --train_data_path_and_name_and_type "${_train_dir}/text,text,text" \
            --train_data_path_and_name_and_type "${_train_dir}/${_scp},speech,${_type}" \
            --eval_data_path_and_name_and_type "${_dev_dir}/text,text,text" \
            --eval_data_path_and_name_and_type "${_dev_dir}/${_scp},speech,${_type}" \
            --train_shape_file "${tts_exp}/stats/train/speech_shape" \
            --train_shape_file "${tts_exp}/stats/train/text_shape" \
            --eval_shape_file "${tts_exp}/stats/eval/speech_shape" \
            --eval_shape_file "${tts_exp}/stats/eval/text_shape" \
            --resume_epoch latest \
            --max_length 150 \
            --max_length ${_max_length} \
            --output_dir "${tts_exp}" \
            ${_opts} ${train_args}
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "Stage 5: Decoding: training_dir=${tts_exp}"

    if ${gpu_decode}; then
        _cmd=${cuda_cmd}
        _ngpu=1
    else
        _cmd=${decode_cmd}
        _ngpu=0
    fi

    _opts=
    if [ -n "${decode_config}" ]; then
        _opts+="--config ${decode_config} "
    fi

    for dset in "${dev_set}" ${eval_sets}; do
        _data="${data_feats}/${dset}"
        _dir="${tts_exp}/${decode_tag}_${dset}"
        _logdir="${_dir}/log"
        mkdir -p "${_logdir}"

        # 1. Split the key file
        key_file=${_data}/text
        split_scps=""
        _nj=$((decode_nj<$(<${key_file} wc -l)?decode_nj:$(<${key_file} wc -l)))
        for n in $(seq ${_nj}); do
            split_scps+=" ${_logdir}/keys.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Submit decoding jobs
        log "Decoding started... log: '${_logdir}/tts_decode.*.log'"
        # shellcheck disable=SC2086
        # NOTE(kan-bayashi): --key_file is useful when we want to use multiple data
        ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/tts_decode.JOB.log \
            python3 -m espnet2.bin.tts_decode \
                --ngpu "${_ngpu}" \
                --data_path_and_name_and_type "${_data}/text,text,text" \
                --key_file "${_logdir}"/keys.JOB.scp \
                --model_file "${tts_exp}"/"${decode_model}" \
                --train_config "${tts_exp}"/config.yaml \
                --output_dir "${_logdir}"/output.JOB \
                --griffin_lim_iters "${griffin_lim_iters}" \
                ${_opts} ${decode_args}

        # 3. Concatenates the output files from each jobs
        mkdir -p "${_dir}"/{norm,denorm,wav}
        for i in $(seq "${_nj}"); do
             cat "${_logdir}/output.${i}/norm/feats.scp"
        done | LC_ALL=C sort -k1 > "${_dir}/norm/feats.scp"
        for i in $(seq "${_nj}"); do
             cat "${_logdir}/output.${i}/denorm/feats.scp"
        done | LC_ALL=C sort -k1 > "${_dir}/denorm/feats.scp"
        for i in $(seq "${_nj}"); do
            mv -u "${_logdir}/output.${i}"/wav/*.wav "${_dir}"/wav
            rm -rf "${_logdir}/output.${i}/wav"
        done
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"