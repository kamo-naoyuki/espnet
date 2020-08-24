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
stage=1              # Processes starts from the specified stage.
stop_stage=10000     # Processes is stopped at the specified stage.
skip_data_prep=false # Skip data preparation stages
skip_train=false     # Skip training stages
skip_eval=false      # Skip decoding and evaluation stages
skip_upload=true     # Skip packing and uploading stages
ngpu=1               # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1          # The number of nodes
nj=32                # The number of parallel jobs.
inference_nj=32      # The number of parallel jobs in decoding.
gpu_inference=true   # Whether to perform gpu decoding.
dumpdir=dump         # Directory to dump features.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands

# Data preparation related
local_data_opts= # The options given to local/data.sh.

# Speed perturbation related
speed_perturb_factors=  # perturbation factors, e.g. "0.9 1.0 1.1" (separated by space).

# Audio format related
feats_type=raw         # Feature type (raw or fbank_pitch).
audio_format=flac      # Audio format (only in feats_type=raw).
fs=16k                 # Sampling rate.
min_wav_duration=1   # Minimum duration in second
max_wav_duration=500   # Maximum duration in second

s=16k                 # Sampling rate.

# sre model related
sre_tag=    # Suffix to the result dir for sre model training.
sre_exp=    # Specify the direcotry path for sre experiment. If this option is specified, sre_tag is ignored.
sre_config= # Config for sre model training.
sre_args=   # Arguments for sre model training, e.g., "--max_epoch 10".
            # Note that it will overwrite args in sre config.
num_splits=1   # Number of splitting for sre corpus

# Decoding related
inference_tag=    # Suffix to the result dir for decoding.
inference_config= # Config for decoding.
inference_args=   # Arguments for decoding, e.g., "--lm_weight 0.1".
                  # Note that it will overwrite args in inference config.
inference_sre_model=train.acc.best.pth # sre model path for decoding.
                                       # e.g.
                                       # inference_sre_model=train.loss.best.pth
                                       # inference_sre_model=3epoch.pth
                                       # inference_sre_model=valid.score.best.pth
                                       # inference_sre_model=valid.score.ave.pth
download_model= # Download a model from Model Zoo and use it for decoding

# [Task dependent] Set the datadir name created by local/data.sh
train_set=       # Name of training set.
valid_set=       # Name of validation set used for monitoring/tuning network training
test_sets=       # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.
lang=noinfo      # The language type of corpus

help_message=$(cat << EOF
Usage: $0 --train-set <train_set_name> --valid-set <valid_set_name> --test_sets <test_set_names> --srctexts <srctexts >

Options:
    # General configuration
    --stage          # Processes starts from the specified stage (default="${stage}").
    --stop_stage     # Processes is stopped at the specified stage (default="${stop_stage}").
    --skip_data_prep # Skip data preparation stages (default="${skip_data_prep}").
    --skip_train     # Skip training stages (default="${skip_train}").
    --skip_eval      # Skip decoding and evaluation stages (default="${skip_eval}").
    --skip_upload    # Skip packing and uploading stages (default="${skip_upload}").
    --ngpu           # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes      # The number of nodes
    --nj             # The number of parallel jobs (default="${nj}").
    --inference_nj   # The number of parallel jobs in decoding (default="${inference_nj}").
    --gpu_inference  # Whether to perform gpu decoding (default="${gpu_inference}").
    --dumpdir        # Directory to dump features (default="${dumpdir}").
    --expdir         # Directory to save experiments (default="${expdir}").
    --python         # Specify python to execute espnet commands (default="${python}").

    # Data preparation related
    --local_data_opts # The options given to local/data.sh (default="${local_data_opts}").

    # Speed perturbation related
    --speed_perturb_factors   # speed perturbation factors, e.g. "0.9 1.0 1.1" (separated by space, default="${speed_perturb_factors}").

    # Audio format related
    --feats_type      # Feature type (raw, fbank_pitch or extracted, default="${feats_type}").
    --audio_format    # Audio format (only in feats_type=raw, default="${audio_format}").
    --fs              # Sampling rate (default="${fs}").
    --min_wav_duration # Minimum duration in second (default="${min_wav_duration}").
    --max_wav_duration # Maximum duration in second (default="${max_wav_duration}").

    # sre model related
    --sre_tag    # Suffix to the result dir for sre model training (default="${sre_tag}").
    --sre_exp    # Specify the direcotry path for sre experiment. If this option is specified, sre_tag is ignored (default="${sre_exp}").
    --sre_config # Config for sre model training (default="${sre_config}").
    --sre_args   # Arguments for sre model training, e.g., "--max_epoch 10" (default="${sre_args}").
                 # Note that it will overwrite args in sre config.
    --num_splits=1   # Number of splitting for srecorpus  (default="${num_splits}").

    # Decoding related
    --inference_tag       # Suffix to the result dir for decoding (default="${inference_tag}").
    --inference_config    # Config for decoding (default="${inference_config}").
    --inference_args      # Arguments for decoding, e.g., "--lm_weight 0.1" (default="${inference_args}").
                       # Note that it will overwrite args in inference config.
    --inference_sre_model # sre model path for decoding (default="${inference_sre_model}").
    --download_model   # Download a model from Model Zoo and use it for decoding  (default="${download_model}").

    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set     # Name of training set (required).
    --valid_set=    # Name of validation set used for monitoring/tuning network training (required).
    --test_sets=    # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified (required).
    --lang              # The language type of corpus (default=${lang}).
EOF
)

log "$0 $*"
# Save command line args for logging (they will be lost after utils/parse_options.sh)
run_args=$(pyscripts/utils/print_args.py $0 "$@")
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh


# Check required arguments
[ -z "${train_set}" ] && { log "${help_message}"; log "Error: --train_set is required"; exit 2; };
[ -z "${valid_set}" ] && { log "${help_message}"; log "Error: --valid_set is required"; exit 2; };
[ -z "${test_sets}" ] && { log "${help_message}"; log "Error: --test_sets is required"; exit 2; };

data_feats=${dumpdir}/raw

# Set tag for naming of model directory
if [ -z "${sre_tag}" ]; then
    if [ -n "${sre_config}" ]; then
        sre_tag="$(basename "${sre_config}" .yaml)"
    else
        sre_tag="train"
    fi
    # Add overwritten arg's info
    if [ -n "${sre_args}" ]; then
        sre_tag+="$(echo "${sre_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
fi
if [ -z "${inference_tag}" ]; then
    if [ -n "${inference_config}" ]; then
        inference_tag="$(basename "${inference_config}" .yaml)"
    else
        inference_tag=inference
    fi
    # Add overwritten arg's info
    if [ -n "${inference_args}" ]; then
        inference_tag+="$(echo "${inference_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
    inference_tag+="_sre_model_$(echo "${inference_sre_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
fi

# The directory used for training commands
if [ -z "${sre_exp}" ]; then
    sre_exp="${expdir}/sre_${sre_tag}"
fi
if [ -n "${speed_perturb_factors}" ]; then
    sre_exp="${sre_exp}_sp"
fi

# ========================== Main stages start from here. ==========================
if ! "${skip_data_prep}"; then
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        log "Stage 1: Data preparation for data/${train_set}, data/${valid_set}, etc."
        # [Task dependent] Need to create data.sh for new corpus
        local/data.sh ${local_data_opts}
    fi

    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        if [ -n "${speed_perturb_factors}" ]; then
           log "Stage 2: Speed perturbation: data/${train_set} -> data/${train_set}_sp"
           for factor in ${speed_perturb_factors}; do
               if [[ $(bc <<<"${factor} != 1.0") == 1 ]]; then
                   scripts/utils/perturb_data_dir_speed.sh "${factor}" "data/${train_set}" "data/${train_set}_sp${factor}"
                   _dirs+="data/${train_set}_sp${factor} "
               else
                   # If speed factor is 1, same as the original
                   _dirs+="data/${train_set} "
               fi
           done
           utils/combine_data.sh "data/${train_set}_sp" ${_dirs}
        else
           log "Skip stage 2: Speed perturbation"
        fi
    fi

    if [ -n "${speed_perturb_factors}" ]; then
        train_set="${train_set}_sp"
    fi


    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        log "Stage 3: Format wav.scp: data/ -> ${data_feats}"

        # ====== Recreating "wav.scp" ======
        # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
        # shouldn't be used in training process.
        # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
        # and it can also change the audio-format and sampling rate.
        # If nothing is need, then format_wav_scp.sh does nothing:
        # i.e. the input file format and rate is same as the output.

        for dset in "${train_set}" "${valid_set}" ${test_sets}; do
            if [ "${dset}" = "${train_set}" ]; then
                _suf="/org"
            else
                _suf=""
            fi
            utils/copy_data_dir.sh data/"${dset}" "${data_feats}${_suf}/${dset}"
            # Copy extra files
            if [ -e data/"${dset}"/label ]; then
                cp data/"${dset}/label" "${data_feats}${_suf}/${dset}/label"
            fi

            rm -f ${data_feats}${_suf}/${dset}/{segments,wav.scp,reco2file_and_channel}
            _opts=
            if [ -e data/"${dset}"/segments ]; then
                # "segments" is used for splitting wav files which are written in "wav".scp
                # into utterances. The file format of segments:
                #   <segment_id> <record_id> <start_time> <end_time>
                #   "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5"
                # Where the time is written in seconds.
                _opts+="--segments data/${dset}/segments "
            fi
            # shellcheck disable=SC2086
            scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
                "data/${dset}/wav.scp" "${data_feats}${_suf}/${dset}"

            # Format for ref.scp
            if [ -e "data/${dset}/ref.scp" ]; then
                if [ -e data/"${dset}"/segments ]; then
                    # "segments" is used for splitting wav files which are written in "wav".scp
                    # into utterances. The file format of segments:
                    #   <segment_id> <record_id> <start_time> <end_time>
                    #   "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5"
                    # Where the time is written in seconds.
                    _opts+="--segments data/${dset}/ref_segments "
                fi
                # shellcheck disable=SC2086
                scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                    --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
                    --out_filename ref.scp \
                    "data/${dset}/ref.scp" "${data_feats}${_suf}/${dset}"
            fi

        done

    fi


    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        log "Stage 4: Remove long/short data: ${data_feats}/org -> ${data_feats}"

        # NOTE(kamo): Applying training data only
        # shellcheck disable=SC2066
        for dset in "${train_set}"; do

            # Copy data dir
            utils/copy_data_dir.sh "${data_feats}/org/${dset}" "${data_feats}/${dset}"
            # Copy extra files
            for f in ref.scp label; do
                if [ -e "${data_feats}/org/${dset}/${f}" ]; then
                    cp "${data_feats}/org/${dset}/${f}" "${data_feats}/${dset}/${f}"
                fi
            done

            # Remove short utterances
            _fs=$(python3 -c "import humanfriendly as h;print(h.parse_size('${fs}'))")
            _min_length=$(python3 -c "print(int(${min_wav_duration} * ${_fs}))")
            _max_length=$(python3 -c "print(int(${max_wav_duration} * ${_fs}))")

            # utt2num_samples is created by format_wav_scp.sh
            <"${data_feats}/org/${dset}/utt2num_samples" \
                awk -v min_length="${_min_length}" -v max_length="${_max_length}" \
                    '{ if ($2 > min_length && $2 < max_length ) print $0; }' \
                    >"${data_feats}/${dset}/utt2num_samples"
            <"${data_feats}/org/${dset}/wav.scp" \
                utils/filter_scp.pl "${data_feats}/${dset}/utt2num_samples"  \
                >"${data_feats}/${dset}/wav.scp"

            # fix_data_dir.sh leaves only utts which exist in all files
            utils/fix_data_dir.sh "${data_feats}/${dset}"
        done

    fi

else
    log "Skip the stages for data preparation"
fi

# ========================== Data preparation is done here. ==========================


if ! "${skip_train}"; then
    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        log "Stage 5: SRE Training: train_set=${data_feats}/${train_set}, valid_set=${data_feats}/${valid_set}"

        _opts=
        if [ -n "${sre_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.sre_train --print_config --optim adam
            _opts+="--config ${sre_config} "
        fi

        if [ "${num_splits}" -gt 1 ]; then
            # If you met a memory error when parsing text files, this option may help you.
            # The corpus is split into subsets and each subset is used for training one by one in order,
            # so the memory footprint can be limited to the memory required for each dataset.

            _split_dir="${sre_exp}/splits${num_splits}"
            if [ ! -f "${_split_dir}/.done" ]; then
                rm -f "${_split_dir}/.done"
                ${python} -m espnet2.bin.split_scps \
                  --scps "${data_feats}/${train_set}/wav.scp" \
                  --num_splits "${num_splits}" \
                  --output_dir "${_split_dir}"
                touch "${_split_dir}/.done"
            else
                log "${_split_dir}/.done exists. Spliting is skipped"
            fi

            _opts+="--train_data_path_and_name_and_type ${_split_dir}/${train_set}/wav.scp,speech,sound "
            _opts+="--multiple_iterator true "

        else
            _opts+="--train_data_path_and_name_and_type ${data_feats}/${train_set}/wav.scp,speech,sound "
        fi

        log "Generate '${sre_exp}/run.sh'. You can resume the process from stage 5 using this script"
        mkdir -p "${sre_exp}"; echo "${run_args} --stage 5 \"\$@\"; exit \$?" > "${sre_exp}/run.sh"; chmod +x "${sre_exp}/run.sh"

        # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case
        log "sre training started... log: '${sre_exp}/train.log'"
        if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
            # SGE can't include "/" in a job name
            jobname="$(basename ${sre_exp})"
        else
            jobname="${sre_exp}/train.log"
        fi

        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.launch \
            --cmd "${cuda_cmd} --name ${jobname}" \
            --log "${sre_exp}"/train.log \
            --ngpu "${ngpu}" \
            --num_nodes "${num_nodes}" \
            --init_file_prefix "${sre_exp}"/.dist_init_ \
            --multiprocessing_distributed true -- \
            ${python} -m espnet2.bin.sre_train \
                --valid_data_path_and_name_and_type "${data_feats}/${valid_set}/wav.scp,speech,sound" \
                --valid_data_path_and_name_and_type "${data_feats}/${valid_set}/ref.scp,reference,sound" \
                --valid_data_path_and_name_and_type "${data_feats}/${valid_set}/label,label,text_int" \
                --valid_shape_file "${data_feats}/${valid_set}/wav.scp" \
                --utt2spk "${data_feats}/${train_set}/utt2spk" \
                --fs "${fs}" \
                --resume true \
                --output_dir "${sre_exp}" \
                ${_opts} ${sre_args}

    fi
else
    log "Skip the training stages"
fi


if [ -n "${download_model}" ]; then
    log "Use ${download_model} for decoding and evaluation"
    sre_exp="${expdir}/${download_model}"
    mkdir -p "${sre_exp}"

    # If the model already exists, you can skip downloading
    espnet_model_zoo_download --unpack true "${download_model}" > "${sre_exp}/config.txt"

    # Get the path of each file
    _sre_model_file=$(<"${sre_exp}/config.txt" sed -e "s/.*'sre_model_file': '\([^']*\)'.*$/\1/")
    _sre_train_config=$(<"${sre_exp}/config.txt" sed -e "s/.*'sre_train_config': '\([^']*\)'.*$/\1/")

    # Create symbolic links
    ln -sf "${_sre_model_file}" "${sre_exp}"
    ln -sf "${_sre_train_config}" "${sre_exp}"
    inference_sre_model=$(basename "${_sre_model_file}")

fi


if ! "${skip_eval}"; then
    if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
        log "Stage 6: Decoding: training_dir=${sre_exp}"

        if ${gpu_inference}; then
            _cmd="${cuda_cmd}"
            _ngpu=1
        else
            _cmd="${decode_cmd}"
            _ngpu=0
        fi

        _opts=
        if [ -n "${inference_config}" ]; then
            _opts+="--config ${inference_config} "
        fi

        # 1. Generate run.sh
        log "Generate '${sre_exp}/${inference_tag}/run.sh'. You can resume the process from stage 6 using this script"
        mkdir -p "${sre_exp}/${inference_tag}"; echo "${run_args} --stage 6 \"\$@\"; exit \$?" > "${sre_exp}/${inference_tag}/run.sh"; chmod +x "${sre_exp}/${inference_tag}/run.sh"

        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _dir="${sre_exp}/${inference_tag}/${dset}"
            _logdir="${_dir}/logdir"
            mkdir -p "${_logdir}"

            # 2. Split the key file
            key_file="${_data}"/wav.scp
            split_scps=""
            _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")
            for n in $(seq "${_nj}"); do
                split_scps+=" ${_logdir}/keys.${n}.scp"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${key_file}" ${split_scps}

            # 3. Submit decoding jobs
            log "Decoding started... log: '${_logdir}/sre_inference.*.log'"
            # shellcheck disable=SC2086
            ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/sre_compute_embedding.JOB.log \
                ${python} -m espnet2.bin.sre_inference \
                    --ngpu "${_ngpu}" \
                    --data_path_and_name_and_type "${data_feats}/${valid_set}/wav.scp,speech,sound" \
                    --data_path_and_name_and_type "${data_feats}/${valid_set}/ref.scp,reference,sound" \
                    --data_path_and_name_and_type "${data_feats}/${valid_set}/label,label,text_int" \
                    --key_file "${_logdir}"/keys.JOB.scp \
                    --train_config "${sre_exp}"/config.yaml \
                    --model_file "${sre_exp}"/"${inference_sre_model}" \
                    --output_dir "${_logdir}"/output.JOB \
                    ${_opts} ${inference_args}

            # 4. Concatenate results
            for f in score label; do
                for i in $(seq "${_nj}"); do
                    cat "${_logdir}/output.${i}/${f}"
                done | LC_ALL=C sort -k1 >"${_dir}/${f}"
            done

            # 5. Calc EER
            ${python} -m espnet2.bin.calc_eer \
                --label "${_dir}"/label \
                --score "${_dir}"/score \
                --output_dir "${_dir}"

        done

        # 6. Show results in Markdown syntax
        scripts/utils/show_sre_result.sh "${sre_exp}" > "${sre_exp}"/RESULTS.md
        cat "${sre_exp}"/RESULTS.md
    fi

else
    log "Skip the evaluation stages"
fi


packed_model="${sre_exp}/${sre_exp##*/}_${inference_sre_model%.*}.zip"
if ! "${skip_upload}"; then
    if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
        log "Stage 7: Pack model: ${packed_model}"

        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.pack sre \
            --sre_train_config "${sre_exp}"/config.yaml \
            --sre_model_file "${sre_exp}"/"${inference_sre_model}" \
            --option "${sre_exp}"/RESULTS.md \
            --outpath "${packed_model}"
    fi


    if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
        log "Stage 8: Upload model to Zenodo: ${packed_model}"

        # To upload your model, you need to do:
        #   1. Sign up to Zenodo: https://zenodo.org/
        #   2. Create access token: https://zenodo.org/account/settings/applications/tokens/new/
        #   3. Set your environment: % export ACCESS_TOKEN="<your token>"

        if command -v git &> /dev/null; then
            _creator_name="$(git config user.name)"
            _checkout="
git checkout $(git show -s --format=%H)"

        else
            _creator_name="$(whoami)"
            _checkout=""
        fi
        # /some/where/espnet/egs2/foo/sre1/ -> foo/sre1
        _task="$(pwd | rev | cut -d/ -f2 | rev)"
        # foo/sre1 -> foo
        _corpus="${_task%/*}"
        _model_name="${_creator_name}/${_corpus}_$(basename ${packed_model} .zip)"

        # Generate description file
        cat << EOF > "${sre_exp}"/description
This model was trained by ${_creator_name} using ${_task} recipe in <a href="https://github.com/espnet/espnet/">espnet</a>.
<p>&nbsp;</p>
<ul>
<li><strong>Python API</strong><pre><code class="language-python">See https://github.com/espnet/espnet_model_zoo</code></pre></li>
<li><strong>Evaluate in the recipe</strong><pre>
<code class="language-bash">git clone https://github.com/espnet/espnet
cd espnet${_checkout}
pip install -e .
cd $(pwd | rev | cut -d/ -f1-3 | rev)
./run.sh --skip_data_prep false --skip_train true --download_model ${_model_name}</code>
</pre></li>
<li><strong>Results</strong><pre><code>$(cat "${sre_exp}"/RESULTS.md)</code></pre></li>
<li><strong>sre config</strong><pre><code>$(cat "${sre_exp}"/config.yaml)</code></pre></li>
</ul>
EOF

        # NOTE(kamo): The model file is uploaded here, but not published yet.
        #   Please confirm your record at Zenodo and publish it by yourself.

        # shellcheck disable=SC2086
        espnet_model_zoo_upload \
            --file "${packed_model}" \
            --title "ESPnet2 pretrained model, ${_model_name}, fs=${fs}, lang=${lang}" \
            --description_file "${sre_exp}"/description \
            --creator_name "${_creator_name}" \
            --license "CC-BY-4.0" \
            --use_sandbox false \
            --publish false
    fi
else
    log "Skip the uploading stages"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
