#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $@"
}
help_message=$(cat << EOF
Usage: $0 <text> <token_list> <output-dir>

Generate token, token_int and token_shape from text

Options:
    --type (str): The tokenize level. Select either one of "bpe", "char" or "word".
    --bpemodel (str): Give bpemodel with --type bpe.
    --nlsyms: non-linguistic symbol list
    --oov (str): default is "<unk>"

EOF
)
SECONDS=0

type=bpe
bpemodel=
nlsyms=


oov="<unk>"


log "$0 $*"
. ./utils/parse_options.sh || exit 1;


if [ $# -ne 3 ]; then
    log "${help_message}"
    log "Invalid arguments"
fi
. ./path.sh

text=$1
token_list=$2
dir=$3

for f in "${text}" "${token_list}"; do
    if [ ! -f "${f}" ]; then
        log "${help_message}"
        log "Error: No such file: '${f}'"
        exit 1
    fi
done


# TODO(kamo): To be pythonized?

mkdir -p "${dir}"

# 0. token_type
echo "${type}" > "${dir}/token_type"
if [ "${text}" != ${dir}/text ]; then
    cp "${text}" "${dir}/text"
fi


# 1. Prepare token
if [ "${type}" = bpe ]; then
    if [ ! -f "${bpemodel}" ]; then
        log "${help_message}"
        log "Error: No such file: '${bpemodel}'"
        exit 1
    fi

    paste -d " " \
        <(awk '{print $1}' "${text}") \
        <(cut -f 2- -d" " "${text}" | spm_encode --model="${bpemodel}" --output_format=piece) \
            > "${dir}"/token


elif [ "${type}" = char ]; then
    trans_type=char
    # non-linguistic symbol list

    # token:
    # uttidA h e l l o
    if [ -n "${nlsyms}" ]; then
        pyscripts/text/text2token.py -s 1 -n 1 -l "${nlsyms}" "${text}" \
            --trans_type "${trans_type}" > "${dir}/token"
    else
        pyscripts/text/text2token.py -s 1 -n 1 "${text}" \
            --trans_type "${trans_type}" > "${dir}/token"
    fi

elif [ "${type}" = word ]; then
    cp "${text}" "${dir}/token"

else
    log "${help_message}"
    log "Error: not supported --type '${type}'"
    exit 1
fi


# 2. Recreate "tokens.txt":
# 0 is blank, 1~Nvocab: token, Nvocab+1: SOS/EOS
cp "${token_list}" "${dir}"/tokens.txt


# 3. Create "token_int"
# Create token2id for mapping symbol to integer-id, which looks like
#    <blank> 0
#    A 1
#    ...
<"${dir}/tokens.txt" awk '{ print($1,NR-1)}' >"${dir}/token2id"
<"${dir}"/token utils/sym2int.pl --map-oov "${oov}" -f 2- "${dir}/token2id" \
    > "${dir}/token_int"


# 4. Create "token_shape", which looks like...
#   uttidA 20,32
#   uttidB 12,32
# where the first column indicates the number of tokens in the text
# and the second is the vocabulary size.
nvocab="$(<"${dir}"/tokens.txt wc -l)"
<"${dir}"/token_int awk -v nvocab="${nvocab}" '{print($1,NF-1 "," nvocab)}' > "${dir}"/token_shape


log "Successfully finished. [elapsed=${SECONDS}s]"
