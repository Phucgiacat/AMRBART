#!/bin/bash
# ==============================================================
# Two-Stage Inference for AMRBART-RGL
#
# Usage:
#   bash inference-amr-2stage.sh [CHECKPOINT_PATH]
#
# Example:
#   bash inference-amr-2stage.sh /content/AMRBART/checkpoint-3124
# ==============================================================

export CUDA_VISIBLE_DEVICES=0
RootDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

Dataset=ViAMR

BasePath=/content/AMRBART
DataPath=$BasePath/data/ViAMR/$Dataset

ModelCate=AMRBART-large

# Model checkpoint (first argument or default)
MODEL=${1:-/content/AMRBART/checkpoint-3124}

# Tokenizer (always use the pretrained tokenizer for vocab)
TOKENIZER=xfbai/AMRBART-large-v2

ModelCache=$BasePath/.cache

# Gold AMR file for smatch evaluation
GoldAMR=$DataPath/test-gold.amr

OutputDir=${RootDir}/outputs/TwoStage-$Dataset-${ModelCate}

if [ ! -d ${OutputDir} ];then
  mkdir -p ${OutputDir}
else
  echo "${OutputDir} already exists, overwriting..."
  rm -rf ${OutputDir}
  mkdir -p ${OutputDir}
fi

echo "=============================================="
echo "Two-Stage Inference for AMRBART-RGL"
echo "=============================================="
echo "Model:      ${MODEL}"
echo "Tokenizer:  ${TOKENIZER}"
echo "Test file:  ${DataPath}/test.jsonl"
echo "Gold AMR:   ${GoldAMR}"
echo "Output:     ${OutputDir}"
echo "=============================================="

# Use test.jsonl from the NLR data directory
# (contains "sent" field for each sentence)
TEST_FILE=${DataPath}/test.jsonl

# Check test file exists, fallback to alternate locations
if [ ! -f "${TEST_FILE}" ]; then
    TEST_FILE=${BasePath}/data/ViAMR/ViAMR_dfs_NLR/test.jsonl
fi

if [ ! -f "${TEST_FILE}" ]; then
    echo "ERROR: Cannot find test.jsonl at ${DataPath} or ViAMR_dfs_NLR"
    exit 1
fi

echo "Using test file: ${TEST_FILE}"

python3 $RootDir/two_stage_inference.py \
    --model_name_or_path $MODEL \
    --tokenizer_name $TOKENIZER \
    --test_file $TEST_FILE \
    --output_dir $OutputDir \
    --cache_dir $ModelCache \
    --max_source_length 400 \
    --max_target_length 512 \
    --generation_num_beams 5 \
    --generation_max_length 512 \
    --per_device_eval_batch_size 4 \
    --unified_input True \
    --fp16 \
    --gold_amr_file $GoldAMR \
    2>&1 | tee $OutputDir/run.log

echo "=============================================="
echo "Done! Results in: ${OutputDir}"
echo "  Stage 1: ${OutputDir}/stage1_predictions.txt"
echo "  Stage 2: ${OutputDir}/stage2_predictions.txt"
echo "  Final:   ${OutputDir}/generated_predictions_penman.txt"
echo "=============================================="
