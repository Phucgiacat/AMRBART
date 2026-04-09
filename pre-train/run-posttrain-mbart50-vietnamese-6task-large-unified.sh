RootDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

dataset=ViAMR

# ===== FIX HF (optional nhưng nên có) =====
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# ===== MODEL LOCAL (PHẢI ĐẶT TRƯỚC) =====
MODEL=/content/mbart-large-50

# ===== DATA =====
if [ -d "/content/drive/MyDrive/AMRBART/data/$dataset" ]; then
  DataPath=/content/drive/MyDrive/AMRBART/data/$dataset
  echo "Using dataset from Google Drive: $DataPath"
else
  DataPath=$RootDir/data/$dataset
  echo "Using local dataset: $DataPath"
fi

interval=1
lr=5e-5

outpath=output/${dataset}-mbart50-large-Unifiedtextinf-JointDenoise-6task-${lr}-AMREOS
DataCache=$DataPath/.cache

DrivePath=/content/drive/MyDrive/AMRBART/${dataset}-mbart50-large-6task-${lr}
mkdir -p "$DrivePath"
mkdir -p $outpath

echo "OutputDir: $outpath"
echo "DrivePath: $DrivePath"

mkdir -p ${DataCache}
export HF_DATASETS_CACHE=$DataCache
export TRANSFORMERS_CACHE=$DataCache

# ===== RESUME =====
RESUME_FLAG=""
if [ -f "$DrivePath/pytorch_model.bin" ]; then
  echo "Resuming from Drive checkpoint: $DrivePath"
  RESUME_FLAG="--model_name_or_path $DrivePath"
else
  RESUME_FLAG="--model_name_or_path $MODEL"
fi

# ===== PYTHON =====
PYTHON=${VENV:-/content/AMRBART/pre-train/.venv}/bin/python
echo "Using Python: $PYTHON"
$PYTHON -c "import sys; print(sys.executable, sys.version)"

# ===== TRAIN =====
CUDA_VISIBLE_DEVICES=0 $PYTHON -m torch.distributed.run --nproc_per_node=1 run_multitask_unified_pretraining.py \
  --train_file $DataPath/train.jsonl \
  --val_file $DataPath/val.jsonl \
  --test_file $DataPath/test.jsonl \
  --output_dir $outpath \
  --mlm \
  --mlm_amr \
  --mlm_text \
  --mlm_amr_plus_text \
  --mlm_text_plus_amr \
  --mlm_joint_to_amr \
  --mlm_joint_to_text \
  --block_size 512 \
  --per_gpu_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --model_type mbart \
  $RESUME_FLAG \
  --save_total_limit 2 \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --num_train_epochs 15 \
  --learning_rate $lr \
  --joint_train_interval $interval \
  --warmup_steps 2500 \
  --max_steps 8000 \
  --logging_steps 500 \
  --save_steps 2000 \
  --overwrite_output_dir \
  2>&1 | tee $outpath/run.log

# ===== SYNC =====
echo "Syncing checkpoints to Google Drive..."
rsync -av --progress $outpath/ "$DrivePath/"
echo "Checkpoints saved to $DrivePath"