RootDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

dataset=ViAMR

# On Colab: read dataset from Google Drive
if [ -d "/content/drive/MyDrive/AMRBART/data/$dataset" ]; then
  DataPath=/content/drive/MyDrive/AMRBART/data/$dataset
  echo "Using dataset from Google Drive: $DataPath"
else
  DataPath=$RootDir/data/$dataset
  echo "Using local dataset: $DataPath"
fi

MODEL=facebook/mbart-large-50
interval=1

lr=5e-5

outpath=output/${dataset}-mbart50-large-Unifiedtextinf-JointDenoise-6task-${lr}-AMREOS
DataCache=$DataPath/.cache

# Google Drive checkpoint path for Colab persistence
DrivePath=/content/drive/MyDrive/AMRBART/${dataset}-mbart50-large-6task-${lr}
mkdir -p "$DrivePath"

mkdir -p $outpath
echo "OutputDir: $outpath"
echo "DrivePath: $DrivePath"

if [ ! -d ${DataCache} ];then
  mkdir -p ${DataCache}
fi

export HF_DATASETS_CACHE=$DataCache

# Resume from Drive checkpoint if available
RESUME_FLAG=""
if [ -f "$DrivePath/pytorch_model.bin" ]; then
  echo "Resuming from Drive checkpoint: $DrivePath"
  RESUME_FLAG="--model_name_or_path $DrivePath"
else
  RESUME_FLAG="--model_name_or_path $MODEL"
fi

# Colab: single GPU setup
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 run_multitask_unified_pretraining.py \
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
  --per_gpu_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --model_type "facebook/mbart-large-50" \
  $RESUME_FLAG \
  --save_total_limit 2 \
  --do_train \
  --do_eval \
  --evaluate_during_training  \
  --num_train_epochs 15  \
  --learning_rate $lr \
  --joint_train_interval $interval \
  --warmup_steps 2500 \
  --max_steps 100000 \
  --logging_steps 500 \
  --save_steps 2000 \
  --fp16 \
  --overwrite_output_dir 2>&1 | tee $outpath/run.log

# Copy checkpoints to Google Drive after training (or interruption)
echo "Syncing checkpoints to Google Drive..."
rsync -av --progress $outpath/ "$DrivePath/"
echo "Checkpoints saved to $DrivePath"
