# training script for training wiki-cs models
QLANG=$1
DLANG=$2
SEED=$3

if [[ -z $QLANG || -z $DLANG || -z $SEED ]]; then
  echo 'one or more variables are undefined'
  exit 1
fi

PROJECT_HOME=/path/to/projects/CodeSwitchCLIR
DATA_HOME=/path/to/data_home/WIKI-CS/translation_prob=1/$QLANG$DLANG
OUT=/path/to/model_home/WIKI-CS/seed\_$SEED/1/$QLANG$DLANG

echo Data home: $DATA_HOME
echo Output dir: $OUT

mkdir -p $OUT/src/
cp -r $PROJECT_HOME/* $OUT/src

python $OUT/src/src/train_ranker.py \
  --model_name_or_path nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large \
  --max_seq_length 512 \
  --train_file $DATA_HOME/train_sbert.jsonl \
  --validation_file $DATA_HOME/dev_sbert.jsonl \
  --output_dir $OUT/ \
  --cache_dir $OUT/.cache/ \
  --do_train \
  --save_steps 20000 \
  --eval_steps 20000 \
  --max_steps 200000 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --overwrite_output_dir \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --fp16 \
  --fp16_backend amp \
  --seed $SEED \
  --log_level info \
  --warmup_steps 5000 \
  --model_str monobert \
  --evaluation_strategy steps \
  --load_best_model_at_end
