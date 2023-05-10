# generic training script for training reranking models
PROJECT_HOME=$1
INPUT_DIR=$2
# output directory
MODEL_HOME=$3

if [[ -z $PROJECT_HOME || -z $INPUT_DIR || -z $MODEL_HOME ]]; then
  echo 'one or more variables are undefined'
  exit 1
fi

echo Data home: $INPUT_DIR
echo Output dir: $MODEL_HOME

mkdir -p $MODEL_HOME/src/
cp -r $PROJECT_HOME/* $MODEL_HOME/src

python $MODEL_HOME/src/src/train_ranker.py \
  --model_name_or_path nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large \
  --max_seq_length 512 \
  --train_file $INPUT_DIR/train_sbert.jsonl \
  --validation_file $INPUT_DIR/dev_sbert.jsonl \
  --output_dir $MODEL_HOME/ \
  --cache_dir $MODEL_HOME/.cache/ \
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
