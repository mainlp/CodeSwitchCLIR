MMARCO_DIR=./google_translations/collections

# remove trailling slashes (credit: https://stackoverflow.com/a/32845647)
MMARCO_DIR=$(echo "$MMARCO_DIR" | sed 's:/*$::')

PYGAGGLE_HOME=""
if [[ -z $PYGAGGLE_HOME ]]; then
  echo 'install pygaggle and set PYGAGGLE_HOME variable in this script manually (https://github.com/castorini/pygaggle)'
  exit 1
fi

languages=( arabic dutch french hindi italian portuguese spanish chinese english german indonesian japanese russian vietnamese )
for LANG in "${languages[@]}"
do

python $PYGAGGLE_HOME/tools/scripts/msmarco/convert_collection_to_jsonl.py \
  --collection-path $MMARCO_DIR/$LANG\_collection.tsv \
  --output-folder $MMARCO_DIR\_jsonl/$LANG

done
