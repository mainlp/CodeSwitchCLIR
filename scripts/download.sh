# download qrels and reranking data
wget "https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/qrels.dev.small.tsv"
wget "https://msmarco.blob.core.windows.net/msmarcoranking/top1000.dev.tar.gz"
tar -xvzf top1000.dev.tar.gz
rm top1000.dev.tar.gz

# seen languages: en de ru ar nl it
# unseen languages: hi id it jp pt es vt fr
languages=( arabic chinese dutch english french german hindi indonesian italian japanese portuguese russian spanish vietnamese )
for LANG in "${languages[@]}"
do

# download train queries
PREFIX=$( pwd )/google_translations/queries/train
mkdir -p $PREFIX
wget "https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/train/${LANG}_queries.train.tsv"
mv "${LANG}_queries.train.tsv" $PREFIX

# download queries.dev.small queries
PREFIX=$( pwd )/google_translations/queries/dev
mkdir -p $PREFIX
wget "https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/queries/dev/${LANG}_queries.dev.small.tsv"
mv "${LANG}_queries.dev.small.tsv" $PREFIX

# download collections
PREFIX=$( pwd )/google_translations/collections
mkdir -p $PREFIX
wget "https://huggingface.co/datasets/unicamp-dl/mmarco/resolve/main/data/google/collections/${LANG}_collection.tsv"
mv "${LANG}_collection.tsv" $PREFIX

done

# download multilingual word embeddings
seen_languages=( ar en de it ru nl )
PREFIX=$( pwd )/muse_embeddings
mkdir -p $PREFIX
for LANG in "${seen_languages[@]}"
do
  wget "https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.${LANG}.vec"
  mv "wiki.multi.${LANG}.vec" $PREFIX
done
