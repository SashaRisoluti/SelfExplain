#export DATA_FOLDER='data/XLNet-SUBJ'
#export TOKENIZER_NAME='xlnet-base-cased'
#export MAX_LENGTH=5

# Creates jsonl files for train and dev

python preprocessing/store_parse_trees.py \
      --data_dir data/XLNet-SUBJ  \
      --tokenizer_name xlnet-base-cased

# Create concept store for SST-2 dataset
# Since SST-2 already provides parsed output, easier to do it this way, for other datasets, need to adapt

python preprocessing/build_concept_store.py \
       -i data/XLNet-SUBJ/train_with_parse.json \
       -o data/XLNet-SUBJ \
       -m xlnet-base-cased \
       -l 5
