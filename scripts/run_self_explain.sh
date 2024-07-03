#!/bin/bash
export TOKENIZERS_PARALLELISM=false
python model/run.py \
    --num_gpus 1 \
    --batch_size 32 \
    --clip_grad 1.0 \
    --dataset_basedir data/XLNet-SUBJ \
    --concept_store data/XLNet-SUBJ/concept_store.pt \
    --model_name xlnet-base-cased \
    --gamma 0.1 \
    --lamda 0.5 \
    --topk 5 \
    --min_lr 1e-5 \
    --h_dim 256 \
    --n_heads 8 \
    --kqv_dim 64 \
    --num_classes 10 \
    --lr 0.001 \
    --weight_decay 0.0001 \
    --warmup_prop 0.1
                         
                         

# for RoBERTa
# python model/run.py --dataset_basedir data/RoBERTa-SUBJ \
#                          --lr 2e-5  --max_epochs 5 \
#                          --gpus 1 \
#                          --concept_store data/RoBERTa-SUBJ/concept_store.pt \
#                          --accelerator ddp \
#                          --model_name roberta-base \
#                          --topk 5 \
#                          --gamma 0.1 \
#                          --lamda 0.1
