##########
# Usage: bash train_generator.sh <generator-target> <dataset> <base-folder>
##########

#!/usr/bin/env bash
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH -t 0
tgt="$1"
dataset="$2"
base_folder="$3"

# Switch to 0 for no bpe
BPE=1
if [ "$BPE" -eq  1 ]; then
    MODEL_PTH=models/$dataset/"bpe"
    echo "Using BPE"
else
    MODEL_PTH=models/$dataset/"nobpe"
    echo "Not using BPE"
fi

mkdir -p $MODEL_PTH

if [ "$BPE" -eq 1 ]; then
    python tag-and-generate-train/src/training.py \
        --cuda \
        --src en \
        --tgt "$tgt" \
        --model-file "$MODEL_PTH/en-${tgt}-generator.pt" \
        --n-layers 4 \
        --n-heads 4 \
        --embed-dim 512 \
        --hidden-dim 512 \
        --dropout 0.2 \
        --bpe \
        --word-dropout 0.05 \
        --lr 1e-3 \
        --n-epochs 15 \
        --tokens-per-batch 8000 \
        --clip-grad 1.1 \
        --base-folder "$base_folder"
else
    python tag-and-generate-train/src/training.py \
        --cuda \
        --src en \
        --tgt "$tgt" \
        --model-file "$MODEL_PTH/en-${tgt}-generator.pt" \
        --n-layers 4 \
        --n-heads 4 \
        --embed-dim 512 \
        --hidden-dim 512 \
        --dropout 0.2 \
        --word-dropout 0.05 \
        --lr 1e-3 \
        --n-epochs 15 \
        --tokens-per-batch 8000 \
        --clip-grad 1.1 \
        --base-folder "$base_folder"
fi
