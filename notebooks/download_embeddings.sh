#!/bin/bash

set -e

MODEL_DIR="../pretrained_models"
SAVED_DIR="../saved_models"

mkdir -p "$MODEL_DIR"
mkdir -p "$SAVED_DIR"

echo "=== Downloading GoogleNews Word2Vec ==="

echo "=== Downloading GoogleNews Word2Vec ==="

if [ ! -f "$MODEL_DIR/GoogleNews-vectors-negative300.bin" ]; then
    wget -c \
        "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz" \
        -O "$MODEL_DIR/GoogleNews-vectors-negative300.bin.gz"

    gunzip "$MODEL_DIR/GoogleNews-vectors-negative300.bin.gz"
else
    echo "GoogleNews Word2Vec already exists."
fi

echo "=== Downloading fastText Wiki News ==="

if [ ! -f "$MODEL_DIR/wiki-news-300d-1M.vec" ]; then
    wget -c \
        https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip \
        -O "$MODEL_DIR/wiki-news-300d-1M.vec.zip"

    unzip -o "$MODEL_DIR/wiki-news-300d-1M.vec.zip" -d "$MODEL_DIR"

    rm "$MODEL_DIR/wiki-news-300d-1M.vec.zip"
else
    echo "fastText already exists."
fi


echo "=== Downloading GloVe 6B ==="

if [ ! -f "$MODEL_DIR/glove.6B.50d.txt" ]; then
    wget -c \
        https://nlp.stanford.edu/data/glove.6B.zip \
        -O "$MODEL_DIR/glove.6B.zip"

    unzip -o "$MODEL_DIR/glove.6B.zip" \
        "glove.6B.50d.txt" \
        -d "$MODEL_DIR"

    rm "$MODEL_DIR/glove.6B.zip"
else
    echo "GloVe 50d already exists."
fi


echo "=== Converting GloVe TXT -> VEC ==="

if [ ! -f "$MODEL_DIR/glove.6B.50d.vec" ]; then

    python3 - <<'PY'
from pathlib import Path

src = Path("../pretrained_models/glove.6B.50d.txt")
dst = Path("../pretrained_models/glove.6B.50d.vec")

with src.open("r", encoding="utf-8") as f:
    lines = f.readlines()

with dst.open("w", encoding="utf-8") as f:
    f.write(f"{len(lines)} 50\n")
    f.writelines(lines)

print(f"Created {dst}")
PY

else
    echo "GloVe VEC already exists."
fi


echo
echo "=== Result ==="
ls -lh "$MODEL_DIR"

echo
echo "Your from-scratch model should be:"
echo "$SAVED_DIR/model_emb_from_scratch.bin"