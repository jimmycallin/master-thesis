#!/bin/bash


# Project:
#SBATCH --account=nn9447k
#
# Wall clock limit:
#SBATCH --time=02:00:00
#
# Max memory usage:
#SBATCH --mem-per-cpu=10G
# Number of threads per task:
#SBATCH --cpus-per-task=2

## Set up job environment:
source /cluster/bin/jobsetup
set -o errexit # exit on errors

cd /usit/abel/u1/jimmycallin/

module purge   # clear any inherited modules
source /usit/abel/u1/jimmycallin/.bashrc

module load intel/2017.0

source activate rnn

cp /usit/abel/u1/jimmycallin/architectures/conll16st-v34-focused-rnns/patch_topology.py /usit/abel/u1/jimmycallin/miniconda2/envs/rnn/lib/python2.7/site-packages/keras/engine/topology.py
cp /usit/abel/u1/jimmycallin/architectures/conll16st-v34-focused-rnns/patch_training.py /usit/abel/u1/jimmycallin/miniconda2/envs/rnn/lib/python2.7/site-packages/keras/engine/training.py

echo "Python environment is set up"


echo "Copying files to $SCRATCH..."
LOCAL_BASE_DIR="/usit/abel/u1/jimmycallin/"

NAME="$1"
TEST_TYPE="$2"
INPUT_PATH="$SCRATCH/resources/conll16st-en-zh-dev-train-test_LDC2016E50/$TEST_TYPE/"
MODEL_PATH="$SCRATCH/models/$NAME/"
EMBEDDING_PATH="$SCRATCH/resources/$3"
EMBEDDING_DIR=${EMBEDDING_PATH%/*}
OUTPUT_PATH="$LOCAL_BASE_DIR/outputs/$NAME-$TEST_TYPE/"
TRAIN_FILE_PATH="$SCRATCH/resources/conll16st-en-zh-dev-train-test_LDC2016E50/conll16st-en-03-29-16-train/"

echo "We are now in conll16st-v34-focused-rnns"

# For testing purposes
# export EMBEDDING_PATH="/usit/abel/u1/jimmycallin/resources/word_embeddings/precompiled/rsv/wiki_2008_d100_w1_ncntx10000.wembed"
# export DATA_BASE_PATH="/usit/abel/u1/jimmycallin/resources/conll16st-en-zh-dev-train-test_LDC2016E50/conll16st-en-03-29-16"
# export MODEL_STORE_PATH="/usit/abel/u1/jimmycallin/models/rsv-wikipedia"

CONFIG=--config="{\"words2vec_bin\": null, \"words_dim\": 20.0, \"filter_fn_name\": \"conn_gt_0\", \"focus_dim\": 6.0, \"random_per_sample\": 24.0, \"final_dropout\": 0.01608656108471007, \"epochs\": 200, \"words2vec_txt\": \"$EMBEDDING_PATH\", \"focus_dropout_W\": 0.4850461135349744, \"rnn_dim\": 50.0, \"focus_dropout_U\": 0.18210894621865603, \"epochs_len\": -1, \"final_dim\": 40.0, \"epochs_patience\": 10, \"rnn_dropout_W\": 0.16649459724958682, \"words_dropout\": 0.3543889040084549, \"rnn_dropout_U\": 0.4899141021546136}"

echo "CONFIG:"
echo "INPUT_PATH: $INPUT_PATH"
echo "EMBEDDING_PATH: $EMBEDDING_PATH"
echo "MODEL_PATH: $MODEL_PATH"
echo "OUTPUT_PATH: $OUTPUT_PATH"
echo "CONFIG: $CONFIG"
echo "========================="

mkdir -p $OUTPUT_PATH
mkdir $SCRATCH/resources
mkdir -p $EMBEDDING_DIR
mkdir $SCRATCH/models

cp -r "$LOCAL_BASE_DIR/resources/conll16st-en-zh-dev-train-test_LDC2016E50" "$SCRATCH/resources/"
cp -r "$LOCAL_BASE_DIR/resources/$3" "$EMBEDDING_PATH"
cp -r "$LOCAL_BASE_DIR/architectures/conll16st-v34-focused-rnns" "$SCRATCH"
cp -r "$LOCAL_BASE_DIR/models/$NAME" "$SCRATCH/models/"

cd "$SCRATCH/conll16st-v34-focused-rnns/"

./v34/classifier.py "en" "$MODEL_PATH" "$INPUT_PATH" "$OUTPUT_PATH" "$CONFIG"
