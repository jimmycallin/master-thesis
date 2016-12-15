#!/bin/bash


# Project:
#SBATCH --account=nn9447k
#
# Wall clock limit:
#SBATCH --time=12:00:00
#
# Max memory usage:
#SBATCH --mem-per-cpu=20G
# Number of threads per task:
#SBATCH --cpus-per-task=2

## Set up job environment:
source /cluster/bin/jobsetup

cd /usit/abel/u1/jimmycallin/

module purge   # clear any inherited modules
source /usit/abel/u1/jimmycallin/.bashrc

module load intel

source activate rnn

cp /usit/abel/u1/jimmycallin/third_party/conll16st-v34-focused-rnns/patch_topology.py /usit/abel/u1/jimmycallin/miniconda2/envs/rnn/lib/python2.7/site-packages/keras/engine/topology.py
cp /usit/abel/u1/jimmycallin/third_party/conll16st-v34-focused-rnns/patch_training.py /usit/abel/u1/jimmycallin/miniconda2/envs/rnn/lib/python2.7/site-packages/keras/engine/training.py

echo "Python environment is set up"
set -o errexit # exit on errors

echo "Copying files to $SCRATCH..."

cp -r resources $SCRATCH
cp -r third_party/conll16st-v34-focused-rnns $SCRATCH
cd $SCRATCH/conll16st-v34-focused-rnns

echo "We are now in conll16st-v34-focused-rnns"

# For testing purposes
# export EMBEDDING_PATH="/usit/abel/u1/jimmycallin/resources/word_embeddings/precompiled/rsv/wiki_2008_d100_w1_ncntx10000.wembed"
# export DATA_BASE_PATH="/usit/abel/u1/jimmycallin/resources/conll16st-en-zh-dev-train-test_LDC2016E50/conll16st-en-03-29-16"
# export MODEL_STORE_PATH="/usit/abel/u1/jimmycallin/models/rsv-wikipedia"

NAME=$1
LOCAL_BASE_DIR="/usit/abel/u1/jimmycallin/"
EMBEDDING_PATH="$SCRATCH/resources/$2"
DATA_BASE_PATH="$SCRATCH/resources/conll16st-en-zh-dev-train-test_LDC2016E50/conll16st-en-03-29-16"
MODEL_STORE_PATH="$LOCAL_BASE_DIR/models/$NAME"
CONFIG=--config="{\"words2vec_bin\": null, \"words_dim\": 20.0, \"filter_fn_name\": \"conn_gt_0\", \"focus_dim\": 6.0, \"random_per_sample\": 24.0, \"final_dropout\": 0.01608656108471007, \"epochs\": 200, \"words2vec_txt\": \"$EMBEDDING_PATH\", \"focus_dropout_W\": 0.4850461135349744, \"rnn_dim\": 50.0, \"focus_dropout_U\": 0.18210894621865603, \"epochs_len\": -1, \"final_dim\": 40.0, \"epochs_patience\": 10, \"rnn_dropout_W\": 0.16649459724958682, \"words_dropout\": 0.3543889040084549, \"rnn_dropout_U\": 0.4899141021546136}"
mkdir -p $MODEL_STORE_PATH


echo "Starting RNN run"
echo "NAME: $NAME"
echo "LOCAL_BASE_DIR: $LOCAL_BASE_DIR"
echo "EMBEDDING_PATH: $EMBEDDING_PATH"
echo "DATA_BASE_PATH: $DATA_BASE_PATH"
echo "MODEL_STORE_PATH: $MODEL_STORE_PATH"
echo "CONFIG: $CONFIG"

./v34/train.py $MODEL_STORE_PATH $DATA_BASE_PATH-train $DATA_BASE_PATH-dev --clean "$CONFIG"
