#!/bin/bash


# Project:
#SBATCH --account=nn9447k
#
# Wall clock limit:
#SBATCH --time=04:00:00
#
# Max memory usage:
#SBATCH --mem-per-cpu=4G
# Number of threads per task:
#SBATCH --cpus-per-task=4

## Set up job environment:
source /cluster/bin/jobsetup

cd /usit/abel/u1/jimmycallin/

module purge   # clear any inherited modules
source /usit/abel/u1/jimmycallin/.bashrc

module load intel

source activate ffnn
echo "Python environment is set up"
set -o errexit # exit on errors

echo "Copying files to $SCRATCH..."


NAME=$1
LOCAL_BASE_DIR="/usit/abel/u1/jimmycallin/"

cd $LOCAL_BASE_DIR/architectures/nn_discourse_parser
echo "We are now in nn_discourse_parser"


EMBEDDING_PATH="$LOCAL_BASE_DIR/resources/$2"
DATA_BASE_PATH="$LOCAL_BASE_DIR/resources/conll16st-en-zh-dev-train-test_LDC2016E50/conll16st-en-03-29-16"
MODEL_STORE_PATH="$LOCAL_BASE_DIR/models/$NAME"
mkdir -p $MODEL_STORE_PATH

echo "Starting FFNN run"
echo "NAME: $NAME"
echo "LOCAL_BASE_DIR: $LOCAL_BASE_DIR"
echo "EMBEDDING_PATH: $EMBEDDING_PATH"
echo "DATA_BASE_PATH: $DATA_BASE_PATH"
echo "MODEL_STORE_PATH: $MODEL_STORE_PATH"
python ./train_models.py implicit_conll_ff_train $EMBEDDING_PATH $DATA_BASE_PATH $MODEL_STORE_PATH static_input_layer
