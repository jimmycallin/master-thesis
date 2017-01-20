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

source activate cnn2
echo "Python environment is set up"
set -o errexit # exit on errors

echo "Copying files to $SCRATCH..."

LOCAL_BASE_DIR="/usit/abel/u1/jimmycallin/"

cp -r $LOCAL_BASE_DIR/resources $SCRATCH
cp -r $LOCAL_BASE_DIR/architectures/conll16st-hd-sdp $SCRATCH
cd $SCRATCH/conll16st-hd-sdp

echo "We are now in conll16st-hd-sdp"

NAME=$1

EMBEDDING_PATH="$SCRATCH/resources/$2"
DATA_BASE_PATH="$SCRATCH/resources/conll16st-en-zh-dev-train-test_LDC2016E50/conll16st-en-03-29-16"
MODEL_STORE_PATH="$LOCAL_BASE_DIR/models/$NAME"
mkdir -p $MODEL_STORE_PATH

echo "Starting LOGREG run"
echo "NAME: $NAME"
echo "LOCAL_BASE_DIR: $LOCAL_BASE_DIR"
echo "EMBEDDING_PATH: $EMBEDDING_PATH"
echo "DATA_BASE_PATH: $DATA_BASE_PATH"
echo "MODEL_STORE_PATH: $MODEL_STORE_PATH"

#run parser in train mode
echo '=========================================='
echo '==============TRAIN======================='
echo '=========================================='
python sup_parser_v2_hierarchy.py en $DATA_BASE_PATH-train ${MODEL_STORE_PATH} "output_dir_not_used" -run_name:$NAME -cmd:train -word2vec_model:$EMBEDDING_PATH -word2vec_load_bin:True -scale_features:True

