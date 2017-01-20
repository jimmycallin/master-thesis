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

LOCAL_BASE_DIR="/usit/abel/u1/jimmycallin/"

NAME=$1
TEST_TYPE=$2

INPUT_PATH="$SCRATCH/resources/conll16st-en-zh-dev-train-test_LDC2016E50/$TEST_TYPE/"
MODEL_PATH="$SCRATCH/models/$NAME/"
OUTPUT_PATH="$LOCAL_BASE_DIR/outputs/$NAME-$TEST_TYPE/"
EMBEDDING_PATH="$SCRATCH/resources/$3"
EMBEDDING_DIR=${EMBEDDING_PATH%/*}


mkdir -p $OUTPUT_PATH
mkdir $SCRATCH/resources
mkdir -p $EMBEDDING_DIR
mkdir $SCRATCH/models


echo "Copying files to $SCRATCH..."
cp -r "$LOCAL_BASE_DIR/resources/conll16st-en-zh-dev-train-test_LDC2016E50" "$SCRATCH/resources/"
cp -r "$LOCAL_BASE_DIR/resources/$3" "$EMBEDDING_PATH"
cp -r "$LOCAL_BASE_DIR/architectures/conll16st-hd-sdp" "$SCRATCH"
cp -r "$LOCAL_BASE_DIR/models/$NAME" "$SCRATCH/models/"
cd $SCRATCH/conll16st-hd-sdp
echo "We are now in conll16st-hd-sdp"

#run parser in train mode
echo '=========================================='
echo '==============TRAIN======================='
echo '=========================================='
python sup_parser_v2_hierarchy.py en $INPUT_PATH ${MODEL_PATH} $OUTPUT_PATH -run_name:$NAME -cmd:test -word2vec_model:$EMBEDDING_PATH -word2vec_load_bin:True -scale_features:True
