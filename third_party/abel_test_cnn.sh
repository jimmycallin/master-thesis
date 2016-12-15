#!/bin/bash


# Project:
#SBATCH --account=nn9447k
#
# Wall clock limit:
#SBATCH --time=02:00:00
#
# Max memory usage:
#SBATCH --mem-per-cpu=4G
# Number of threads per task:
#SBATCH --cpus-per-task=4

## Set up job environment:
source /cluster/bin/jobsetup

cd /usit/abel/u1/jimmycallin/

module purge   # clear any inherited modules
module load intel
source /usit/abel/u1/jimmycallin/.bashrc

source activate cnn
echo "Python environment is set up"
set -o errexit # exit on errors

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

echo "NAME: $NAME"
echo "TEST_TYPE: $TEST_TYPE"
echo "INPUT_PATH: $INPUT_PATH"
echo "MODEL_PATH: $MODEL_PATH"
echo "EMBEDDING_PATH: $EMBEDDING_PATH"
echo "EMBEDDING_DIR: $EMBEDDING_DIR"
echo "OUTPUT_PATH: $OUTPUT_PATH"
echo "TRAIN_FILE_PATH: $TRAIN_FILE_PATH"

mkdir -p $OUTPUT_PATH
mkdir $SCRATCH/resources
mkdir -p $EMBEDDING_DIR
mkdir $SCRATCH/third_party
mkdir $SCRATCH/models

cp -r "$LOCAL_BASE_DIR/resources/conll16st-en-zh-dev-train-test_LDC2016E50" "$SCRATCH/resources/"
cp -r "$LOCAL_BASE_DIR/resources/$3" "$EMBEDDING_PATH"
cp -r "$LOCAL_BASE_DIR/third_party/CoNLL2016-CNN" "$SCRATCH"
cp -r "$LOCAL_BASE_DIR/models/$NAME" "$SCRATCH/models/"

cd $SCRATCH/CoNLL2016-CNN

echo "We are now in CoNLL2016-CNN and begin testing..."

python ./model_trainer/tf_cnn_implicit/trainer.py $NAME $INPUT_PATH $EMBEDDING_PATH $MODEL_PATH "test" $OUTPUT_PATH $TRAIN_FILE_PATH
