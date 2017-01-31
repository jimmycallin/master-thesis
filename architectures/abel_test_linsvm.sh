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

source activate svm_baseline
echo "Python environment is set up"
set -o errexit # exit on errors

LOCAL_BASE_DIR="/usit/abel/u1/jimmycallin/"

NAME=$1
TEST_TYPE=$2

INPUT_PATH="$LOCAL_BASE_DIR/resources/conll16st-en-zh-dev-train-test_LDC2016E50/$TEST_TYPE/"
MODEL_PATH="$LOCAL_BASE_DIR/models/$NAME/"
OUTPUT_PATH="$LOCAL_BASE_DIR/outputs/$NAME-$TEST_TYPE/"
EMBEDDING_PATH="$LOCAL_BASE_DIR/resources/$3"

mkdir -p $OUTPUT_PATH

cd $LOCAL_BASE_DIR/architectures/svm_baseline/
echo "We are now in svm_baseline"

#run parser in train mode
echo '=========================================='
echo '==============TRAIN======================='
echo '=========================================='
python $LOCAL_BASE_DIR/architectures/svm_baseline/main.py --test --test-path $INPUT_PATH --embedding-path $EMBEDDING_PATH --model-store-path $MODEL_PATH --test-output-path $OUTPUT_PATH --svm-kernel linear
