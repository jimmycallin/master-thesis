#!/bin/bash


# Project:
#SBATCH --account=nn9447k
#
# Wall clock limit:
#SBATCH --time=03:00:00
#
# Max memory usage:
#SBATCH --mem-per-cpu=14G

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

LOCAL_BASE_DIR="/usit/abel/u1/jimmycallin/"


cd $LOCAL_BASE_DIR/architectures/nn_discourse_parser

echo "We are now in nn_discourse_parser"

NAME=$1
TEST_TYPE=$2

INPUT_PATH="$LOCAL_BASE_DIR/resources/conll16st-en-zh-dev-train-test_LDC2016E50/$TEST_TYPE/"
MODEL_PATH="$LOCAL_BASE_DIR/models/$NAME/"
OUTPUT_PATH="$LOCAL_BASE_DIR/outputs/$NAME-$TEST_TYPE/"
EMBEDDING_PATH="$LOCAL_BASE_DIR/resources/$3"
mkdir -p $OUTPUT_PATH
python ./neural_discourse_parser.py $NAME $MODEL_PATH $INPUT_PATH $OUTPUT_PATH $EMBEDDING_PATH

