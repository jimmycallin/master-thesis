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

cp -r resources $SCRATCH
cp -r third_party $SCRATCH
cd $SCRATCH/third_party/nn_discourse_parser

echo "We are now in nn_discourse_parser"

NAME=$1
LOCAL_BASE_DIR="/usit/abel/u1/jimmycallin/"
EMBEDDING_PATH="$SCRATCH/resources/$2"
DATA_BASE_PATH="$SCRATCH/resources/conll16st-en-zh-dev-train-test_LDC2016E50/conll16st-en-03-29-16"
MODEL_STORE_PATH="$LOCAL_BASE_DIR/models/$NAME"
mkdir $MODEL_STORE_PATH
mkdir $LOCAL_BASE_DIR/run_$SLURM_JOBID-$SLURM_JOB_NAME

echo "Starting FFNN run"
echo "NAME: $NAME"
echo "LOCAL_BASE_DIR: $LOCAL_BASE_DIR"
echo "EMBEDDING_PATH: $EMBEDDING_PATH"
echo "DATA_BASE_PATH: $DATA_BASE_PATH"
echo "MODEL_STORE_PATH: $MODEL_STORE_PATH"
python ./train_models.py implicit_conll_ff_train $EMBEDDING_PATH $DATA_BASE_PATH $MODEL_STORE_PATH
