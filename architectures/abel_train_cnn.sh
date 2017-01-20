#!/bin/bash


# Project:
#SBATCH --account=nn9447k
#
# Wall clock limit:
#SBATCH --time=12:00:00
#
# Max memory usage:
#SBATCH --mem-per-cpu=8G
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

cp -r resources $SCRATCH
cp -r architectures/CoNLL2016-CNN $SCRATCH
cd $SCRATCH/CoNLL2016-CNN

chkfile "slurm*"

echo "We are now in CoNLL2016-CNN"

NAME=$1
LOCAL_BASE_DIR="/usit/abel/u1/jimmycallin/"
EMBEDDING_PATH="$SCRATCH/resources/$2"
DATA_BASE_PATH="$SCRATCH/resources/conll16st-en-zh-dev-train-test_LDC2016E50/conll16st-en-03-29-16"
MODEL_STORE_PATH="$LOCAL_BASE_DIR/models/$NAME"
mkdir -p $MODEL_STORE_PATH

echo "Starting CNN run"
echo "NAME: $NAME"
echo "LOCAL_BASE_DIR: $LOCAL_BASE_DIR"
echo "EMBEDDING_PATH: $EMBEDDING_PATH"
echo "DATA_BASE_PATH: $DATA_BASE_PATH"
echo "MODEL_STORE_PATH: $MODEL_STORE_PATH"
python model_trainer/tf_cnn_implicit/trainer.py $NAME $DATA_BASE_PATH $EMBEDDING_PATH $MODEL_STORE_PATH "train"
