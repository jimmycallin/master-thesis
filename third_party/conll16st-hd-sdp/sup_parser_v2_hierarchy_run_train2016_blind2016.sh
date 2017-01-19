#!/usr/bin/env bash

# set params
#input_dataset_train=data/conll16st-en-01-12-16-trial
#input_dataset_test=data/conll16st-en-01-12-16-trial

input_dataset_train=/Users/jimmy/dev/edu/master-thesis/resources/conll16st-en-zh-dev-train-test_LDC2016E50/conll16st-en-03-29-16-train/
input_dataset_test=/Users/jimmy/dev/edu/master-thesis/resources/conll16st-en-zh-dev-train-test_LDC2016E50/conll15st-en-03-29-16-blind-test/

run_type=svm_base

run_name=${run_type}_sup_v2_hier_tr16blind16
if [ -n "$1" ]
then
  run_name=$1
fi     # $String is null.


#output dir for parsing results - used for test operations
output_dir=/Users/jimmy/dev/edu/master-thesis/reproduce_output/${run_name}
mkdir -p ${output_dir}

#model dir where output models are saved after train
model_dir=/Users/jimmy/dev/edu/master-thesis/reproduce_models/${run_name}
#rm -rf -- ${model_dir}
mkdir -p ${model_dir}

scale_features=True

# resources
# word2vec_model=resources/external/w2v_embeddings/qatarliving_qc_size20_win10_mincnt5_rpl_skip1_phrFalse_2016_02_23.word2vec.bin
word2vec_model=/Users/jimmy/dev/edu/master-thesis/resources/word_embeddings/precompiled/cbow/GoogleNews-vectors-negative300.txt

# word2vec_load_bin=False
word2vec_load_bin=True # for google pretrained embeddings

log_file=${run_name}_$(date +%y-%m-%d-%H-%M).log
. sup_parser_v2_hierarchy_run_partial.sh   > ${log_file}
