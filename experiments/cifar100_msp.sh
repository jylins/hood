#!/bin/bash
GPU=$1

DATASET='cifar100'
MODEL='wrn'
seeds='0'
DIRNAME=${DATASET}_${MODEL}_msp

for seed in ${seeds};
do
  CUDA_VISIBLE_DEVICES=${GPU} python train_base.py \
    ${DATASET} \
    --model ${MODEL} \
    --save ./outputs/${DIRNAME}/seed_${seed} \
    --seed ${seed}

  # test 10 times for sfm
  mkdir -p outputs/${DIRNAME}/seed_${seed}/test
  echo outputs/${DIRNAME}/seed_${seed}/test
  CUDA_VISIBLE_DEVICES=${GPU} python test.py \
    --method_name ${DIRNAME} \
    --save ./outputs/${DIRNAME}/seed_${seed} \
    --load ./outputs/${DIRNAME}/seed_${seed}/checkpoints/ckp-99.pth \
    --num_to_avg 10
done


# collect results
mkdir -p outputs/csv_results
mkdir -p ./outputs/${DIRNAME}/test/sfm
python utils/merge_csv.py \
  --save ./outputs/${DIRNAME} \
  --csv_file sfm \
  --test_dir test/sfm \
  --seeds ${seeds} > ./outputs/${DIRNAME}/test/sfm/results.log
cp ./outputs/${DIRNAME}/test/sfm/${DIRNAME}_sfm.csv ./outputs/csv_results


