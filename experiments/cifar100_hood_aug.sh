#!/bin/bash
GPU=$1

DATASET='cifar100'
MODEL='wrn'
seeds='0'
hoodW=1.0
hoodT=5
augN=4
DIRNAME=${DATASET}_${MODEL}_hood_t${hoodT}_w${hoodW}_augn${augN}

for seed in ${seeds}; do
  CUDA_VISIBLE_DEVICES=${GPU} python train.py \
    ${DATASET} \
    --model ${MODEL} \
    --save ./outputs/${DIRNAME}/seed_${seed} \
    --hsic-weight ${hoodW} \
    --hsic-tau ${hoodT} \
    --disable_random 1 \
    --aug 1 \
    --aug-n ${augN} \
    --seed ${seed}

  # test 10 times for sfm metric
  mkdir -p outputs/${DIRNAME}/seed_${seed}/test
  echo outputs/${DIRNAME}/seed_${seed}/test
  CUDA_VISIBLE_DEVICES=${GPU} python test.py \
    --method_name ${DIRNAME} \
    --save ./outputs/${DIRNAME}/seed_${seed} \
    --load ./outputs/${DIRNAME}/seed_${seed}/checkpoints/ckp-99.pth \
    --num_to_avg 10

  # test 10 times for cor metric
  mkdir -p outputs/${DIRNAME}/seed_${seed}/test
  echo outputs/${DIRNAME}/seed_${seed}/test
  CUDA_VISIBLE_DEVICES=${GPU} python test_cor.py \
    --method_name ${DIRNAME} \
    --save ./outputs/${DIRNAME}/seed_${seed} \
    --load ./outputs/${DIRNAME}/seed_${seed}/checkpoints/ckp-99.pth \
    --num_to_avg 10
done

# collect results for sfm
mkdir -p outputs/csv_results
mkdir -p ./outputs/${DIRNAME}/test/sfm
python utils/merge_csv.py \
  --save ./outputs/${DIRNAME} \
  --csv_file sfm \
  --test_dir test/sfm \
  --seeds ${seeds} >./outputs/${DIRNAME}/test/sfm/results.log
cp ./outputs/${DIRNAME}/test/sfm/${DIRNAME}_sfm.csv ./outputs/csv_results

# collect results for cor
mkdir -p outputs/csv_results
mkdir -p outputs/${DIRNAME}/test/cor
python utils/merge_csv.py \
  --save ./outputs/${DIRNAME} \
  --csv_file cor \
  --test_dir test/cor \
  --seeds ${seeds} >./outputs/${DIRNAME}/test/cor/results.log
cp ./outputs/${DIRNAME}/test/cor/${DIRNAME}_cor.csv ./outputs/csv_results
