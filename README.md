# Out-of-Distribution Detection with Hilbert-Schmidt Independence Optimization
This repository is the official [PyTorch](http://pytorch.org/) implementation of **HOOD** (**H**SIC assisted **OOD** detection).

## 0 Requirements

- Python 3.6
- [PyTorch](http://pytorch.org) install = 1.6.0
- torchvision install = 0.7.0
- CUDA 10.1
- Other dependencies: numpy, sklearn, six, pickle, lmdb

## 1 Training
We release a demo for the proposed HOOD method. The demo includes several OOD detection methods and baselines: MSP, OE, HOOD, HOOD+aug. All of the models are built based on WideResNet-40-2 architecture, trained for 100 epochs.

### 1.1 MSP

To train [MSP](./experiments/cifar100_msp.sh) for 100 epochs, run:

```shell
DATASET='cifar100'
MODEL='wrn'
seeds='0'
DIRNAME=${DATASET}_${MODEL}_msp

python train_base.py \
    ${DATASET} \
    --model ${MODEL} \
    --save ./outputs/${DIRNAME}/seed_${seed} \
    --seed ${seed}
```

### 1.2 OE

To train [OE](./experiments/cifar100_oe.sh) for 100 epochs, run:

```shell
DATASET='cifar100'
MODEL='wrn'
seeds='0'
DIRNAME=${DATASET}_${MODEL}_oe

python train.py \
    ${DATASET} \
    --model ${MODEL} \
    --save ./outputs/${DIRNAME}/seed_${seed} \
    --oe-weight 0.5 \
    --disable_random 1 \
    --seed ${seed}
```

### 1.3 HOOD

To train [HOOD](./experiments/cifar100_hood.sh) for 100 epochs, run:

```shell
DATASET='cifar100'
MODEL='wrn'
seeds='0'
hoodW=1.0
hoodT=5
augN=0
DIRNAME=${DATASET}_${MODEL}_hood_t${hoodT}_w${hoodW}_augn${augN}

python train.py \
    ${DATASET} \
    --model ${MODEL} \
    --save ./outputs/${DIRNAME}/seed_${seed} \
    --hsic-weight ${hoodW} \
    --hsic-tau ${hoodT} \
    --disable_random 1 \
    --aug 0 \
    --aug-n ${augN} \
    --seed ${seed}
```

### 1.4 HOOD+aug

To train [HOOD+aug](./experiments/cifar100_hood_aug.sh) for 100 epochs, run:

```shell
DATASET='cifar100'
MODEL='wrn'
seeds='0'
hoodW=1.0
hoodT=5
augN=4
DIRNAME=${DATASET}_${MODEL}_hood_t${hoodT}_w${hoodW}_augn${augN}

python train.py \
    ${DATASET} \
    --model ${MODEL} \
    --save ./outputs/${DIRNAME}/seed_${seed} \
    --hsic-weight ${hoodW} \
    --hsic-tau ${hoodT} \
    --disable_random 1 \
    --aug 1 \
    --aug-n ${augN} \
    --seed ${seed}
```

## 2 Evaluation

We present a demo for two evaluation metrics, including Softmax (**SFM**) metric and Correlation (**COR**) metric.

### 2.1 Softmax Metric

```shell
DIRNAME=dirname_demo
seeds=seed_demo

python test.py \
    --method_name ${DIRNAME} \
    --save ./outputs/${DIRNAME}/seed_${seed} \
    --load ./outputs/${DIRNAME}/seed_${seed}/checkpoints/ckp-99.pth \
    --num_to_avg 10
```

### 2.2 Correlation Metric

```shell
DIRNAME=dirname_demo
seeds=seed_demo

python test_cor.py \
    --method_name ${DIRNAME} \
    --save ./outputs/${DIRNAME}/seed_${seed} \
    --load ./outputs/${DIRNAME}/seed_${seed}/checkpoints/ckp-99.pth \
    --num_to_avg 10
```
