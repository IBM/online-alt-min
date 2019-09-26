#!/bin/bash

if [ ! -d output ] ; then
    echo 'Creating output directory'
    .kdir output
fi

# Adam baseline:
python run_adam_baseline.py --dataset mnist --model feedforward --n-hiddens 100 --log-first-epoch --seed 1
python run_adam_baseline.py --dataset mnist --model feedforward --n-hiddens 500 --log-first-epoch --seed 1

python run_adam_baseline.py --dataset cifar10 --model feedforward --n-hiddens 100 --log-first-epoch --seed 1
python run_adam_baseline.py --dataset cifar10 --model feedforward --n-hiddens 500 --log-first-epoch --seed 1

python run_adam_baseline.py --dataset mnist --model lenet --log-first-epoch --seed 1
python run_adam_baseline.py --dataset higgs --model feedforward --n-hiddens 300 --epochs 1 --save-interval 200 --seed 1


# Alt-Min SGD (Adam):
python run_altmin_sgd.py --dataset mnist --model feedforward --n-hiddens 100 --log-first-epoch --seed 1
python run_altmin_sgd.py --dataset mnist --model feedforward --n-hiddens 500 --log-first-epoch --seed 1

python run_altmin_sgd.py --dataset cifar10 --model feedforward --n-hiddens 100 --log-first-epoch --seed 1
python run_altmin_sgd.py --dataset cifar10 --model feedforward --n-hiddens 500 --log-first-epoch --seed 1

python run_altmin_sgd.py --dataset mnist --model lenet --log-first-epoch --seed 1
python run_altmin_sgd.py --dataset higgs --model feedforward --n-hiddens 300 --epochs 1 --save-interval 200 --seed 1


# Alt-Min SGD on non-differentiable architecture:
python run_altmin_sgd.py --dataset mnist --model binary --n-hiddens 500 --n-iter-codes 10 --seed 1
