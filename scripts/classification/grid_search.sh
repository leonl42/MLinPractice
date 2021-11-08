#!/bin/bash

mkdir -p data/classification

# specify hyperparameter values
max_depth=("10 20 30 40")
max_features=("2 3 4 5")


# different execution modes
if [ $1 = local ]
then
    echo "[local execution]"
    cmd="scripts/classification/classifier.sge"
elif [ $1 = grid ]
then
    echo "[grid execution]"
    cmd="qsub code/classification/classifier.sge"
else
    echo "[ERROR! Argument not supported!]"
    exit 1
fi

# do the grid search
for d in $max_depth; do
for f in $max_features; do
    echo $d
    echo $f
    $cmd 'data/classification/clf_'"$k"'.pickle' --rndmforest 500 $d $f -s 42 --accuracy --kappa -f1
done
done