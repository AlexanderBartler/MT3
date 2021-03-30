#!/bin/bash

# Define result path and create folder
RESULT=./results/byol_$(date +"%F-%H-%M-%S-%3N")
mkdir $(echo $RESULT)
# get absolute path
RESULT=$(realpath $RESULT)

# Run training
python3 ./train_byol.py --path $RESULT

# Run test
python3 ./test_byol.py --path $RESULT
