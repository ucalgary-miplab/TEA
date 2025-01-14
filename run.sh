#!/bin/bash

if [ $# -eq 0 ]; then
  echo "No arguments provided. Please provide the bias type."
  exit 1
fi

bias=$1

# python 0_preprocess.py $bias

# if [ $? -eq 0 ]; then
#     echo "Preprocessing script ran successfully"
# else
#     echo "Preprocessing failed" >&2
#     exit 1
# fi

# python 1_encode.py $bias

# if [ $? -eq 0 ]; then
#     echo "Encoding script ran successfully"
# else
#     echo "Encoding failed" >&2
#     exit 1
# fi

# python 2_trainMACAW.py $bias

# if [ $? -eq 0 ]; then
#     echo "Training MACAW script ran successfully"
# else
#     echo "Training MACAW failed" >&2
#     exit 1
# fi

# python 3_generateCF.py $bias
# if [ $? -eq 0 ]; then
#     echo "Counterfactual generation script ran successfully"
# else
#     echo "Counterfactual generation failed" >&2
#     exit 1
# fi

python 4_trainSFCN.py $bias
if [ $? -eq 0 ]; then
    echo "Training SFCN ran successfully"
else
    echo "Training SFCN failed" >&2
    exit 1
fi