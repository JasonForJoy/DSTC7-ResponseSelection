# DSTC7-ResponseSelection
Implementation of the model for 7th Edition of the Dialog System Technology Challenges with Tensorflow

This repository contains an implementation with Tensorflow of the model presented in the paper ["Building Sequential Inference Models for End-to-End Response Selection"](http://www.aclweb.org/anthology/P17-1152) by Gu et al. in AAAI 2019 Workshop on Dialog System Technology Challenges.

# Dependencies
Python 2.7 <br>
Tensorflow 1.4.0

# Running the scripts
## Download and preprocess
```
cd data
bash fetch_and_preprocess.sh
```

## Train and test a new model
```
cd scripts
bash train.sh
```
The training process and results are in log.txt file.

## Test a trained model
```
bash test.sh
```
The test results are in log_test.txt file.
