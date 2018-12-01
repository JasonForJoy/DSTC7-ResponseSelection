# Dialog System Technology Challenges 7 (DSTC7) Track 1
Implementation of the model for Track 1 of 7th Edition of the Dialog System Technology Challenges with Tensorflow

This repository contains an implementation with Tensorflow of the model presented in the paper ["Building Sequential Inference Models for End-to-End Response Selection"](http://www.aclweb.org/anthology/P17-1152) by Gu et al. in AAAI 2019 Workshop on DSTC7. <br>
In this challenge, the model has achieved the results of rank 2nd on Ubuntu dataset and rank 3rd on Flex dataset.

# Dependencies
Python 2.7 <br>
Tensorflow 1.4.0

# Download
You can refer to the official website [DSTC7](https://github.com/IBM/dstc7-noesis) to download the dataset.

# Running the scripts
## Train a new model
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
