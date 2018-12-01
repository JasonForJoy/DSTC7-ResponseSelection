# Dialog System Technology Challenges 7 (DSTC7) Track 1
Implementation of the model for Track 1 of 7th Edition of the Dialog System Technology Challenges with Tensorflow

This repository contains an implementation with Tensorflow of the model presented in the paper ["Building Sequential Inference Models for End-to-End Response Selection"](http://www.aclweb.org/anthology/P17-1152) by Gu et al. in AAAI 2019 Workshop on DSTC7. <br>
In this challenge, the model has achieved the results of rank 2nd on Ubuntu dataset and rank 3rd on Flex dataset.

# Dependencies
Python 2.7 <br>
Tensorflow 1.4.0

# Data Preparation
Step 1. Refer to the official website [DSTC7](https://github.com/IBM/dstc7-noesis) to download the dataset. <br>

Step 2. Create a vocabulary file composed of words in the datast and name it as "vocab.txt". <br>

Step 3. Download the [Glove embeddings](http://www-nlp.stanford.edu/data/glove.840B.300d.zip) and train word embeddings with Word2Vec. Then combine these two embedding files as proposed in the paper by running the following commands. <br>

```
cd data
python train_word2vec.py
python combine_emb.py
```

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
