# Steps
First, you should download the dataset from the official website. <br>

Second, create a vocabulary file and name it as "vocab.txt". <br>

Third, download the [Glove embeddings](http://www-nlp.stanford.edu/data/glove.840B.300d.zip) and train word embeddings with Word2Vec. Then combine these two embedding files as proposed in the paper by running the command. <br>

```
python combine_emb.py
```
