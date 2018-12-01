# Steps
First, you should download the dataset from the official website. <br>

Second, create a vocabulary file and name it as "vocab.txt". <br>

Third, download the [Glove embeddings](http://www-nlp.stanford.edu/data/glove.840B.300d.zip) and train word embeddings with Word2Vec, then combine them as proposed in the paper and name it as "glove_42B_300d_vec_plus_word2vec_100.txt" by running the script. <br>

```
python combine_emb.py
```
