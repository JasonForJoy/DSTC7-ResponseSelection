import numpy as np
import data_helpers


def load_embed_vectors(fname, dim):
    # vectors = { 'the': [0.2911, 0.3288, 0.2002,...], ... }
    vectors = {}
    for index, line in enumerate(open(fname, 'rt')):
        items = line.strip().split(' ')
        if len(items[0]) <= 0:
            continue
        vec = [float(items[i]) for i in range(1, dim+1)]
        vectors[items[0]] = vec

        if (index+1)%1000 == 0:
            print("Load {} words.".format(index+1))

    return vectors

def combine_emb(vocab, emb_vector1, dim1, emb_vector2, dim2):
    vocab_size = len(vocab)
    embeddings = np.zeros((vocab_size, dim1+dim2), dtype='float32')
    in_both = 0
    in_emb1 = 0
    in_emb2 = 0
    for word, code in vocab.items():

        if word in emb_vector1 and word in emb_vector2:
            embeddings[code, :dim1] = emb_vector1[word]
            embeddings[code, dim1:] = emb_vector2[word]
            in_both += 1

        elif word in emb_vector1 and word not in emb_vector2:
            embeddings[code, :dim1] = emb_vector1[word]
            in_emb1 += 1

        elif word not in emb_vector1 and word  in emb_vector2:
            embeddings[code, dim1:] = emb_vector2[word]
            in_emb2 += 1

    print("vocab_size: {}".format(vocab_size))
    print("in_both: {}".format(in_both))
    print("in_emb1: {}".format(in_emb1))
    print("in_emb2: {}".format(in_emb2))

    return embeddings
 
def save_emb(vocab, embeddings, outputfile):
    with open(outputfile, 'w') as f:
        for word, code in vocab.items():
            embed_vec = embeddings[code].tolist()
            embed_vec = [str(round(ele, 4)) for ele in embed_vec]
            line = embed_vec.insert(0, word)
            f.write(" ".join(embed_vec))
            f.write("\n")


if __name__ == '__main__':

    glove_embedding_file    = "glove.42B.300d.txt"
    word2vec_embedding_file = "word2vec_100_embedding.txt"
    outputfile = "glove_42B_300d_vec_plus_word2vec_100.txt"

    print("Loading data...")

    # vocab = {"<PAD>": 0, ...}
    vocab, idf = data_helpers.loadVocab("vocab.txt")
    print("vocabulary size: {}".format(len(vocab)))

    word2vec_embedding = load_embed_vectors(word2vec_embedding_file, dim=100)
    print("word2vec_embedding size: {}".format(len(word2vec_embedding)))
    # print(word2vec_embedding["it"])
    glove_embedding = load_embed_vectors(glove_embedding_file, dim=300)
    print("glove_embedding size: {}".format(len(glove_embedding)))

    embeddings = combine_emb(vocab, glove_embedding, 300, word2vec_embedding, 100)

    save_emb(vocab, embeddings, outputfile)