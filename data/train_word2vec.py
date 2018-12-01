#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import logging
import os
import sys
import multiprocessing

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

import ijson
from nltk.tokenize import WordPunctTokenizer

def process_dialog(dialog):
    """
    Add EOU and EOT tags between utterances and create a single context string.
    """
    row = []
    utterances = dialog['messages-so-far']

    # Create the context
    context = ""
    speaker = None
    for msg in utterances:
        if speaker is None:
            context += msg['utterance'] + " __eou__ "
            speaker = msg['speaker']
        elif speaker != msg['speaker']:
            context += "__eot__ " + msg['utterance'] + " __eou__ "
            speaker = msg['speaker']
        else:
            context += msg['utterance'] + " __eou__ "

    context += "__eot__"
    row.append(context.lower())

    # Create the next utterance options and the target label
    correct_answer = dialog['options-for-correct-answers'][0]
    target_id = correct_answer['candidate-id']
    target_index = None
    for i, utterance in enumerate(dialog['options-for-next']):
        if utterance['candidate-id'] == target_id:
            target_index = i
        row.append(utterance['utterance'].lower() + " __eou__ ")

    if target_index is None:
        print('Correct answer not found in options-for-next - example {}. Setting 0 as the correct index'.format(dialog['example-id']))
    else:
        row.append(target_index)

    return row

def select_ahead_hinder_part(row, maxlen, hinder=True):
    if hinder:
        context = row[0]
        candidates = row[1:101]
        target = row[101]
        mofidied_row = []

        context = WordPunctTokenizer().tokenize(context)[-maxlen:]
        context = " ".join(context)
        mofidied_row.append(context)

        for candidate in candidates:
            candidate = WordPunctTokenizer().tokenize(candidate)[-maxlen:]
            candidate = " ".join(candidate)
            mofidied_row.append(candidate)

        mofidied_row.append(target)

        return mofidied_row
    else:
        return row


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    input_file = "ubuntu_train_subtask_1.json"
    output_file = "word2vec_100_embedding.txt"
    
    sentences = []
    with open(input_file, 'rb') as f:
        json_data = ijson.items(f, 'item')
        for i, entry in enumerate(json_data):
            row = process_dialog(entry)[:-1]  # [ctx, utr0, utr1, ..., utr99]
            # row = select_ahead_hinder_part(row, maxlen=160, hinder=True)
            for uter in row:
                sentences.append(WordPunctTokenizer().tokenize(uter))
            if (i+1) % 1000 == 0:
                print("Cached {} examples.".format(i+1))

    model = Word2Vec(sentences, size=100, window=5, min_count=1, sg=1,
                     workers=multiprocessing.cpu_count())

    model.wv.save_word2vec_format(output_file, binary=False)
