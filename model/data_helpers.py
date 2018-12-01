import numpy as np
import random
import ijson
from nltk.tokenize import WordPunctTokenizer

max_lt_len = 50

def loadVocab(fname):
    '''
    vocab = {"<PAD>": 0, ...}
    '''
    vocab={}
    with open(fname, 'rt') as f:
        for index,line in enumerate(f):
            line = line.decode('utf-8').strip()
            fields = line.split('\t')
            term_id = int(index)
            vocab[fields[0]] = term_id
    return vocab

def process_dialog(dialog):
    """
    Add EOU and EOT tags between utterances and create a single context string.
    """
    row = []
    row_ids = []
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
    row_ids.append(dialog['example-id'])

    # Create the next utterance options and the target label
    correct_answer = dialog['options-for-correct-answers'][0]
    target_id = correct_answer['candidate-id']
    target_index = None
    for i, utterance in enumerate(dialog['options-for-next']):
        if utterance['candidate-id'] == target_id:
            target_index = i
        row.append(utterance['utterance'].lower() + " __eou__ ")
        row_ids.append(utterance['candidate-id'])

    if target_index is None:
        print('Correct answer not found in options-for-next - example {}. Setting 0 as the correct index'.format(dialog['example-id']))
    else:
        row.append(target_index)

    return row, row_ids

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

def toVec(tokens, vocab, maxlen):
    '''
    length: length of the input sequence
    vec: map the token to the vocab_id, return a varied-length array [3, 6, 4, 3, ...]
    '''
    n = len(tokens)
    length = 0
    vec=[]
    for i in range(n):
        length += 1
        if tokens[i] in vocab:
            vec.append(vocab[tokens[i]])
        else:
            vec.append(vocab["<UNK>"])

    return length, np.array(vec)

# main
def loadDataset(fname, vocab, maxlen):
    dataset=[]
    with open(fname, 'rb') as f:
        json_data = ijson.items(f, 'item')
        for i, entry in enumerate(json_data):
            row, row_ids = process_dialog(entry)  # [ctx, utr0, utr1, ..., utr99, target]
            row = select_ahead_hinder_part(row, maxlen, hinder=True)  # truncate to maxlen

            question = row[0]
            answers = row[1:101]
            target = row[101]

            q_tokens = question.split(" ")
            q_len = len(q_tokens)
            q_len_, q_vec = toVec(q_tokens[:maxlen], vocab, maxlen)
            assert q_len == q_len_

            # last turn
            last_turn = question.split(' __eot__ ')[-1]
            lt_tokens = last_turn.split(' ')
            lt_len, lt_vec = toVec(lt_tokens[:max_lt_len], vocab, max_lt_len)

            as_tokens = []
            as_len = []
            as_vec = []
            for answer in answers:
                a_tokens = answer.split(" ")
                a_len = len(a_tokens)
                a_len_, a_vec = toVec(a_tokens[:maxlen], vocab, maxlen)
                as_tokens.append(a_tokens)
                as_len.append(a_len)
                as_vec.append(a_vec)
                assert a_len == a_len_

            q_id = row_ids[0]
            as_id = row_ids[1:]
            dataset.append((q_len, q_vec, q_tokens, lt_len, lt_vec, lt_tokens, as_len, as_vec, as_tokens, q_id, as_id, target))
            if (i+1) % 1000 == 0:
                print("Cached {} examples.".format(i+1))

    return dataset

def normalize_vec(vec, maxlen):
    '''
    pad the original vec to the same maxlen
    [3, 4, 7] maxlen=5 --> [3, 4, 7, 0, 0]
    '''
    if len(vec) == maxlen:
        return vec

    new_vec = np.zeros(maxlen, dtype='int32')
    for i in range(len(vec)):
        new_vec[i] = vec[i]
    return new_vec

def batch_iter(data, batch_size, num_epochs, maxlen, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size)# + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            random.shuffle(data)
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            x_question = []
            x_answer = []
            x_question_len = []
            x_answer_len = []
            x_lastTurn = []
            x_lastTurn_len = []

            targets = []
            q_id_list = []
            as_id_list = []

            for rowIdx in range(start_index, end_index): 
                q_len, q_vec, q_tokens, lt_len, lt_vec, lt_tokens, as_len, as_vec, as_tokens, q_id, as_id, target = data[rowIdx]
                new_q_vec = normalize_vec(q_vec, maxlen)    # pad the original vec to the same maxlen
                new_as_vec = np.zeros((100, maxlen))
                for i, a_vec in enumerate(as_vec):
                    new_as_vec[i, :] = normalize_vec(a_vec, maxlen)

                x_question.append(new_q_vec)
                x_question_len.append(q_len)
                x_answer.append(new_as_vec)
                x_answer_len.append(as_len)
                targets.append(target)

                new_lt_vec = normalize_vec(lt_vec, max_lt_len)
                x_lastTurn.append(new_lt_vec)
                x_lastTurn_len.append(lt_len)

                q_id_list.append(q_id)
                as_id_list.append(as_id)

            yield np.array(x_question), np.array(x_answer), np.array(x_question_len), np.array(x_answer_len), \
            np.array(x_lastTurn), np.array(x_lastTurn_len), q_id_list, as_id_list, np.array(targets)
