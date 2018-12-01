import tensorflow as tf
import numpy as np

FLAGS = tf.flags.FLAGS
max_lt_len = 50

def get_embeddings(vocab):
    print("get_embedding")
    initializer = load_word_embeddings(vocab, FLAGS.embedding_dim)
    return tf.constant(initializer, name="word_embedding")

def load_embed_vectors(fname, dim):
    vectors = {}
    for line in open(fname, 'rt'):
        items = line.strip().split(' ')
        if len(items[0]) <= 0:
            continue
        vec = [float(items[i]) for i in range(1, dim+1)]
        vectors[items[0]] = vec
    return vectors

def load_word_embeddings(vocab, dim):
    vectors = load_embed_vectors(FLAGS.embeded_vector_file, dim)
    vocab_size = len(vocab)
    embeddings = np.zeros((vocab_size, dim), dtype='float32')
    for word, code in vocab.items():
        if word in vectors:
            embeddings[code] = vectors[word]
        #else:
        #    embeddings[code] = np.random.uniform(-0.25, 0.25, dim) 
    return embeddings 


def lstm_layer(inputs, input_seq_len, rnn_size, dropout_keep_prob, scope, scope_reuse=False):
    with tf.variable_scope(scope, reuse=scope_reuse) as vs:
        fw_cell = tf.contrib.rnn.LSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True, reuse=scope_reuse)
        fw_cell  = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout_keep_prob)
        bw_cell = tf.contrib.rnn.LSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True, reuse=scope_reuse)
        bw_cell  = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=dropout_keep_prob)
        rnn_outputs, rnn_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell,
                                                                inputs=inputs,
                                                                sequence_length=input_seq_len,
                                                                dtype=tf.float32)
        return rnn_outputs, rnn_states

def multi_lstm_layer(inputs, input_seq_len, rnn_size, dropout_keep_prob, num_layers, scope, scope_reuse=False):
    with tf.variable_scope(scope, reuse=scope_reuse) as vs:
        multi_outputs = []
        multi_states = []
        cur_inputs = inputs
        for i_layer in range(num_layers):
            rnn_outputs, rnn_states = lstm_layer(cur_inputs, input_seq_len, rnn_size, dropout_keep_prob, scope+str(i_layer), scope_reuse)
            rnn_outputs = tf.concat(axis=2, values=rnn_outputs)
            rnn_states = tf.concat(axis=1, values=[rnn_states[0].h, rnn_states[1].h])
            multi_outputs.append(rnn_outputs)
            multi_states.append(rnn_states)
            cur_inputs = rnn_outputs

        # multi_layer_aggregation
        ml_weights = tf.nn.softmax(tf.get_variable("ml_scores", [num_layers, ], initializer=tf.constant_initializer(0.0)))

        # output
        multi_outputs = tf.stack(multi_outputs, axis=-1)   # [batch_size, max_len, 2*rnn_size(400), num_layers]
        max_len = multi_outputs.get_shape()[1].value
        dim = multi_outputs.get_shape()[2].value
        flattened_multi_outputs = tf.reshape(multi_outputs, [-1, num_layers])                        # [batch_size * max_len * 2*rnn_size(400), num_layers]
        aggregated_ml_outputs = tf.matmul(flattened_multi_outputs, tf.expand_dims(ml_weights, 1))    # [batch_size * max_len * 2*rnn_size(400), 1]
        aggregated_ml_outputs = tf.reshape(aggregated_ml_outputs, [-1, max_len, dim])                # [batch_size , max_len , 2*rnn_size(400)]

        # state
        multi_states = tf.stack(multi_states, axis=-1)   # [batch_size, 2*rnn_size(400), num_layers]
        flattened_multi_states = tf.reshape(multi_states, [-1, num_layers])                      # [batch_size * 2*rnn_size(400), num_layers]
        aggregated_ml_states = tf.matmul(flattened_multi_states, tf.expand_dims(ml_weights, 1))  # [batch_size * 2*rnn_size(400), 1]
        aggregated_ml_states = tf.reshape(aggregated_ml_states, [-1, dim])                       # [batch_size , 2*rnn_size(400)]

        return aggregated_ml_outputs, aggregated_ml_states


# input: [batch_size, n_seq, emb]
def pooling(input_value, seq_len, scope, scope_reuse):
    with tf.variable_scope(scope, reuse=scope_reuse):
        # batch_size = input_value.get_shape()[0].value
        n_seq = input_value.get_shape()[1].value
        emb = input_value.get_shape()[-1].value

        W_0 = tf.get_variable("W_0", shape=[emb, emb], initializer=tf.orthogonal_initializer())
        b_0 = tf.get_variable("bias_0", shape=[emb, ], initializer=tf.zeros_initializer())
        W_1 = tf.get_variable("W_1", shape=[emb, emb], initializer=tf.orthogonal_initializer())
        b_1 = tf.get_variable("bias_1", shape=[emb, ], initializer=tf.zeros_initializer())

        # multi-dimention pooling
        value_flatten = tf.reshape(input_value, [-1, emb])
        tmp = tf.nn.elu(tf.matmul(value_flatten, W_0) + b_0)
        value_weight = tf.matmul(tmp, W_1) + b_1
        value_weight = tf.reshape(value_weight, [-1, n_seq, emb])
        value_weight = tf.nn.softmax(value_weight)
        weighted_multi_dim = value_weight * input_value   # [batch_size, n_seq, emb]

        # mask
        mask = tf.sequence_mask(seq_len, n_seq, dtype=tf.float32)  # [batch_size, n_seq]
        masked_multi_dim = weighted_multi_dim * tf.expand_dims(mask, -1)  # [batch_size, n_seq, emb]
        pooled_multi_dim = tf.reduce_sum(masked_multi_dim, 1)    # [batch_size, emb]

        return pooled_multi_dim

def question_answer_similarity_matrix(question, answer):
    q_len = question.get_shape()[1].value
    a_len = answer.get_shape()[1].value
    dim = question.get_shape()[2].value

    q_w = question

    #answer : batch_size * a_len * dim
    #[batch_size, dim, q_len]
    q2 = tf.transpose(q_w, perm=[0,2,1])

    #[batch_size, a_len, q_len]
    similarity = tf.matmul(answer, q2, name='similarity_matrix')

    return similarity


def self_attended(similarity_matrix, inputs):
    #similarity_matrix: [batch_size, len, len]
    #inputs: [batch_size, len, dim]

    attended_w = tf.nn.softmax(similarity_matrix, dim=-1)

    #[batch_size, len, dim]
    attended_out = tf.matmul(attended_w, inputs)
    return attended_out

def attended_answers(similarity_matrix, questions):
    #similarity_matrix: [batch_size, a_len, q_len]
    #questions: [batch_size, q_len, dim]
    
    # masked similarity_matrix
    # mask_q = tf.sequence_mask(question_len, question_max_len, dtype=tf.float32)  # [batch_size, q_len]
    # mask_q = tf.expand_dims(mask_q, 1)                                           # [batch_size, 1, q_len]
    # similarity_matrix = similarity_matrix * mask_q + -1e9 * (1-mask_q)           # [batch_size, a_len, q_len]

    #[batch_size, a_len, q_len]
    attention_weight_for_q = tf.nn.softmax(similarity_matrix, dim=-1)

    #[batch_size, a_len, dim]
    attended_answers = tf.matmul(attention_weight_for_q, questions)
    return attended_answers

def attended_questions(similarity_matrix, answers):
    #similarity_matrix: [batch_size, a_len, q_len]
    #answers: [batch_size, a_len, dim]

    # masked similarity_matrix
    # mask_a = tf.sequence_mask(answers_len, answers_max_len, dtype=tf.float32)    # [batch_size, a_len]
    # mask_a = tf.expand_dims(mask_a, 2)                                           # [batch_size, a_len, 1]
    # similarity_matrix = similarity_matrix * mask_a + -1e9 * (1-mask_a)           # [batch_size, a_len, q_len]

    #[batch_size, q_len, a_len]
    attention_weight_for_a = tf.nn.softmax(tf.transpose(similarity_matrix, perm=[0,2,1]), dim=-1)

    #[batch_size, q_len, dim]
    attended_questions = tf.matmul(attention_weight_for_a, answers)
    return attended_questions


class ESIM(object):
    def __init__(
      self, sequence_length, vocab_size, embedding_size, vocab, rnn_size):

        n_answers = 100
        num_layers = 3

        self.question = tf.placeholder(tf.int32, [None, sequence_length], name="question")
        self.answer = tf.placeholder(tf.int32, [None, n_answers, sequence_length], name="answer")
        self.lastTurn = tf.placeholder(tf.int32, [None, max_lt_len], name="lastTurn")

        self.question_len = tf.placeholder(tf.int32, [None], name="question_len")
        self.answer_len = tf.placeholder(tf.int32, [None, n_answers], name="answer_len")
        self.lastTurn_len = tf.placeholder(tf.int32, [None], name="lastTurn_len")

        self.target = tf.placeholder(tf.int64, [None], name="target")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.name_scope("embedding"):
            W = get_embeddings(vocab)
            question_embedded = tf.nn.embedding_lookup(W, self.question)  # [batch_size, q_len, word_dim]
            lastTurn_embedded = tf.nn.embedding_lookup(W, self.lastTurn)  # [batch_size, max_lt_len, word_dim]
            answer_embedded = tf.nn.embedding_lookup(W, self.answer)      # [batch_size, n_answers, a_len, word_dim]
            print("shape of question_embedded: {}".format(question_embedded.get_shape()))
            print("shape of lastTurn_embedded: {}".format(lastTurn_embedded.get_shape()))
            print("shape of answer_embedded: {}".format(answer_embedded.get_shape()))
        
            answer_embedded   = tf.reshape(answer_embedded, [-1, sequence_length, embedding_size])       # [batch_size*n_answers, sequence_length, embedding_size]
            question_len_tile = tf.tile(tf.expand_dims(self.question_len, 1), [1, n_answers]) # [batch_size, n_answers]
            question_len_flatten = tf.reshape(question_len_tile, [-1, ])                      # [batch_size*n_answers, ]
            answer_len_flatten = tf.reshape(self.answer_len, [-1, ])                          # [batch_size*n_answers, ]


        with tf.variable_scope("encoding_layer") as vs:
            rnn_scope_name = "multi_lstm_rnn"
            # [batch_size,           sequence_length, 2*rnn_size(400)], [batch_size,           2*rnn_size(400)]
            question_output, question_state = multi_lstm_layer(question_embedded, self.question_len,  rnn_size, self.dropout_keep_prob, num_layers, rnn_scope_name, scope_reuse=False)
            # [batch_size,           sequence_length, 2*rnn_size(400)], [batch_size,           2*rnn_size(400)]
            lastTurn_output, lastTurn_state = multi_lstm_layer(lastTurn_embedded, self.lastTurn_len,  rnn_size, self.dropout_keep_prob, num_layers, rnn_scope_name, scope_reuse=True)
            # [batch_size*n_answers, sequence_length, 2*rnn_size(400)], [batch_size*n_answers, 2*rnn_size(400)]
            answer_output, answer_state   = multi_lstm_layer(answer_embedded,   answer_len_flatten, rnn_size, self.dropout_keep_prob, num_layers, rnn_scope_name, scope_reuse=True)
            output_dim = question_output.get_shape()[2].value
            print('multi_lstm_layer : {}'.format(num_layers))


        with tf.variable_scope("lastTurn_with_answer") as vs:
            output_dim = lastTurn_output.get_shape()[2].value
            M = tf.get_variable("M", shape=[output_dim, output_dim], initializer=tf.orthogonal_initializer())
            b = tf.get_variable("bias", shape=[output_dim, ], initializer=tf.zeros_initializer())

            match_score = tf.matmul(lastTurn_state, M)  # [batch_size, dim]
            match_score = tf.matmul(tf.expand_dims(match_score, 1),
                                    tf.transpose(tf.reshape(answer_state, [-1, n_answers, output_dim]), [0, 2, 1]))   # [batch_size, 1, n_answers]
            match_score = tf.squeeze(match_score, [1])    # [batch_size, n_answers]
            print('polish with lastTurn')
        
        
        with tf.variable_scope("matching_layer") as vs:
            question_output = tf.tile(tf.expand_dims(question_output, 1), [1, n_answers, 1, 1])
            question_output = tf.reshape(question_output, [-1, sequence_length, output_dim])#[batch_size*n_answers, question_len, dim]
            similarity = question_answer_similarity_matrix(question_output, answer_output)  #[batch_size*n_answers, answer_len, question_len]
            attended_answer_output = attended_answers(similarity, question_output)          #[batch_size*n_answers, answer_len, dim]
            attended_question_output = attended_questions(similarity, answer_output)        #[batch_size*n_answers, question_len, dim]

            # [batch_size*n_answers, question_len, 4*dim]
            m_a = tf.concat(axis=2, values=[answer_output, attended_answer_output, tf.multiply(answer_output, attended_answer_output), answer_output-attended_answer_output])
            m_q = tf.concat(axis=2, values=[question_output, attended_question_output, tf.multiply(question_output, attended_question_output), question_output-attended_question_output])
            
            rnn_scope_layer2 = 'bidirectional_rnn_cross'
            rnn_size_layer_2 = rnn_size
            rnn_output_q_2, rnn_states_q_2 = lstm_layer(m_q, question_len_flatten, rnn_size_layer_2, self.dropout_keep_prob, rnn_scope_layer2, scope_reuse=False)
            rnn_output_a_2, rnn_states_a_2 = lstm_layer(m_a, answer_len_flatten, rnn_size_layer_2, self.dropout_keep_prob, rnn_scope_layer2, scope_reuse=True)

            question_output_2 = tf.concat(axis=2, values=rnn_output_q_2)     # [batch_size*n_answers, question_len, 2*rnn_size(400)]
            answer_output_2   = tf.concat(axis=2, values=rnn_output_a_2)     # [batch_size*n_answers, answer_len,   2*rnn_size(400)]

            # 1. simple max pooling
            # final_question_max = tf.reduce_max(question_output_2, axis=1)    # [batch_size*n_answers, 2*rnn_size(400)]
            # final_answer_max   = tf.reduce_max(answer_output_2, axis=1)      # [batch_size*n_answers, 2*rnn_size(400)]
            # 2. multi-dim pooling
            final_question_max = pooling(question_output_2, question_len_flatten, scope="pooling", scope_reuse=False)
            final_answer_max = pooling(answer_output_2, answer_len_flatten, scope="pooling", scope_reuse=True)
            print('multi_dimensional_pooling')

            layer_q_last_state = tf.concat(axis=1, values=[rnn_states_q_2[0].h, rnn_states_q_2[1].h])   # [batch_size*n_answers, 2*rnn_size(400)]
            layer_a_last_state = tf.concat(axis=1, values=[rnn_states_a_2[0].h, rnn_states_a_2[1].h])   # [batch_size*n_answers, 2*rnn_size(400)]
            print('last_state_pooling')


        with tf.name_scope("convolution-1"):
            joined_feature =  tf.concat(axis=1, values=[final_question_max, final_answer_max, layer_q_last_state, layer_a_last_state])  # [batch_size*n_answers, 8*rnn_size(1600)]
            print("shape of joined feature: {}".format(joined_feature.get_shape()))

            hidden_input_size = joined_feature.get_shape()[1].value
            hidden_output_size = 256
            full_out = tf.contrib.layers.fully_connected(joined_feature, hidden_output_size,
                                                            activation_fn=tf.nn.relu,
                                                            reuse=False,
                                                            trainable=True,
                                                            scope="projected_layer")   # [batch_size*n_answers, hidden_output_size(256)]

            last_weight_dim = full_out.get_shape()[1].value
            print("last_weight_dim: {}".format(last_weight_dim))
            bias = tf.Variable(tf.constant(0.1, shape=[1]), name="bias")
            s_w = tf.get_variable("s_w", shape=[last_weight_dim, 1], initializer=tf.contrib.layers.xavier_initializer())

            logits = tf.matmul(full_out, s_w) + bias     # [batch_size*n_answers, 1]
            logits = tf.reshape(logits, [-1, n_answers]) # [batch_size, n_answers]
            factor = tf.Variable(tf.constant(1.0, shape=[1]), name="factor")
            logits = logits + factor * match_score       # [batch_size, n_answers]
            
            print("shape of logits: {}".format(logits.get_shape()))

            self.probs = tf.nn.softmax(logits, name="prob")   # [batch_size, n_answers]
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.target)
            self.mean_loss = tf.reduce_mean(losses, name="mean_loss")

        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(self.probs, 1), self.target)    # [batch_size, ]
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy")
