import tensorflow as tf
import numpy as np
import os
import time
import datetime
from model import data_helpers
from model.model_ESIM import ESIM
from model import recall_metrics

# Files
tf.flags.DEFINE_string("train_file", "", "train file containing (question, positive and negative answer ids)")
tf.flags.DEFINE_string("valid_file", "", "validation file containg (question, positive and negative response ids")
tf.flags.DEFINE_string("vocab_file", "", "vocabulary file (map word to integer)")
tf.flags.DEFINE_string("embeded_vector_file", "", "pre-trained embedded word vector")

# Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 400, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("max_sequence_length", 160, "max sequence length")
tf.flags.DEFINE_integer("rnn_size", 200, "number of RNN units")

# Training parameters
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_integer("batch_size", 2, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_epochs", 1000000, "Number of training epochs (default: 200)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Load data
print("Loading data...")

vocab = data_helpers.loadVocab(FLAGS.vocab_file)
print("vocabulary size: {}".format(len(vocab)))

SEQ_LEN = FLAGS.max_sequence_length
train_dataset = data_helpers.loadDataset(FLAGS.train_file, vocab, SEQ_LEN)
print('train_pairs: {}'.format(len(train_dataset)))
valid_dataset = data_helpers.loadDataset(FLAGS.valid_file, vocab, SEQ_LEN)
print('valid_pairs: {}'.format(len(valid_dataset)))
print("Load data successfully.")

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        esim = ESIM(
            sequence_length=SEQ_LEN,
            vocab_size=len(vocab),
            embedding_size=FLAGS.embedding_dim,
            vocab=vocab,
            rnn_size=FLAGS.rnn_size)
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        starter_learning_rate = 0.001
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                                   5000, 0.96, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(esim.mean_loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        """
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)
        """

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        """
        loss_summary = tf.scalar_summary("loss", esim.mean_loss)
        acc_summary = tf.scalar_summary("accuracy", esim.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph_def)
        """

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_question, x_answer, x_question_len, x_answer_len, x_lastTurn, x_lastTurn_len, q_id_list, as_id_list, x_target):
            """
            A single training step
            """
            feed_dict = {
              esim.question: x_question,
              esim.answer: x_answer,
              esim.question_len: x_question_len,
              esim.answer_len: x_answer_len,
              esim.target: x_target,
              esim.dropout_keep_prob: FLAGS.dropout_keep_prob,
              esim.lastTurn: x_lastTurn,
              esim.lastTurn_len: x_lastTurn_len
            }

            _, step, loss, accuracy, predicted_prob = sess.run(
                [train_op, global_step, esim.mean_loss, esim.accuracy, esim.probs],
                feed_dict)

            if step % 100 = 0:
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            #train_summary_writer.add_summary(summaries, step)


        def dev_step():
            results = []
            num_test = 0
            num_correct = 0.0
            valid_batches = data_helpers.batch_iter(valid_dataset, FLAGS.batch_size, 1, SEQ_LEN, shuffle=True)
            for valid_batch in valid_batches:
                x_question, x_answer, x_question_len, x_answer_len, x_lastTurn, x_lastTurn_len, q_id_list, as_id_list, x_target = valid_batch
                feed_dict = {
                  esim.question: x_question,
                  esim.answer: x_answer,
                  esim.question_len: x_question_len,
                  esim.answer_len: x_answer_len,
                  esim.target: x_target,
                  esim.dropout_keep_prob: 1.0,
                  esim.lastTurn: x_lastTurn,
                  esim.lastTurn_len: x_lastTurn_len
                }
                batch_accuracy, predicted_prob = sess.run([esim.accuracy, esim.probs], feed_dict)
                num_test += len(predicted_prob)
                if num_test % 10 == 0:
                    print(num_test)

                num_correct += len(predicted_prob) * batch_accuracy
                results.append( (predicted_prob, x_target) )

            probs_list = []
            labels_list = [] 
            for probs, labels in results:
                probs_list.append(probs)
                labels_list.append(labels)
            probs_aggre = np.concatenate(probs_list, axis=0)
            labels_aggre = np.concatenate(labels_list, axis=0)

            #calculate top-1 precision
            print('num_test_samples: {}  test_accuracy: {}'.format(num_test, num_correct/num_test))
            recall, mrr = recall_metrics.compute_recall(probs_aggre, labels_aggre)
            print('recall@1: {}, recall@2: {}, recall@5: {}, recall@10: {}'.format(recall['@1'], recall['@2'], recall['@5'], recall['@10']))
            
            return recall['@1']

        best_recall_at_1 = 0.0
        batches = data_helpers.batch_iter(train_dataset, FLAGS.batch_size, FLAGS.num_epochs, SEQ_LEN, shuffle=True)
        for batch in batches:
            x_question, x_answer, x_question_len, x_answer_len, x_lastTurn, x_lastTurn_len, q_id_list, as_id_list, targets = batch
            train_step(x_question, x_answer, x_question_len, x_answer_len, x_lastTurn, x_lastTurn_len, q_id_list, as_id_list, targets)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                valid_recall_at_1 = dev_step()
                if valid_recall_at_1 > best_recall_at_1:
                    best_recall_at_1 = valid_recall_at_1
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

