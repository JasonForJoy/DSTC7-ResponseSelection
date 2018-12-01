import tensorflow as tf
import numpy as np
from model import data_helpers
import recall_metrics

# Files
tf.flags.DEFINE_string("test_file", "", "test file containing (question, positive and negative answer ids)")
tf.flags.DEFINE_string("vocab_file", "", "vocabulary file (map word to integer)")

# Hyperparameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("max_sequence_length", 160, "max sequence length")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

vocab = data_helpers.loadVocab(FLAGS.vocab_file)
print("vocabulary size: {}".format(len(vocab)))

SEQ_LEN = FLAGS.max_sequence_length
test_dataset = data_helpers.loadDataset(FLAGS.test_file, vocab, SEQ_LEN)
print('test_pairs: {}'.format(len(test_dataset)))

print("\nEvaluating...\n")

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
print(checkpoint_file)

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        question_x = graph.get_operation_by_name("question").outputs[0]
        answer_x   = graph.get_operation_by_name("answer").outputs[0]
        lastTurn_x = graph.get_operation_by_name("lastTurn").outputs[0]

        question_len_x = graph.get_operation_by_name("question_len").outputs[0]
        answer_len_x   = graph.get_operation_by_name("answer_len").outputs[0]
        lastTurn_len_x = graph.get_operation_by_name("lastTurn_len").outputs[0]

        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        prob = graph.get_operation_by_name("convolution-1/prob").outputs[0]

        results = []
        num_test = 0
        test_batches = data_helpers.batch_iter(test_dataset, FLAGS.batch_size, 1, SEQ_LEN, shuffle=False)
        for test_batch in test_batches:
            x_question, x_answer, x_question_len, x_answer_len, x_lastTurn, x_lastTurn_len, x_q_id_list, x_as_id_list, x_target = test_batch
            feed_dict = {
                question_x: x_question,
                answer_x: x_answer,
                lastTurn_x: x_lastTurn,
                question_len_x: x_question_len,
                answer_len_x: x_answer_len,
                lastTurn_len_x: x_lastTurn_len,
                dropout_keep_prob: 1.0
            }
            predicted_prob = sess.run(prob, feed_dict)
            num_test += len(predicted_prob)
            print('num_test_sample={}'.format(num_test))

            results.append( (predicted_prob, x_target, x_q_id_list, x_as_id_list) )

probs_total = []
labels_total = []
q_id_total = []
as_id_total = []
for i, batch in enumerate(results):
    probs, labels, q_id_list, as_id_list = batch
    probs_total.append(probs)
    labels_total.append(labels)
    q_id_total.extend(q_id_list)
    as_id_total.extend(as_id_list)
probs_total = np.concatenate(probs_total, axis=0)
labels_total = np.concatenate(labels_total, axis=0)

num_data = len(q_id_total)
print("num_data = {}".format(num_data))

print("probs_total: {}".format(probs_total.shape))
print("labels_total: {}".format(labels_total.shape))

recall, mrr = recall_metrics.compute_recall(probs_total, labels_total)
print('recall@1: {}, recall@2: {}, recall@5: {}, recall@10: {}, recall@20: {}, recall@50: {}, MRR: {}'.format\
    (recall['@1'], recall['@2'], recall['@5'], recall['@10'], recall['@20'], recall['@50'], mrr))