cur_dir=`pwd`
parentdir="$(dirname $cur_dir)"

DATA_DIR=${parentdir}/data
train_file=$DATA_DIR/ubuntu_train_subtask_1.json
valid_file=$DATA_DIR/ubuntu_dev_subtask_1.json

embedded_vector_file=$DATA_DIR/glove_42B_300d_vec_plus_word2vec_100.txt
vocab_file=$DATA_DIR/vocab.txt

PKG_DIR=${parentdir}
PYTHONPATH=${PKG_DIR}:$PYTHONPATH CUDA_VISIBLE_DEVICES=0 python -u ${PKG_DIR}/model/train.py \
                --train_file $train_file \
                --valid_file $valid_file \
                --vocab_file $vocab_file \
                --embeded_vector_file $embedded_vector_file \
                --rnn_size 200 \
                --embedding_dim 400 \
                --max_sequence_length 160 \
                --dropout_keep_prob 1 \
                --batch_size 2 \
                --evaluate_every 5000 > log.txt 2>&1 &
