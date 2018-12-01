cur_dir=`pwd`
parentdir="$(dirname $cur_dir)"

latest_run=`ls -dt runs/* |head -n 1`
latest_checkpoint=${latest_run}/checkpoints
echo $latest_checkpoint

DATA_DIR=${parentdir}/data
test_file=$DATA_DIR/ubuntu_dev_subtask_1.json
vocab_file=$DATA_DIR/vocab.txt

PKG_DIR=${parentdir}
PYTHONPATH=${PKG_DIR}:$PYTHONPATH CUDA_VISIBLE_DEVICES=1 python -u ${PKG_DIR}/model/eval.py \
                  --test_file $test_file \
                  --vocab_file $vocab_file \
                  --max_sequence_length 160 \
                  --batch_size 2 \
                  --checkpoint_dir $latest_checkpoint > log_test.txt 2>&1 &
