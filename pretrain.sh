       
       
       
CHECKPOINT_PATH=checkpoints/bert_345m
VOCAB_FILE=../indic_corpus/vocab_files/as-32000.txt
DATA_PATH=as-bert_text_sentence

BERT_ARGS="--num-layers 24 \
           --hidden-size 1024 \
           --num-attention-heads 16 \
           --seq-length 512 \
           --max-position-embeddings 512 \
           --lr 0.0001 \
           --lr-decay-iters 990000 \
           --train-iters 2000000 \
           --min-lr 0.00001 \
           --lr-warmup-fraction 0.01 \
	   --micro-batch-size 4 \
           --global-batch-size 8 \
           --vocab-file $VOCAB_FILE \
           --split 99,1,0 \
           --fp16"

OUTPUT_ARGS="--log-interval 10 \
             --save-interval 500 \
             --eval-interval 100 \
             --eval-iters 10 \
             --tensorboard-dir ./tb_logs \
             --activations-checkpoint-method uniform"

python -m torch.distributed.launch pretrain_bert.py \
       $BERT_ARGS \
       $OUTPUT_ARGS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH