MODEL_NAME='google/t5-v1_1-base'
BATCH_SIZE=16
DATASET='wos-all'
OPTIMIZER='adafactor'
DATA_BENCH='hiera_multilabel_bench'
LEARNING_RATE=5e-4
T5_LABEL_ENCODING=true
STATIC_LABEL_ENCODING=true
ZLPR=true
USE_BIDIRECTIONNAL_ATTENTION=false
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
for SEED in 84
do
  python experiments/train_classifier.py \
  --use_t5_label_encoding ${T5_LABEL_ENCODING} \
  --static_label_encoding ${STATIC_LABEL_ENCODING} \
  --use_bidirectional_attention ${USE_BIDIRECTIONNAL_ATTENTION} \
  --model_name ${MODEL_NAME} \
  --dataset_name ${DATASET} \
  --dataset_bench ${DATA_BENCH}\
  --use_zlpr_loss ${ZLPR}\
  --output_dir data/output/${OPTIMIZER}/${DATASET}/seed_${SEED} \
  --max_seq_length 512 \
  --do_train \
  --do_eval \
  --do_pred \
  --overwrite_output_dir \
  --load_best_model_at_end \
  --metric_for_best_model macro-micro-f1 \
  --greater_is_better True \
  --eval_strategy steps \
  --save_strategy steps \
  --num_train_epochs 30 \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --per_device_eval_batch_size ${BATCH_SIZE} \
  --seed ${SEED} \
  --warmup_ratio 0.1 \
  --optim ${OPTIMIZER} \
  --gradient_accumulation_steps 1 \
  --eval_accumulation_steps 1 \
  --learning_rate ${LEARNING_RATE}

done
$SHELL