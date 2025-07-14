@echo off
setlocal

set "MODEL_NAME=google/t5-v1_1-base"
set "BATCH_SIZE=16"
set "DATASET=wos-all"
set "OPTIMIZER=adafactor"
set "DATA_BENCH=hiera_multilabel_bench"
set "LEARNING_RATE=5e-4"
set "T5_LABEL_ENCODING=true"
set "STATIC_LABEL_ENCODING=true"
set "ZLPR=true"
set "USE_BIDIRECTIONNAL_ATTENTION=false"
set "PYTHONPATH=."
set "CUDA_VISIBLE_DEVICES=1"
set "TOKENIZERS_PARALLELISM=false"

for %%S in (84) do (
    set "SEED=%%S"
    call python experiments/train_classifier.py ^
        --use_t5_label_encoding %T5_LABEL_ENCODING% ^
        --static_label_encoding %STATIC_LABEL_ENCODING% ^
        --use_bidirectional_attention %USE_BIDIRECTIONNAL_ATTENTION% ^
        --model_name %MODEL_NAME% ^
        --dataset_name %DATASET% ^
        --dataset_bench %DATA_BENCH% ^
        --use_zlpr_loss %ZLPR% ^
        --output_dir data/output/%OPTIMIZER%/%DATASET%/seed_%%S ^
        --max_seq_length 512 ^
        --do_train ^
        --do_eval ^
        --do_pred ^
        --overwrite_output_dir ^
        --load_best_model_at_end ^
        --metric_for_best_model macro-micro-f1 ^
        --greater_is_better True ^
        --eval_strategy steps ^
        --save_strategy steps ^
        --num_train_epochs 30 ^
        --per_device_train_batch_size %BATCH_SIZE% ^
        --per_device_eval_batch_size %BATCH_SIZE% ^
        --seed %%S ^
        --warmup_ratio 0.1 ^
        --optim %OPTIMIZER% ^
        --gradient_accumulation_steps 1 ^
        --eval_accumulation_steps 1 ^
        --learning_rate %LEARNING_RATE%
)

cmd /k