export PYTHONPATH="${PYTHONPATH}:path/of/WatermarkAlgorithm"

RUN_NAME=DualGD_kl

batch_DIR=$PWD

DATA_DIR="path/of/dataset"


MODEL_DIR="path/of/llama-7b"


exp_name=DualGD_kl
GENERATION_OUTPUT_DIR="$batch_DIR"

accelerate launch --multi_gpu --num_processes 4\
    path/of/DualGD_generate.py \
    --n_proc=4 \
    --model_name=$MODEL_DIR \
    --dataset_name_or_path=$DATA_DIR \
    --dataset_config_name='realnewslike' \
    --max_new_tokens=200 \
    --min_prompt_tokens=50 \
    --min_generations=500 \
    --input_truncation_strategy=completion_length \
    --input_filtering_strategy=prompt_and_completion_length \
    --output_filtering_strategy=max_new_tokens \
    --seeding_scheme=selfhash \
    --run_name="$RUN_NAME"_gen \
    --wandb=False \
    --verbose=False \
    --output_dir=$GENERATION_OUTPUT_DIR \
    --generation_batch_size=16 \
    --distributed=True \
    --load_fp16=False  \
    --DEBUG=False \
    --exp_name=$exp_name \
    --h_func="2x2" \
    --wm_algorithm="DualGD" \
    --version="KL" \
    --eta_list=05 \
    --gamma_list=05 \
    --D_list=01,02,03,04,05 \
    --automode=True \
    --autoeta=10
