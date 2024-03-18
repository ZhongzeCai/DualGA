export PYTHONPATH="${PYTHONPATH}:path/of/WatermarkAlgorithm"

RUN_NAME=Maryland

batch_DIR=$PWD

DATA_DIR="path/of/dataset"


MODEL_DIR="path/of/llama-7b"

GENERATION_OUTPUT_DIR="$batch_DIR"

accelerate launch --multi_gpu --num_processes 4\
    path/of/generate.py \
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
    --gamma=0.1 \
    --delta=1.0 \
    --run_name="$RUN_NAME"_gen \
    --wandb=False \
    --verbose=False \
    --output_dir=$GENERATION_OUTPUT_DIR \
    --generation_batch_size=16 \
    --distributed=True \
    --load_fp16=False  \
    --DEBUG=False \
    --exp_name=$exp_name \
    --wm_algorithm="Maryland" \
    --mass=True \
    --delta_list=0,1,2,5,10 \
    --gamma_list=025,05
