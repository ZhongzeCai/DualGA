export PYTHONPATH="${PYTHONPATH}:path/of/WatermarkAlgorithm"

batch_DIR=$PWD
MODEL_DIR="path/of/llama-7b"

data_root_folder="path/of/generated/records"
filepath_list="${data_root_folder}/OpenAI_t05/OpenAI_t05_record.jsonl, \
                ${data_root_folder}/OpenAI_t08/OpenAI_t08_record.jsonl, \
                ${data_root_folder}/OpenAI_t1/OpenAI_t1_record.jsonl"
 
filename_list="OpenAI_t05, \
                OpenAI_t08, \
                OpenAI_t1"
                

temp_list="05,08,1"


OUTPUT_DIR="${batch_DIR}"


######################## insert ##########################################

python path/of/evaluate.py \
    --json_path=None \
    --gened_key=None \
    --model_name_or_path=$MODEL_DIR \
    --method="openai" \
    --seeding='hash' \
    --ngram=1 \
    --gamma=0.5 \
    --scoring_method="v2" \
    --output_dir="$OUTPUT_DIR" \
    --exp_name=None \
    --nsamples=500 \
    --filepath_list="$filepath_list" \
    --filename_list="$filename_list" \
    --temp_list="$temp_list" \
    --auto_inferGamma=False \
    --attack_mode="insert" \
    --attack_eps_list="005,01,03,05,07"


######################## delete ##########################################

python path/of/evaluate.py \
    --json_path=None \
    --gened_key=None \
    --model_name_or_path=$MODEL_DIR \
    --method="openai" \
    --seeding='hash' \
    --ngram=1 \
    --gamma=0.5 \
    --scoring_method="v2" \
    --output_dir="$OUTPUT_DIR" \
    --exp_name=None \
    --nsamples=500 \
    --filepath_list="$filepath_list" \
    --filename_list="$filename_list" \
    --temp_list="$temp_list" \
    --auto_inferGamma=False \
    --attack_mode="del" \
    --attack_eps_list="005,01,03,05,07"


######################## substitude ##########################################

python path/of/evaluate.py \
    --json_path=None \
    --gened_key=None \
    --model_name_or_path=$MODEL_DIR \
    --method="openai" \
    --seeding='hash' \
    --ngram=1 \
    --gamma=0.5 \
    --scoring_method="v2" \
    --output_dir="$OUTPUT_DIR" \
    --exp_name=None \
    --nsamples=500 \
    --filepath_list="$filepath_list" \
    --filename_list="$filename_list" \
    --temp_list="$temp_list" \
    --auto_inferGamma=False \
    --attack_mode="sub" \
    --attack_eps_list="005,01,03,05,07"
    