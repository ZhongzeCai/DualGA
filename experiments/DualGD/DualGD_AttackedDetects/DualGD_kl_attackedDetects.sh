export PYTHONPATH="${PYTHONPATH}:path/of/WatermarkAlgorithm"

batch_DIR=$PWD
MODEL_DIR="path/of/llama-7b"

data_root_folder="path/of/generated/records"
filepath_list="${data_root_folder}/DualGD_auto_D01/DualGD_auto_D01_record.jsonl, \
                ${data_root_folder}/DualGD_auto_D02/DualGD_auto_D02_record.jsonl, \
                ${data_root_folder}/DualGD_auto_D03/DualGD_auto_D03_record.jsonl, \
                ${data_root_folder}/DualGD_auto_D04/DualGD_auto_D04_record.jsonl, \
                ${data_root_folder}/DualGD_auto_D05/DualGD_auto_D05_record.jsonl"
 
filename_list="DualGD_auto_D01, \
                DualGD_auto_D02, \
                DualGD_auto_D03, \
                DualGD_auto_D04, \
                DualGD_auto_D05"


D_list="01,02,03,04,05"


OUTPUT_DIR="${batch_DIR}"


######################## insert ##########################################

python path/of/evaluate.py \
    --json_path=None \
    --gened_key=None \
    --model_name_or_path=$MODEL_DIR \
    --method="maryland" \
    --seeding='hash' \
    --ngram=1 \
    --gamma=0.5 \
    --scoring_method="v2" \
    --output_dir="$OUTPUT_DIR" \
    --exp_name=None \
    --nsamples=500 \
    --filepath_list="$filepath_list" \
    --filename_list="$filename_list" \
    --D_list="$D_list" \
    --auto_inferGamma=True \
    --attack_mode="insert" \
    --attack_eps_list="005,01,03,05,07"


######################## delete ##########################################

python path/of/evaluate.py \
    --json_path=None \
    --gened_key=None \
    --model_name_or_path=$MODEL_DIR \
    --method="maryland" \
    --seeding='hash' \
    --ngram=1 \
    --gamma=0.5 \
    --scoring_method="v2" \
    --output_dir="$OUTPUT_DIR" \
    --exp_name=None \
    --nsamples=500 \
    --filepath_list="$filepath_list" \
    --filename_list="$filename_list" \
    --D_list="$D_list" \
    --auto_inferGamma=True \
    --attack_mode="del" \
    --attack_eps_list="005,01,03,05,07"


######################## substitude ##########################################

python path/of/evaluate.py \
    --json_path=None \
    --gened_key=None \
    --model_name_or_path=$MODEL_DIR \
    --method="maryland" \
    --seeding='hash' \
    --ngram=1 \
    --gamma=0.5 \
    --scoring_method="v2" \
    --output_dir="$OUTPUT_DIR" \
    --exp_name=None \
    --nsamples=500 \
    --filepath_list="$filepath_list" \
    --filename_list="$filename_list" \
    --D_list="$D_list" \
    --auto_inferGamma=True \
    --attack_mode="sub" \
    --attack_eps_list="005,01,03,05,07"
    