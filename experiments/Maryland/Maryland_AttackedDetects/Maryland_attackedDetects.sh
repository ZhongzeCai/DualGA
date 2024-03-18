export PYTHONPATH="${PYTHONPATH}:path/of/WatermarkAlgorithm"

batch_DIR=$PWD
MODEL_DIR="path/of/llama-7b"

data_root_folder="path/of/generated/records"
filepath_list="${data_root_folder}/Maryland_d0_g05/Maryland_d0_g05_record.jsonl, \
                ${data_root_folder}/Maryland_d0_g025/Maryland_d0_g025_record.jsonl, \
                ${data_root_folder}/Maryland_d1_g05/Maryland_d1_g05_record.jsonl, \
                ${data_root_folder}/Maryland_d1_g025/Maryland_d1_g025_record.jsonl, \
                ${data_root_folder}/Maryland_d2_g05/Maryland_d2_g05_record.jsonl, \
                ${data_root_folder}/Maryland_d2_g025/Maryland_d2_g025_record.jsonl, \
                ${data_root_folder}/Maryland_d5_g05/Maryland_d5_g05_record.jsonl, \
                ${data_root_folder}/Maryland_d5_g025/Maryland_d5_g025_record.jsonl, \
                ${data_root_folder}/Maryland_d10_g05/Maryland_d10_g05_record.jsonl, \
                ${data_root_folder}/Maryland_d10_g025/Maryland_d10_g025_record.jsonl"
 
filename_list="Maryland_d0_g05, \
                Maryland_d0_g025, \
                Maryland_d1_g05, \
                Maryland_d1_g025, \
                Maryland_d2_g05, \
                Maryland_d2_g025, \
                Maryland_d5_g05, \
                Maryland_d5_g025, \
                Maryland_d10_g05, \
                Maryland_d10_g025"


gamma_list="05,025,05,025,05,025,05,025,05,025"


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
    --gamma_list="$gamma_list" \
    --auto_inferGamma=False \
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
    --gamma_list="$gamma_list" \
    --auto_inferGamma=False \
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
    --gamma_list="$gamma_list" \
    --auto_inferGamma=False \
    --attack_mode="sub" \
    --attack_eps_list="005,01,03,05,07"
    