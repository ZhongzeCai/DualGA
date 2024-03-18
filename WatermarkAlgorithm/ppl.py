
import argparse
from utils.submitit import str2bool
import os
import time
import json
from utils.load_model import load_model
from utils.perplexity_calculator import ppl_for_wm_gened
from utils.io import write_jsonlines



def ppl_parse_args():

    parser = argparse.ArgumentParser(
        description="use larger model to calculate PPL"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/opt-1.3b",
        help="Main model, path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--load_fp16",
        type=str2bool,
        default=False,
        help="Whether to run model in float16 precsion.",
    )
    parser.add_argument(
        "--use_gpu",
        type=str2bool,
        default=True,
        help="Whether to run inference and watermark hashing/seeding/permutation on gpu.",
    )
    parser.add_argument(
        "--modelName",
        type=str,
        default="llama13b",
    )
    parser.add_argument(
        "--data_dir",
        type=str
    )


    args = parser.parse_args()

    return args


def list_relative_jsonl_file_paths(directory):
    relative_paths = []
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        # Iterate over files in the current directory
        for file in files:
            # Check if the file contains "jsonl" in its name
            if "jsonl" in file:
                # Construct relative path for each file
                relative_path = os.path.relpath(os.path.join(root, file), directory)
                relative_paths.append(directory + "/" + relative_path)
    return relative_paths


if __name__ == "__main__":

    start = time.time()

    args = ppl_parse_args()

    all_jsons = list_relative_jsonl_file_paths(args.data_dir)

    model, tokenizer, _ = load_model(args)

    modelName = args.modelName

    for path in all_jsons:

        print("loaded: "+path)
        sub_start = time.time()

        data=[]
        with open(path, 'r') as file:
            for line in file:
                # Convert each line to a dictionary and append to the list
                data.append(json.loads(line))

        ppl_for_wm_gened(data, model, tokenizer, modelName)

        write_jsonlines(data, path)
        print("subtime: ",time.time() - sub_start)

    print("time: ",str(time.time()-start))



        
