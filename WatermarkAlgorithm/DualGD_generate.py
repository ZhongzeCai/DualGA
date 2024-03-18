import argparse
from utils.submitit import str2bool
import time
from accelerate import Accelerator
from accelerate.utils import gather_object
from utils.data_processing import load_hf_dataset, tokenize_the_prompt, construct_reclaimable_dataset
from utils.load_model import load_model
from utils.DualGD_generator import DualGDGenerator, DualGDGenerator_KL, MinACFGenerator
from torch.utils.data.dataloader import DataLoader
import torch
from utils.io import write_jsonlines
import os
import copy
import numpy as np


def list_of_strings(arg):
    return arg.split(',')
 

def gen_parse_args():

    parser = argparse.ArgumentParser(
        description="Run watermarked huggingface LM generation pipeline"
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
        default=True,
        help="Whether to run model in float16 precsion.",
    )
    parser.add_argument(
        "--use_gpu",
        type=str2bool,
        default=True,
        help="Whether to run inference and watermark hashing/seeding/permutation on gpu.",
    )
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        default="c4",
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="The split of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--stream_dataset",
        type=str2bool,
        default=True,
        help="Whether to stream the dataset.",
    )
    parser.add_argument(
        "--columns_to_remove",
        type=str,
        default=None,
        help="Comma separated list of columns to remove from the dataset before generation.",
    )
    parser.add_argument(
        "--shuffle_dataset",
        type=str2bool,
        default=True,
        help="Whether to shuffle the dataset before sampling.",
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=1234,
        help="The seed to use for dataset shuffle op.",
    )
    parser.add_argument(
        "--shuffle_buffer_size",
        type=int,
        default=10000,
        help="The buffer size to use for dataset shuffle op - takes n rows first, then shuffles those indices",
    )
    parser.add_argument(
        "--prompt_id",
        type=int,
        default=0,
        help="If the dataset supports multiple instruction prompts, denotes which one to use. 0 is default/no prompt.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="The number of tokens to generate using the model, and the num tokens removed from real text sample",
    )
    parser.add_argument(
        "--min_prompt_tokens",
        type=int,
        default=50,  # 500
        help="The number of examples (first N) to process from the dataset.",
    )
    parser.add_argument(
        "--min_sample_tokens",
        type=int,
        default=0,
        help="The the minimum length of raw prompt samples to consider.",
    )
    parser.add_argument(
        "--limit_indices",
        type=int,
        default=None,
        help="The number of examples (first N) to pull from the dataset, if None, pull all, and then set this arg to the number of rows in the dataset.",
    )
    parser.add_argument(
        "--min_generations",
        type=int,
        default=500,
        help="The minimum number of valid generations according to the output check strat to sample.",
    )
    parser.add_argument(
        "--input_truncation_strategy",
        type=str,
        default="completion_length",
        choices=["no_truncation", "completion_length", "prompt_length"],
        help="The strategy to use when tokenizing and truncating raw inputs to make prompts.",
    )
    parser.add_argument(
        "--input_filtering_strategy",
        type=str,
        default="completion_length",
        choices=["no_filter", "completion_length", "prompt_length", "prompt_and_completion_length"],
        help="The strategy to use when tokenizing and truncating raw inputs to make prompts.",
    )
    parser.add_argument(
        "--output_filtering_strategy",
        type=str,
        default="no_filter",
        choices=["no_filter", "max_new_tokens"],
        help=(
            f"The strategy to use when filtering/skipping rows if the model didn't ",
            f"generate enough tokens to facilitate analysis.",
        ),
    )
    parser.add_argument(
        "--use_sampling",
        type=str2bool,
        default=True,
        help=("Whether to perform sampling during generation. (non-greedy decoding)"),
    )
    parser.add_argument(
        "--sampling_temp",
        type=float,
        default=1.,
        help="The temperature to use when generating using multinom sampling",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="The top k to use when generating using top_k version of multinom sampling",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="The top p to use when generating using top_p version of sampling",
    )
    parser.add_argument(
        "--typical_p",
        type=float,
        default=1.0,
        help="The typical p to use when generating using typical decoding version of multinom sampling",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="The number of beams to use where '1' is no beam search.",
    )
    parser.add_argument(
        "--generation_seed",
        type=int,
        default=0,
        help="Seed for setting the torch rng prior to generation using any decoding scheme with randomness.",
    )
    parser.add_argument(
        "--salt_key",
        type=int,
        default=35317
    )
    parser.add_argument(
        "--generation_batch_size",
        type=int,
        default=4,
        help="The batch size to use for generation.",
    )
    parser.add_argument(
        "--seeding_scheme",
        type=str,
        default="simple_1",
        help="The seeding procedure to use for the watermark.",
    )
    parser.add_argument(
        "--seeding",
        type=str,
        default="hash",
        help="The seeding scheme in three bricks.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.1,
        help="Step size",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.25,
        help="The ratio of tokens to put in the greenlist when splitting the vocabulary",
    )
    parser.add_argument(
        "--b",
        type=float,
        default=0.2,
        help="Mean KL budget",
    )
    parser.add_argument(
        "--D",
        type=float,
        default=0.3,
        help="Min green prob increase",
    )
    parser.add_argument(
        "--ngram",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--store_spike_ents",
        type=str2bool,
        default=False,
        help=("Whether to store the spike entropies while generating with watermark processor. "),
    )
    parser.add_argument(
        "--verbose",
        type=str2bool,
        default=False,
        help="Whether to log the generations to stdout.",
    )
    parser.add_argument(
        "--wandb",
        type=str2bool,
        default=False,
        help="Whether to log to wandb.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="Watermark",
        help="The name of the wandb project.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="",
        help="The wandb entity/user for the project.",
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        default="",
        help="The comma separated list of tags to add to the wandb run.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="The unique name for the run.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The unique name for the run.",
    )
    parser.add_argument(
        "--overwrite",
        type=str2bool,
        default=False,
        help="Allow overwriting of old generation files at the same output location.",
    )
    parser.add_argument(
        "--DEBUG",
        type=str2bool,
        default=False,
        help="DEBUG.",
    )
    parser.add_argument(
        "--distributed",
        type=str2bool,
        default=False,
        help="Distributed Generate",
    )
    parser.add_argument(
        "--n_proc",
        type=int,
        help="number of processes",
    )
    parser.add_argument(
        "--wm_algorithm",
        type=str,
    )
    parser.add_argument(
        "--exp_name",
        type=str,
    )
    parser.add_argument(
        "--version",
        type=str,
        default= "DG"   # DG or KL
    )
    parser.add_argument(
        "--h_func",
        type=str,
        default= "2x2"      # choose "2x2" or "xlnx"
    )
    parser.add_argument('--eta_list', type=list_of_strings,
        help='a list of eta values')
    parser.add_argument('--gamma_list', type=list_of_strings,
        help='a list of gamma values')
    parser.add_argument('--b_list', type=list_of_strings,
        help='a list of b values')
    parser.add_argument('--D_list', type=list_of_strings,
        help='a list of D values')
    parser.add_argument(
        "--automode",
        type=str2bool,
        default=True,
        help="Automatically choose gamma and eta",
    )
    parser.add_argument(
        "--autoeta",
        type=int,
        default=1,
        help="stepsize on auto lr rate",
    )

    parser.add_argument('--init_lambda', type=float,
        default=0.2)    # default is 0.3 for DG version
    parser.add_argument('--init_lambda_KL', type=float,
        default=3)      # default is 3 for KL version


    parser.add_argument('--EXP_seed', default=1145, type=int)

    args = parser.parse_args()

    ###########################################################################
    # Argument validation and conditional setting
    ###########################################################################
    # for removing some columns to save space
    args.columns_to_remove = args.columns_to_remove.split(",") if args.columns_to_remove else []

    
    # split wandb tags
    if args.wandb_tags != "":
        args.wandb_tags = args.wandb_tags.split(",")
    else:
        args.wandb_tags = []

    
    if not args.output_dir:

        args.output_dir = args.exp_name

    return args


def DualGD_generation(wm_algorithm = "DualGD", args=None, exp_name = "DualGD_e01_g05_b02", version = "DG"):

    start = time.time()
    # args = gen_parse_args()

    accelerator = Accelerator(
        mixed_precision="bf16",
        split_batches=True
    )
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("test")

    # Load datasets and create dataloaders.

    data_iter = load_hf_dataset(args)

    model, tokenizer, _ = load_model(args)

    max_embeded = model.config.max_position_embeddings

    dataset_tokenized = tokenize_the_prompt(max_embeded, tokenizer, data_iter, args)

    # Load datasets and create dataloaders.

    padid = model.config.pad_token_id

    model_args = {
        "max_sequence_length": model.config.max_sequence_length,
        "pad_token_id": model.config.pad_token_id,
        "eos_token_id": model.config.eos_token_id,
        "is_decoder_only_model": args.is_decoder_only_model
    }

    with accelerator.main_process_first():

        reclaim_dataset = construct_reclaimable_dataset(tokenizer, 
                                                    dataset_tokenized, 
                                                    padid,
                                                    args.generation_batch_size,
                                                    reclaim_batsize = args.n_proc * args.generation_batch_size)
        
        train_dataloader = DataLoader(reclaim_dataset, batch_size=args.generation_batch_size)


    train_dataloader = accelerator.prepare(train_dataloader)

    assert wm_algorithm in ["DualGD", "MinACF"]

    if wm_algorithm == "DualGD":

    
        if version == "DG":
            wm_generator = DualGDGenerator(
                        tokenizer, 
                        ngram = args.ngram,
                        seed = args.generation_seed,
                        seeding = args.seeding,
                        salt_key = args.salt_key,
                        eta = args.eta,
                        autoeta = args.autoeta,
                        gamma = args.gamma,
                        b = args.b,
                        h_func = args.h_func,
                        init_lambda = args.init_lambda,
                        model_args = model_args,
                    )

        elif version == "KL":

            wm_generator = DualGDGenerator_KL(
                        tokenizer, 
                        ngram = args.ngram,
                        seed = args.generation_seed,
                        seeding = args.seeding,
                        salt_key = args.salt_key,
                        eta = args.eta,
                        autoeta = args.autoeta,
                        gamma = args.gamma,
                        D = args.D,
                        h_func = args.h_func,
                        init_lambda_KL = args.init_lambda_KL,
                        model_args = model_args,
                    )
        else:

            raise NotImplementedError

    elif wm_algorithm == "MinACF":

        wm_generator = MinACFGenerator(
                        tokenizer, 
                        ngram = args.ngram,
                        seed = args.generation_seed,
                        seeding = args.seeding,
                        salt_key = args.salt_key,
                        eta = args.eta,
                        autoeta = args.autoeta,
                        gamma = args.gamma,
                        b = args.b,
                        h_func = args.h_func,
                        init_lambda = args.init_lambda,
                        model_args = model_args,
                    )


    
    

    for step, batch in enumerate(train_dataloader, start=1):

        if args.DEBUG and accelerator.is_local_main_process:

            print("###########################")
            print(f"step {step}, process ID: {accelerator.process_index}")
            print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(accelerator.process_index)/1024/1024/1024))
            print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(accelerator.process_index)/1024/1024/1024))
            print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(accelerator.process_index)/1024/1024/1024))
            print("###########################")

        
        # split then generate then combine
        hidden_indices = batch[:,0]

        input_ids = batch[:,1:]

        input_width = input_ids.shape[0]
        input_length =input_ids.shape[1]
        # print(hidden_indices)

        processed_dic = wm_generator.generate_metric(
            model,
            index_list = hidden_indices,
            prompt_batch = input_ids[:, :int(input_length/2)],
            mask_batch = input_ids[:, int(input_length/2):].bool(),
            max_gen_len=args.max_new_tokens,
            temperature = args.sampling_temp,
            label= exp_name+"_generated_str"
        )
        
        
        processed_objs = gather_object(processed_dic)


        if accelerator.is_local_main_process:

            reclaim_dataset.reclaim(processed_objs)

            if step % 10 == 0:

                print(f"step {step}")

        
        if step > int(args.min_generations / args.generation_batch_size): 
            break


    if accelerator.is_local_main_process:
        accelerator.end_training()
        filename = args.output_dir + "/" + exp_name+"_record.jsonl"
        record = reclaim_dataset.retrieve_record()
        write_jsonlines(record,filename=filename)

        print(f"time spent: {time.time()-start}")

    accelerator.end_training()


def mass_DualGD_generate(wm_algorithm = "DualGD", version = "DG", args=None, exp_name = "DualGD_e01_g05_b02", \
                            eta_list = None, gamma_list = None, b_list = None, D_list = None):

    assert wm_algorithm in ["DualGD", "MinACF"]
    assert version in ["DG", "KL"]

    mapper = {
        "025": 0.25,
        "005": 0.05,
        "001": 0.01,
        "02": 0.2,
        "05": 0.5,
        "01": 0.1,
        "1": 1,
        "10": 10,
        "03": 0.3,
        "04": 0.4,
        "07": 0.7,
        "08": 0.8,
        "15": 1.5
    }


    for eta in eta_list:
        for gamma in gamma_list:

            if wm_algorithm == "MinACF" or version == "DG":
                Db_list = b_list
            elif version == "KL":
                Db_list = D_list


            for Db in Db_list:

                assert eta in mapper.keys()
                assert gamma in mapper.keys()
                assert Db in mapper.keys()

                new_args = copy.deepcopy(args)

                if new_args.automode:
                    gamma_map = {
                        "01": 0.467,
                        "02": 0.433,
                        "03": 0.398,
                        "04": 0.362,
                        "05": 0.325,
                        "07": 0.237,
                        "08": 0.18
                    }
                    init_KL_map = {
                        "01": 0.8,  # to change
                        "02": 1.5,
                        "03": 2.2,
                        "04": 3.4, # to change
                        "05": 4.,
                        "07": 6.5,
                        "08": 9.,
                    }
                    # override gamma and eta automatically
                    assert version == "KL"
                    new_exp_name = wm_algorithm + "_auto"
                    new_args.gamma = gamma_map[Db]
                    new_args.eta = 0

                else:
                
                    new_exp_name = wm_algorithm + "_e" + eta + "_g" + gamma 
                    new_args.eta = mapper[eta]
                    new_args.gamma = mapper[gamma]
                


                if wm_algorithm == "MinACF" or version == "DG":

                    new_exp_name = new_exp_name + "_b" + Db
                    new_args.b = mapper[Db]

                elif version == "KL":
                    new_exp_name = new_exp_name + "_D" + Db
                    new_args.D = mapper[Db]
                    # new_args.init_lambda_KL = np.log(1 + new_args.D / new_args.gamma) + np.log(1 + new_args.D / (1-new_args.gamma-new_args.D))
                    new_args.init_lambda_KL = init_KL_map[Db]
                    
                new_args.exp_name = new_exp_name

                
                
                new_args.output_dir = args.output_dir + "/" + new_exp_name

                os.makedirs(new_args.output_dir, exist_ok=True)

                DualGD_generation(wm_algorithm, new_args, new_exp_name, version)




if __name__ == "__main__":

    args = gen_parse_args()

    assert args.wm_algorithm in ["DualGD", "MinACF"]

    if args.wm_algorithm == "MinACF" or args.version == "DG":

        mass_DualGD_generate(
            wm_algorithm = args.wm_algorithm,
            version = args.version,
            args = args,
            exp_name = args.exp_name,
            eta_list = args.eta_list,
            gamma_list = args.gamma_list,
            b_list = args.b_list
        )

    elif args.version == "KL":

        mass_DualGD_generate(
            wm_algorithm = args.wm_algorithm,
            version = args.version,
            args = args,
            exp_name = args.exp_name,
            eta_list = args.eta_list,
            gamma_list = args.gamma_list,
            D_list = args.D_list
        )

    else:

        raise NotImplementedError