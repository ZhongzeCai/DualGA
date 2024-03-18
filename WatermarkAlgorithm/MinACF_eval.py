# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
python main_eval.py \
    --json_path examples/results.jsonl --text_key result \
    --json_path_ori data/alpaca_data.json --text_key_ori output \
    --do_wmeval True --method openai --seeding hash --ngram 2 --scoring_method v2 \
    --payload 0 --payload_max 4 \
    --output_dir output/ 
"""

import argparse
from typing import List
import os
import json
import time

import tqdm
import pandas as pd
import numpy as np

import torch
from peft import PeftModel    
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

from utils.three_bricks_evaluate import (OpenaiDetector, OpenaiDetectorZ, 
                MarylandDetector, MarylandDetectorZ, MinACFDetector)
import utils
from utils.io import write_jsonlines
from utils.submitit import str2bool
from utils.attack_light import corrupt
import copy


def list_of_strings(arg):
    return [arg_str.strip() for arg_str in arg.split(',')]

def eva_args_parser():
    parser = argparse.ArgumentParser('Args', add_help=False)

    # model parameters
    parser.add_argument('--json_path', type=str, required=True)
    parser.add_argument('--gened_key', type=str, default=None,
                        help='key to access text in json dict')
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument('--text_key_ori', type=str, default=None,
                        help='key to access text in json dict')

    # watermark parameters
    parser.add_argument('--method', type=str, default='none',
                        help='watermark detection method')
    parser.add_argument('--seeding', type=str, default='hash', 
                        help='seeding method for rng key generation as introduced in https://github.com/jwkirchenbauer/lm-watermarking')
    parser.add_argument('--ngram', type=int, default=4, 
                        help='watermark context width for rng key generation')
    parser.add_argument('--gamma', type=float, default=0.25, 
                        help='gamma for maryland: proportion of greenlist tokens')
    parser.add_argument('--salt_key', type=int, default=35317, 
                        help='hash key for rng key generation')
    parser.add_argument('--scoring_method', type=str, default='none', 
                        help='method for scoring. choose between: \
                        none (score every tokens), v1 (score token when wm context is unique), \
                        v2 (score token when {wm context + token} is unique')


    # multibit
    parser.add_argument('--payload', type=int, default=0, 
                        help='message')
    parser.add_argument('--payload_max', type=int, default=0, 
                        help='maximal message')
    
    # attack
    
    # useless
    parser.add_argument('--delta', type=float, default=2.0, 
                        help='delta for maryland (useless for detection)')
    parser.add_argument('--temperature', type=float, default=1., 
                        help='temperature for generation (useless for detection)')

    # expe parameters
    parser.add_argument('--nsamples', type=int, default=None, 
                        help='number of samples to evaluate, if None, take all texts')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--split', type=int, default=None,
                        help='split the texts in nsplits chunks and chooses the split-th chunk. \
                        Allows to run in parallel. \
                        If None, treat texts as a whole')
    parser.add_argument('--nsplits', type=int, default=None,
                        help='number of splits to do. If None, treat texts as a whole')
    parser.add_argument('--filepath_list', type=list_of_strings,
                        default=[],
                        help='a list of evaluating paths')
    parser.add_argument('--filename_list', type=list_of_strings,
                        default=None,
                        help='a list of file names')
    parser.add_argument('--gamma_list', type=list_of_strings,
                        default=None,
                        help='a list of gammas')
    parser.add_argument('--D_list', type=list_of_strings,
                        default=None,
                        help='a list of D values for DualGD KL')
    parser.add_argument('--temp_list', type=list_of_strings)
    parser.add_argument(
        "--auto_inferGamma",
        type=str2bool,
        default=False,
        help="Infer gamma from D",
    )
    parser.add_argument(
        "--attack_mode",
        type=str,
        default=None,
        help="choose mode in del, insert, sub",
    )
    parser.add_argument('--attack_eps', type=float,
                        default=None,
                        help='attack eps')
    parser.add_argument('--attack_eps_list', type=list_of_strings,
                        default=None,
                        help='a list of attack eps values')
    


    args = parser.parse_args()
    return args

def load_full_results(json_path: str, nsamples: int=None) -> List[str]:
    with open(json_path, "r") as f:
        if json_path.endswith('.json'):
            prompts = json.loads(f.read())
        else:
            prompts = [json.loads(line) for line in f.readlines()] # load jsonl

    if nsamples:
        prompts = prompts[:nsamples]
    return prompts

def main_evaluate(args):

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
        "07": 0.7,
        "08": 0.8,
        "04": 0.4,
        "06": 0.6,
        "08": 0.8,
        "0": 0,
        "12":1.2,
        "15":1.5
    }

    # print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    # print("{}".format(args).replace(', ', ',\n'))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    start = time.time()

    # build tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    
    assert args.method == "MinACF"
    
    detector = MinACFDetector(tokenizer, args.ngram, args.seed, args.seeding, args.salt_key, gamma=args.gamma, delta=args.delta)

    # load results and (optional) do splits
    results = load_full_results(json_path=args.json_path, nsamples=args.nsamples)
    print(f"Loaded {len(results)} results.")
    if args.split is not None:
        nresults = len(results)
        left = nresults * args.split // args.nsplits 
        right = nresults * (args.split + 1) // args.nsplits if (args.split != args.nsplits - 1) else nresults
        results = results[left:right]
        print(f"Creating results from {left} to {right}")

    # attack results
    attack_mode = args.attack_mode
    attack_eps_list = args.attack_eps_list

    if attack_mode:

        assert attack_mode in ["del", "insert", "sub"]

        deletion_eps = 0
        insertion_eps = 0
        substitution_eps = 0

        for dic in results:

            for eps in attack_eps_list:

                if attack_mode == "del":

                    deletion_eps = mapper[eps]

                elif attack_mode == "insert":

                    insertion_eps = mapper[eps]

                elif attack_mode == "sub":

                    substitution_eps = mapper[eps]


                dic[args.gened_key + "_" + attack_mode + eps] = corrupt(dic[args.gened_key],
                        tokenizer, deletion_eps, insertion_eps, substitution_eps) 

    # evaluate
    os.makedirs(args.output_dir, exist_ok=True)

    for ii, full_dic in tqdm.tqdm(enumerate(results), total=len(results)):

        text = full_dic[args.gened_key]
        

        scores_no_aggreg = detector.get_scores_by_t([text], scoring_method=args.scoring_method, payload_max=args.payload_max)   
        pvalue = detector.get_MinACF_pvalues(scores_no_aggreg) 

        results[ii]['pvalue'] =  float(pvalue[0])
        results[ii]['log10_pvalue'] = float(np.log10(pvalue[0]))
        

        # if attacked, repeat the above process on attacked text again

        if args.attack_mode:

            for eps in attack_eps_list:

                attacked_text = full_dic[args.gened_key + "_" + attack_mode + eps]

                scores_no_aggreg = detector.get_scores_by_t([attacked_text], scoring_method=args.scoring_method, payload_max=args.payload_max)   
       
                pvalue = detector.get_MinACF_pvalues(scores_no_aggreg) 

                results[ii]['pvalue_attacked_'+attack_mode + eps] =  float(pvalue[0])
                results[ii]['log10_pvalue_attacked_'+attack_mode + eps] = float(np.log10(pvalue[0]))


        '''
        if args.json_path_ori is not None:
            # compute sbert score

            
            text_orig = results_orig[ii]
            xs = sbert_model.encode([text, text_orig], convert_to_tensor=True)
            score_sbert = cossim(xs[0], xs[1]).item()
            log_stat['score_sbert'] = score_sbert
            

        '''
    write_jsonlines(results, filename=os.path.join(args.output_dir, args.exp_name+'_with_scores.jsonl'))

    print(f"time spent: {time.time()-start}")



def batch_evaluate(args):

    # batch evaluate relys on gamma, which is implicit for auto DualGD

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
        "07": 0.7,
        "08": 0.8,
        "04": 0.4,
        "06": 0.6,
        "08": 0.8,
        "0": 0,
        "12": 1.2,
        "15": 1.5
    }

    infer_mapper = {
                        "01": 0.467,
                        "02": 0.433,
                        "03": 0.398,
                        "04": 0.362,
                        "05": 0.325,
                        "07": 0.237,
                        "08": 0.18
                    }
    
    filepath_list = args.filepath_list
    if len(filepath_list) == 0:
        main_evaluate(args)

    else:

        if args.method in ["maryland", "MinACF"]:
            filename_list = args.filename_list
            assert len(filename_list) == len(filepath_list)
            if not args.auto_inferGamma:

                gamma_list = args.gamma_list
                assert len(filename_list) == len(gamma_list)
                gamma_list = [mapper[gamma] for gamma in gamma_list]

            else:

                D_list = args.D_list
                assert len(D_list) == len(filename_list)
                gamma_list = [infer_mapper[DD] for DD in D_list]


            # iterate over files

            for i in range(len(filepath_list)):

                filepath = filepath_list[i]
                filename = filename_list[i]
                new_gamma = gamma_list[i]

                new_args = copy.deepcopy(args)

                new_args.json_path = filepath
                new_args.gened_key = filename+"_generated_str"
                new_args.gamma=new_gamma
                new_args.output_dir=args.output_dir+"/"+filename
                new_args.exp_name = filename

                if args.attack_mode:

                    new_args.output_dir=args.output_dir+"/"+filename+"_"+args.attack_mode
                    
                main_evaluate(new_args)
        
        elif args.method == "openai":

            filename_list = args.filename_list
            assert len(filename_list) == len(filepath_list)

            # iterate over files

            for i in range(len(filepath_list)):

                filepath = filepath_list[i]
                filename = filename_list[i]
                temp = args.temp_list[i]

                new_args = copy.deepcopy(args)

                new_args.json_path = filepath
                new_args.gened_key = filename+"_generated_str"
                new_args.temperature=temp
                new_args.output_dir=args.output_dir+"/"+filename
                new_args.exp_name = filename

                if args.attack_mode:

                    new_args.output_dir=args.output_dir+"/"+filename+"_"+args.attack_mode
                    
                main_evaluate(new_args)

            

        else:

            raise NotImplementedError



    
    


if __name__ == "__main__":
    args = eva_args_parser()

    batch_evaluate(args)

