# coding=utf-8
# Copyright 2023 Authors of "A Watermark for Large Language Models"
# available at https://arxiv.org/abs/2301.10226
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import openai
import random
import torch
import argparse
import json
import nltk
import time
import os
import tqdm

from nltk.tokenize import sent_tokenize

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

nltk.download("punkt")

SUPPORTED_ATTACK_METHODS = ["gpt", "dipper", "copy-paste", "scramble"]


def generate_dipper_paraphrases(
    data,
    model_name="kalpeshk2011/dipper-paraphraser-xxl",
    no_ctx=True,
    sent_interval=3,
    start_idx=None,
    end_idx=None,
    paraphrase_file=".output/dipper_attacks.jsonl",
    lex=20,
    order=0,
    args=None,
):
    if no_ctx:
        paraphrase_file = paraphrase_file.split(".jsonl")[0] + "_no_ctx" + ".jsonl"

    if sent_interval == 1:
        paraphrase_file = paraphrase_file.split(".jsonl")[0] + "_sent" + ".jsonl"

    output_file = (
        paraphrase_file.split(".jsonl")[0]
        + "_L_"
        + f"{lex}"
        + "_O_"
        + f"{order}"
        + "_pp"
        + ".jsonl"
    )

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            num_output_points = len([json.loads(x) for x in f.read().strip().split("\n")])
    else:
        num_output_points = 0
    print(f"Skipping {num_output_points} points")

    time1 = time.time()
    tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    print("Model loaded in ", time.time() - time1)
    # model.half()
    model.cuda()
    model.eval()

    data = (
        data.select(range(0, len(data)))
        if start_idx is None or end_idx is None
        else data.select(range(start_idx, end_idx))
    )

    # iterate over data and tokenize each instance
    w_wm_output_attacked = []
    dipper_inputs = []
    for idx, dd in tqdm.tqdm(enumerate(data), total=len(data)):
        if idx < num_output_points:
            continue
        # tokenize prefix
        if "w_wm_output_attacked" not in dd:
            # paraphrase_outputs = {}

            if args.no_wm_attack:
                if isinstance(dd["no_wm_output"], str):
                    input_gen = dd["no_wm_output"].strip()
                else:
                    input_gen = dd["no_wm_output"][0].strip()
            else:
                if isinstance(dd["w_wm_output"], str):
                    input_gen = dd["w_wm_output"].strip()
                else:
                    input_gen = dd["w_wm_output"][0].strip()

            # The lexical and order diversity codes used by the actual model correspond to "similarity" rather than "diversity".
            # Thus, for a diversity measure of X, we need to use control code value of 100 - X.
            lex_code = int(100 - lex)
            order_code = int(100 - order)

            # remove spurious newlines
            input_gen = " ".join(input_gen.split())
            sentences = sent_tokenize(input_gen)
            prefix = " ".join(dd["truncated_input"].replace("\n", " ").split())
            output_text = ""
            final_input_text = ""

            for sent_idx in range(0, len(sentences), sent_interval):
                curr_sent_window = " ".join(sentences[sent_idx : sent_idx + sent_interval])
                if no_ctx:
                    final_input_text = f"lexical = {lex_code}, order = {order_code} <sent> {curr_sent_window} </sent>"
                else:
                    final_input_text = f"lexical = {lex_code}, order = {order_code} {prefix} <sent> {curr_sent_window} </sent>"

                if idx == 0 and lex_code == 60 and order_code == 60:
                    print(final_input_text)

                final_input = tokenizer([final_input_text], return_tensors="pt")
                final_input = {k: v.cuda() for k, v in final_input.items()}

                with torch.inference_mode():
                    outputs = model.generate(
                        **final_input, do_sample=True, top_p=0.75, top_k=None, max_length=512
                    )
                outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                prefix += " " + outputs[0]
                output_text += " " + outputs[0]

            # paraphrase_outputs[f"lex_{lex_code}_order_{order_code}"] = {
            #     "final_input": final_input_text,
            #     "output": [output_text.strip()],
            #     "lex": lex_code,
            #     "order": order_code
            # }
            # dd["w_wm_output_attacked"] = paraphrase_outputs
            w_wm_output_attacked.append(output_text.strip())
            dipper_inputs.append(final_input_text)

        # with open(output_file, "a") as f:
        #     f.write(json.dumps(dd) + "\n")
    # add w_wm_output_attacked to hf dataset object as a column
    data = data.add_column("w_wm_output_attacked", w_wm_output_attacked)
    data = data.add_column(f"dipper_inputs_Lex{lex}_Order{order}", dipper_inputs)

    return data

def single_insertion(
    attack_len,
    min_token_count,
    tokenized_no_wm_output,  # dst
    tokenized_w_wm_output,  # src
):
    top_insert_loc = min_token_count - attack_len
    rand_insert_locs = torch.randint(low=0, high=top_insert_loc, size=(2,))

    # tokenized_no_wm_output_cloned = torch.clone(tokenized_no_wm_output) # used to be tensor
    tokenized_no_wm_output_cloned = torch.tensor(tokenized_no_wm_output)
    tokenized_w_wm_output = torch.tensor(tokenized_w_wm_output)

    tokenized_no_wm_output_cloned[
        rand_insert_locs[0].item() : rand_insert_locs[0].item() + attack_len
    ] = tokenized_w_wm_output[rand_insert_locs[1].item() : rand_insert_locs[1].item() + attack_len]

    return tokenized_no_wm_output_cloned


def triple_insertion_single_len(
    attack_len,
    min_token_count,
    tokenized_no_wm_output,  # dst
    tokenized_w_wm_output,  # src
):
    tmp_attack_lens = (attack_len, attack_len, attack_len)

    while True:
        rand_insert_locs = torch.randint(low=0, high=min_token_count, size=(len(tmp_attack_lens),))
        _, indices = torch.sort(rand_insert_locs)

        if (
            rand_insert_locs[indices[0]] + attack_len <= rand_insert_locs[indices[1]]
            and rand_insert_locs[indices[1]] + attack_len <= rand_insert_locs[indices[2]]
            and rand_insert_locs[indices[2]] + attack_len <= min_token_count
        ):
            break

    # replace watermarked sections into unwatermarked ones
    # tokenized_no_wm_output_cloned = torch.clone(tokenized_no_wm_output) # used to be tensor
    tokenized_no_wm_output_cloned = torch.tensor(tokenized_no_wm_output)
    tokenized_w_wm_output = torch.tensor(tokenized_w_wm_output)

    for i in range(len(tmp_attack_lens)):
        start_idx = rand_insert_locs[indices[i]]
        end_idx = rand_insert_locs[indices[i]] + attack_len

        tokenized_no_wm_output_cloned[start_idx:end_idx] = tokenized_w_wm_output[start_idx:end_idx]

    return tokenized_no_wm_output_cloned


def k_insertion_t_len(
    num_insertions,
    insertion_len,
    min_token_count,
    tokenized_dst_output,  # dst
    tokenized_src_output,  # src
    verbose=False,
):
    insertion_lengths = [insertion_len] * num_insertions

    # these aren't save to rely on indiv, need to use the min of both
    # dst_length = len(tokenized_dst_output)
    # src_length = len(tokenized_src_output) # not needed, on account of considering only min_token_count
    # as the max allowed index

    while True:
        rand_insert_locs = torch.randint(
            low=0, high=min_token_count, size=(len(insertion_lengths),)
        )
        _, indices = torch.sort(rand_insert_locs)

        if verbose:
            print(
                f"indices: {[rand_insert_locs[indices[i]] for i in range(len(insertion_lengths))]}"
            )
            print(
                f"gaps: {[rand_insert_locs[indices[i + 1]] - rand_insert_locs[indices[i]] for i in range(len(insertion_lengths) - 1)] + [min_token_count - rand_insert_locs[indices[-1]]]}"
            )

        # check for overlap condition for all insertions
        overlap = False
        for i in range(len(insertion_lengths) - 1):
            if (
                rand_insert_locs[indices[i]] + insertion_lengths[indices[i]]
                > rand_insert_locs[indices[i + 1]]
            ):
                overlap = True
                break

        if (
            not overlap
            and rand_insert_locs[indices[-1]] + insertion_lengths[indices[-1]] < min_token_count
        ):
            break

    # replace watermarked sections into unwatermarked ones

    tokenized_dst_output_cloned = torch.tensor(tokenized_dst_output)
    tokenized_src_output = torch.tensor(tokenized_src_output)

    for i in range(len(insertion_lengths)):
        start_idx = rand_insert_locs[indices[i]]
        end_idx = rand_insert_locs[indices[i]] + insertion_lengths[indices[i]]

        tokenized_dst_output_cloned[start_idx:end_idx] = tokenized_src_output[start_idx:end_idx]

    return tokenized_dst_output_cloned



def scramble_attack(example, tokenizer=None, args=None):
    # check if the example is long enough to attack
    for column in ["w_wm_output", "no_wm_output"]:
        if not check_output_column_lengths(example, min_len=args.cp_attack_min_len):
            # # if not, copy the orig w_wm_output to w_wm_output_attacked
            # NOTE changing this to return "" so that those fail/we can filter out these examples
            example[f"{column}_attacked"] = ""
            example[f"{column}_attacked_length"] = 0
        else:
            sentences = example[column].split(".")
            random.shuffle(sentences)
            example[f"{column}_attacked"] = ".".join(sentences)
            example[f"{column}_attacked_length"] = len(
                tokenizer(example[f"{column}_attacked"])["input_ids"]
            )
    return example


def gpt_attack(example, attack_prompt=None, args=None):
    assert attack_prompt, "Prompt must be provided for GPT attack"

    gen_row = example

    if args.no_wm_attack:
        original_text = gen_row["no_wm_output"]
    else:
        original_text = gen_row["w_wm_output"]

    attacker_query = attack_prompt + original_text
    query_msg = {"role": "user", "content": attacker_query}

    from tenacity import retry, stop_after_attempt, wait_random_exponential

    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(25))
    def completion_with_backoff(model, messages, temperature, max_tokens):
        return openai.ChatCompletion.create(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
        )

    outputs = completion_with_backoff(
        model=args.attack_model_name,
        messages=[query_msg],
        temperature=args.attack_temperature,
        max_tokens=args.attack_max_tokens,
    )

    attacked_text = outputs.choices[0].message.content
    assert (
        len(outputs.choices) == 1
    ), "OpenAI API returned more than one response, unexpected for length inference of the output"
    example["w_wm_output_attacked_length"] = outputs.usage.completion_tokens
    example["w_wm_output_attacked"] = attacked_text
    if args.verbose:
        print(f"\nOriginal text (T={example['w_wm_output_length']}):\n{original_text}")
        print(f"\nAttacked text (T={example['w_wm_output_attacked_length']}):\n{attacked_text}")

    return example


def dipper_attack(dataset, lex=None, order=None, args=None):
    dataset = generate_dipper_paraphrases(dataset, lex=lex, order=order, args=args)
    return dataset


def check_output_column_lengths(example, min_len=0):
    baseline_completion_len = example["baseline_completion_length"]
    no_wm_output_len = example["no_wm_output_length"]
    w_wm_output_len = example["w_wm_output_length"]
    conds = all(
        [
            baseline_completion_len >= min_len,
            no_wm_output_len >= min_len,
            w_wm_output_len >= min_len,
        ]
    )
    return conds


def tokenize_for_copy_paste(example, tokenizer=None, args=None):
    for text_col in OUTPUT_TEXT_COLUMN_NAMES:
        if text_col in example:
            example[f"{text_col}_tokd"] = tokenizer(
                example[text_col], return_tensors="pt", add_special_tokens=False
            )["input_ids"][0]
    return example


def copy_paste_attack(example, tokenizer=None, args=None):
    # check if the example is long enough to attack
    if not check_output_column_lengths(example, min_len=args.cp_attack_min_len):
        # # if not, copy the orig w_wm_output to w_wm_output_attacked
        # NOTE changing this to return "" so that those fail/we can filter out these examples
        example["w_wm_output_attacked"] = ""
        example["w_wm_output_attacked_length"] = 0
        return example

    # else, attack

    # Understanding the functionality:
    # we always write the result into the "w_wm_output_attacked" column
    # however depending on the detection method we're targeting, the
    # "src" and "dst" columns will be different. However,
    # the internal logic for these functions has old naming conventions of
    # watermarked always being the insertion src and no_watermark always being the dst

    tokenized_dst = example[f"{args.cp_attack_dst_col}_tokd"]
    tokenized_src = example[f"{args.cp_attack_src_col}_tokd"]
    min_token_count = min(len(tokenized_dst), len(tokenized_src))

    if args.cp_attack_type == "single-single":  # 1-t
        tokenized_attacked_output = single_insertion(
            args.cp_attack_insertion_len,
            min_token_count,
            tokenized_dst,
            tokenized_src,
        )
    elif args.cp_attack_type == "triple-single":  # 3-t
        tokenized_attacked_output = triple_insertion_single_len(
            args.cp_attack_insertion_len,
            min_token_count,
            tokenized_dst,
            tokenized_src,
        )
    elif args.cp_attack_type == "k-t":
        tokenized_attacked_output = k_insertion_t_len(
            args.cp_attack_num_insertions,  # k
            args.cp_attack_insertion_len,  # t
            min_token_count,
            tokenized_dst,
            tokenized_src,
            verbose=args.verbose,
        )
    elif args.cp_attack_type == "k-random":  # k-t | k>=3, t in [floor(T/2k), T/k)
        raise NotImplementedError(f"Attack type {args.cp_attack_type} not implemented")
    elif args.cp_attack_type == "triple-triple":  # 3-(k_1,k_2,k_3)
        raise NotImplementedError(f"Attack type {args.cp_attack_type} not implemented")
    else:
        raise ValueError(f"Invalid attack type: {args.cp_attack_type}")

    example["w_wm_output_attacked"] = tokenizer.batch_decode(
        [tokenized_attacked_output], skip_special_tokens=True
    )[0]
    example["w_wm_output_attacked_length"] = len(tokenized_attacked_output)

    return example