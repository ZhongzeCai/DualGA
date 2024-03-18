from datasets import load_dataset, IterableDataset
from .data.lfqa import load_lfqa
from .data.essays import load_essays
from .data.wikitext import load_wikitext
import torch
from tokenizers import Tokenizer
from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    DataCollatorWithPadding,
)
from functools import partial
from torch.utils.data.dataloader import DataLoader




def dict_remove(dic, key):

    ava_keys = list(dic.keys())
    ava_keys.remove(key)

    return {key: dic[key] for key in ava_keys}


def add_idx(example, idx):
    example.update({"idx": idx})
    return example


def load_hf_dataset(args):
    dataset_name, dataset_config_name = args.dataset_name_or_path, args.dataset_config_name

    if "lfqa" in dataset_name:
        dataset = load_lfqa(args)
        args.__dict__.update(
            {
                "truncate_input_for_prompt": False,
                "input_col_name": "prefix",
                "ref_output_col_name": "gold_completion",
            }
        )
        # other args set within the load_lfqa function
    elif dataset_name == "wikitext":
        dataset = load_wikitext(args)
        args.__dict__.update(
            {
                "truncate_input_for_prompt": True,
                "input_col_name": "text",
                "ref_output_col_name": None,
            }
        )
        # other args set within the load_wikitext function
    elif dataset_name == "essays":
        dataset = load_essays(args)
        args.__dict__.update(
            {
                "truncate_input_for_prompt": False,
                "input_col_name": "instructions",
                "ref_output_col_name": "essays",
            }
        )
    elif dataset_name == "cml_pile":
        subsets = [dataset_config_name]
        dataset = load_dataset(
            "./data/cml_pile.py",
            subsets=subsets,
            streaming=args.stream_dataset,
            split=None,
            ignore_verifications=True,
        )[args.dataset_split]
        args.__dict__.update(
            {
                "truncate_input_for_prompt": True,
                "input_col_name": "text",
                "ref_output_col_name": None,
            }
        )
    else:
        dataset = load_dataset(
            dataset_name,
            dataset_config_name,
            split=args.dataset_split,
            streaming=args.stream_dataset,
            trust_remote_code = True
        )
        if "c4" in dataset_name:
            args.__dict__.update(
                {
                    "truncate_input_for_prompt": True,
                    "input_col_name": "text",
                    "ref_output_col_name": None,
                }
            )
            args.columns_to_remove = list(
                set(args.columns_to_remove + ["text", "timestamp", "url"])
            )
        elif "pile" in dataset_name:
            args.__dict__.update(
                {
                    "truncate_input_for_prompt": True,
                    "input_col_name": "text",
                    "ref_output_col_name": None,
                }
            )
            args.columns_to_remove = list(set(args.columns_to_remove + ["text", "meta"]))
        else:
            raise NotImplementedError(
                f"Dataset {dataset_name} not yet supported. Please add specs to load_hf_dataset function."
            )

    # add index to each row of dataset
    # indexed_dataset = dataset.map(add_idx, batched=False, with_indices=True)
    # don't add index. Accelerate

    # shuffle the first shuffle_buffer_size rows of streaming dataset, or whole dataset if not streaming
    # and take/select only the first n rows of the dataset (which caps the total number of pipeline iters possible)

    '''
    if args.DEBUG:
        print(f"name: {dataset_name}")
        print(f"iterable? {isinstance(dataset, IterableDataset)}")
    '''
    
    return dataset


def check_input_lengths(
    example,
    min_sample_len=0,
    min_prompt_len=0,
    min_completion_len=0,
    max_input_len=None,
    max_new_tokens=None,
):
    orig_sample_length = example["orig_sample_length"]
    prompt_length = example["prompt_length"]
    real_completion_length = example["baseline_completion_length"]

    if max_input_len is not None:
        assert (
            max_new_tokens is not None
        ), "need to specify max_new_tokens if max_input_length is specified"

    conds = all(
        [
            orig_sample_length >= min_sample_len,
            prompt_length >= min_prompt_len,
            real_completion_length >= min_completion_len,
            (
                ((prompt_length + max_new_tokens) <= max_input_len)
                if max_input_len is not None
                else True
            ),
        ]
    )
    return conds


def check_output_lengths(example, min_output_len=0):
    # FIXME, maybe should check baseline completion length too
    no_wm_output_len = example["no_wm_output_length"]
    w_wm_output_len = example["w_wm_output_length"]
    conds = all(
        [
            no_wm_output_len >= min_output_len,
            w_wm_output_len >= min_output_len,
        ]
    )
    return conds


def tokenize_and_truncate(
    example: dict,
    input_col_name: str = "text",
    completion_length: int = None,
    prompt_length: int = None,
    hf_model_name: str = None,
    tokenizer=None,
    truncate_left=False,
    model_max_length=None,
):
    """take hf dataset entry and preprocess it for completion by a model"""
    assert hf_model_name is not None, "need model name to know whether to adjust wrt special tokens"
    assert input_col_name in example, f"expects {input_col_name} field to be present"
    # tokenize
    inputs_ids = tokenizer(example[input_col_name], return_tensors="pt")["input_ids"]
    example.update({"untruncated_inputs": inputs_ids})

    if truncate_left:
        # truncate left
        inputs_ids = inputs_ids[:, -model_max_length:]
        if example["untruncated_inputs"].shape != inputs_ids.shape:
            print(
                "Input too long for model! ",
                "Left truncating under assumption that this is the prompt+output ",
                "to be fed to the *oracle* model",
            )
        example.update({"untruncated_inputs": inputs_ids})

    if (completion_length is not None) and (prompt_length is None):
        # leave at least one token as prefix # FIXME I think plus 1 since 0 is start tok
        slice_length = min(inputs_ids.shape[1] - 1, completion_length)
    elif (prompt_length is not None) and (completion_length is None):
        desired_comp_len = (inputs_ids.shape[1] - 1) - prompt_length
        slice_length = desired_comp_len if desired_comp_len > 0 else 0
    else:
        raise ValueError(
            (
                f"Can only tokenize and truncate based on either the desired prompt length or desired completion length,",
                f" but got completion_length:{completion_length},prompt_length:{prompt_length}",
            )
        )

    # truncate
    inputs_ids = inputs_ids[:, : inputs_ids.shape[1] - slice_length]
    # logic depending on special tokens for the model
    if "t5" in hf_model_name or "T0" in hf_model_name:
        inputs_ids[0, -1] = 1
    # else: pass
    example.update({"input_ids": inputs_ids})
    return example


def tokenize_only(
    example: dict,
    input_col_name: str = "text",
    ref_output_col_name: str = None,
    tokenize_ref_output: bool = False,
    hf_model_name: str = None,
    tokenizer=None,
    model_max_length=None,
):
    """take hf dataset entry and preprocess it for completion by a model
    (but don't truncate) where the dataset optionally has a secondary column
    that is the reference output to be scored against"""

    """take hf dataset entry and preprocess it for completion by a model"""
    assert hf_model_name is not None, "need model name to know whether to adjust wrt special tokens"
    assert input_col_name in example, f"expects {input_col_name} field to be present"
    if ref_output_col_name is not None:
        assert ref_output_col_name in example, f"expects {ref_output_col_name} field to be present"

    # tokenize input
    input_ids = tokenizer(
        example[input_col_name], return_tensors="pt", truncation=True, max_length=model_max_length
    )["input_ids"]

    example.update({"input_ids": input_ids})

    if tokenize_ref_output:
        # NOTE not sure this logic is useful/required
        if ref_output_col_name is not None:
            # tokenize ref output
            ref_output_ids = tokenizer(
                example[ref_output_col_name],
                return_tensors="pt",
                truncation=True,
                max_length=model_max_length,
            )["input_ids"]

        tokd_input_len, tokd_ref_output_length = input_ids.shape[1], ref_output_ids.shape[1]
        if tokd_input_len + tokd_ref_output_length > model_max_length:
            # truncate the ref output
            original_ref_output_len = tokd_ref_output_length
            ref_output_ids = ref_output_ids[:, : model_max_length - tokd_input_len]
            if original_ref_output_len != ref_output_ids.shape[1]:
                print(
                    "Right truncating output, input+ref output too long for model. "
                    "Note, since this is generation time truncating the reference doesn't affect anything really."
                )
        example.update({"ref_output_ids": ref_output_ids})

    # logic depending on special tokens for the model
    if "t5" in hf_model_name or "T0" in hf_model_name:
        raise NotImplementedError("T5 style model not yet supported")

    return example


def tokenize_for_generation(
    example: dict,
    max_new_tokens: int = None,
    min_prompt_tokens: int = None,
    hf_model_name: str = None,
    tokenizer: Tokenizer = None,
    args: dict = None,
):
    # preprocessing, generation & scoring
    assert isinstance(example, dict), "Expect no batch dimension currently!"

    if not args.truncate_input_for_prompt:
        tokenize_ref_output = True  # NOTE, note really sure how necessary this is
        # preprocess for model generation/completion
        example = tokenize_only(
            example,
            input_col_name=args.input_col_name,
            ref_output_col_name=args.ref_output_col_name,
            hf_model_name=hf_model_name,
            tokenizer=tokenizer,
            model_max_length=args.model_max_length,
            tokenize_ref_output=tokenize_ref_output,
        )
        # Parse the results of tokenization. Simple, since
        # the prompt and baseline completion are from the raw text
        re_decoded_input = example[args.input_col_name]
        decoded_baseline_completion = example[args.ref_output_col_name]
        prompt_len = example["input_ids"].shape[1]
        baseline_completion_len = example["ref_output_ids"].shape[1]
        full_sample_len = prompt_len + baseline_completion_len
        # for now, remove this here, since it's not used downstream
        example.pop("ref_output_ids")
    else:
        # preprocess for model generation/completion
        example = tokenize_and_truncate(
            example,
            completion_length=max_new_tokens,
            prompt_length=min_prompt_tokens,
            hf_model_name=hf_model_name,
            tokenizer=tokenizer,
        )
        # Logic to parse the results of tokenzation and splitting to
        # construct string versions of the prompt and baseline completion
        inputs = example["input_ids"]
        prompt_len = inputs.shape[1]
        # for isolating the "gold" baseline completion
        untruncated_inputs = example.pop("untruncated_inputs")
        full_sample_len = untruncated_inputs.shape[1]
        # decode the preprocessed input to store for audit
        re_decoded_input = tokenizer.batch_decode(inputs, skip_special_tokens=True)[0]
        # also decode the original suffix of the input for audit as the baseline
        baseline_completion_tokens = untruncated_inputs[:, inputs.shape[-1] :]
        decoded_baseline_completion = tokenizer.batch_decode(
            baseline_completion_tokens, skip_special_tokens=True
        )[0]
        baseline_completion_len = full_sample_len - prompt_len

    example.update(
        {
            "truncated_input": re_decoded_input,
            "baseline_completion": decoded_baseline_completion,
            "orig_sample_length": full_sample_len,
            "prompt_length": prompt_len,
            "baseline_completion_length": baseline_completion_len,
        }
    )
    return example


def tokenize_the_prompt(model_position_embed, tokenizer, data_iter, args):

    token_kwargs = dict(
        hf_model_name=args.model_name_or_path,
        tokenizer=tokenizer,
        args=args,
    )
    if args.input_truncation_strategy == "prompt_length":
        token_kwargs.update(dict(min_prompt_tokens=args.min_prompt_tokens))
    elif args.input_truncation_strategy == "completion_length":
        token_kwargs.update(dict(max_new_tokens=args.max_new_tokens))
    elif args.input_truncation_strategy == "no_truncation":
        # truncate_input_for_prompt is a bool flag, that is set by
        # the dataset loading function, semi-redundant, to make sure
        # people are very aware of which input data style they are using
        assert (
            args.truncate_input_for_prompt == False
        ), "Cannot truncate input for prompt if 'no_truncation' strategy is specified"
        pass
    else:
        ValueError(f"Unknown input truncation strategy {args.input_truncation_strategy}")
    tokenize_prompts = partial(tokenize_for_generation, **token_kwargs)

    input_check_kwargs = dict(
        min_sample_len=args.min_sample_tokens,
        max_input_len=model_position_embed,
        max_new_tokens=args.max_new_tokens,
    )
    if args.input_filtering_strategy == "prompt_length":
        input_check_kwargs.update(dict(min_prompt_len=args.min_prompt_tokens, min_completion_len=0))
    elif args.input_filtering_strategy == "completion_length":
        input_check_kwargs.update(dict(min_prompt_len=0, min_completion_len=args.max_new_tokens))
    elif args.input_filtering_strategy == "prompt_and_completion_length":
        input_check_kwargs.update(
            dict(min_prompt_len=args.min_prompt_tokens, min_completion_len=args.max_new_tokens)
        )
    elif args.input_filtering_strategy == "no_filter":
        input_check_kwargs.update(dict(min_prompt_len=0, min_completion_len=0))
    else:
        ValueError(f"Unknown input filtering strategy {args.input_filtering_strategy}")
    input_check = partial(check_input_lengths, **input_check_kwargs)

    dataset_w_prompts = data_iter.map(tokenize_prompts, batched=False)

    dataset_input_len_filtered = dataset_w_prompts.filter(input_check, batched=False)

    return dataset_input_len_filtered


def collate_batch(input_ids: list, collator: DataCollatorWithPadding = None):
    """collate batch of input_ids into a padded batch of tensors"""
    assert (
        input_ids[0].shape[0] == 1 and input_ids[0].shape[1] > 0
    ), "expecting batch dimension of each tensor to be 1"
    # remove batch dimension for each tensor
    input_ids = [x.squeeze(0) for x in input_ids]
    return collator({"input_ids": input_ids})["input_ids"]



class BufferedDataset(IterableDataset):

    def __init__(
        self,
        tokenizer,
        fullyProcessed_data_iterator,
        infinite=False,
        buffer_len=500,
        DEBUG=False,
        hidden_index=True
    ):
        
        self.DEBUG = DEBUG
        if DEBUG:
            buffer_len = 7
        self.tokenizer = tokenizer
        # self.concat_token_id = tokenizer.bos_token_id
        self.data_iterator = fullyProcessed_data_iterator
        self.buffer_len = buffer_len
        self.epoch = 0
        self.infinite = infinite
        self.current_size = 0
        self.GenRecord = []
        self.hidden_index = hidden_index
        if hidden_index:
            self.index_allocation = 0
            self.reclaim_indicator = []

        self.output_record = []

        


    def __iter__(self):
        iterator = iter(self.data_iterator)
        more_examples = True
        while more_examples:
            buffer, buffer_length = [], 0
            while True:
                if buffer_length >= self.buffer_len:
                    break
                try:
                    next_out = next(iterator)
                    if self.hidden_index:
                        
                        next_out["hidden_index"] = self.index_allocation
                        self.index_allocation += 1
                        
                    # buffer.append(next_out["input_ids"].squeeze(0))
                    buffer.append(next_out)

                    if self.DEBUG:
                        storable = dict_remove(next_out, "input_ids")
                        self.GenRecord.append(storable)

                    buffer_length += 1
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.data_iterator)
                        self.epoch += 1
                    else:
                        more_examples = False
                        break
            
            for i in range(0, len(buffer)):

                output_item = buffer[i]

                
                self.output_record.append(dict_remove(output_item, "input_ids"))
                # print(f"output length: {len(self.output_record)}")

                if self.hidden_index:
                    assert output_item["hidden_index"] == len(self.output_record) - 1

                    self.reclaim_indicator.append(0)

                    # secretly attach an index key ahead

                    body = output_item["input_ids"].squeeze(0)
                    type_preserve = body.dtype
                    yield torch.cat((torch.Tensor([output_item["hidden_index"]]), body)).type(type_preserve)

                else:
                    yield output_item["input_ids"].squeeze(0)

    def reclaim(self, dic_list):

        assert self.hidden_index

        for dic in dic_list:

            index = dic["hidden_index"]

            assert self.output_record[index]["hidden_index"] == index

            self.output_record[index].update(dic)

            self.reclaim_indicator[index] = 1

    def retrieve_record(self):

        return self.output_record


"""
def shuffle(self, buffer_size=1000):
    return ShufflerIterDataPipe(self, buffer_size=buffer_size)


"""



class collateBuffer(IterableDataset):

    def __init__(
            self,
            iterable_dataset,
            pad_id,
            padding_mask_output=True,
            collate_batchsize = 4
    ):

        self.iterable_dataset = iterable_dataset
        self.collate_batchsize = collate_batchsize
        self.buffer = []
        self.collated_flag = False
        if iterable_dataset.hidden_index:
            self.hidden_index = True
        else:
            self.hidden_index = False

        self.pad_id = pad_id
        self.padding_mask_output = padding_mask_output

        # self.pad_id = model.config.pad_token_id
        # self.eos_id = model.config.eos_token_id

    def __iter__(self):

        iter_data = iter(self.iterable_dataset)
        while True:
            try:
                next_sample = next(iter_data)

            except StopIteration:
                break

            self.buffer.append(next_sample)

            if len(self.buffer) >= self.collate_batchsize:

                # start collating
                self.collated_flag = True

                if self.hidden_index:
                    new_buffer = []
                    index_collect = []
                    for item in self.buffer:
                        new_buffer.append(item[1:])
                        index_collect.append(item[0])

                else:
                    new_buffer = self.buffer
                    
                    
                min_prompt_size = min([len(t) for t in new_buffer])
                max_prompt_size = max([len(t) for t in new_buffer])

                padding_bed = torch.full((len(new_buffer), max_prompt_size), self.pad_id).long()
                mask_bed = torch.zeros((len(new_buffer), max_prompt_size)).long()


                for row in range(len(new_buffer)):
                    token_input = new_buffer[row]
                    padding_bed[row][:len(token_input)] = token_input
                    mask_bed[row][:len(token_input)] = 1


                  
                if self.padding_mask_output:
                    if self.hidden_index:
                        collated_list = torch.cat((torch.Tensor(index_collect).view(-1,1), padding_bed, mask_bed),1).long()
                    else:
                        collated_list = torch.cat((padding_bed, mask_bed),1).long()
                else:
                    if self.hidden_index:
                        collated_list = torch.cat((torch.Tensor(index_collect).view(-1,1), padding_bed),1).long()
                    else:
                        collated_list = padding_bed.long()
                


                
                for j in range(len(collated_list)):

                    yield collated_list[j]

                self.collated_flag = False
                self.buffer = []

    def reclaim(self, dic_list):

        self.iterable_dataset.reclaim(dic_list)

        return

    def retrieve_record(self):

        return self.iterable_dataset.retrieve_record()




# this dataset requires that, everytime
class reclaimDataset(IterableDataset):

    def __init__(
            self, iterable_dataset, reclaim_batsize
    ):
        
        
        self.iterable_dataset = iterable_dataset

        assert iterable_dataset.hidden_index

        # assume iterable_dataset is a collateBuffer
        self.reclaim_hold = reclaim_batsize

        self.reclaim_owing = 0

    def __iter__(
            self
    ):

        iter_data = iter(self.iterable_dataset)
        while True:

            # will not produce new sample if previous outputs has not been claimed
            if self.reclaim_owing < 0 or self.reclaim_owing >= self.reclaim_hold:

                print(f"owing: {self.reclaim_owing}")

                raise ValueError
            
            else:

                try:
                    next_sample = next(iter_data)

                except StopIteration:
                    break

                self.reclaim_owing += 1

                yield next_sample


    def reclaim(self, dic_list):

        # take as input a list of dictionaries

        self.iterable_dataset.reclaim(dic_list)

        self.reclaim_owing -= len(dic_list)

        return

    def retrieve_record(self):

        return self.iterable_dataset.retrieve_record()


def construct_reclaimable_dataset(tokenizer, data_iter, pad_id, batchsize, reclaim_batsize = 0):

    if reclaim_batsize <= 0:
        reclaim_batsize = batchsize

    train_dataset = BufferedDataset(
            tokenizer,
            data_iter ,
            infinite=True,
            DEBUG=False
        )
        
        # train_dataset = train_dataset.shuffle(buffer_size=10000)

    collate_dataset = collateBuffer(
        train_dataset,
        pad_id = pad_id,
        collate_batchsize=batchsize
    )

    reclaimSet = reclaimDataset(
        collate_dataset,
        reclaim_batsize= reclaim_batsize
    )
    
    return reclaimSet

