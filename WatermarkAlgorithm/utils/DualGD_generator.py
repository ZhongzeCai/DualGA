# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import numpy as np
import copy

def denom_func(delta, g):

    return g * torch.exp(delta) - g + 1

def KLt(delta, g):

    return (g * delta * torch.exp(delta)/denom_func(delta, g) - torch.log(denom_func(delta, g)))

def DGt(delta, g):

    return (1-g)*(1-1/denom_func(delta, g))

def obj_func(delta, lamb, g):

    return DGt(delta, g) - \
        lamb * KLt(delta, g)


# the generator can not take as input the model
# as its properties will not be accessible in DDP setting
class DualGDGenerator():
    def __init__(self, 
            tokenizer: LlamaTokenizer, 
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317,
            payload: int = 0,
            eta: float = 0.1,
            autoeta: int = 1,
            gamma: float = 0.5,
            b: float = 2.0,
            h_func: str = "2x2",
            init_lambda: str = 0.3,
            model_args = None,
            **kwargs
        ):
        # model config
        self.tokenizer = tokenizer
        
        self.max_seq_len = model_args["max_sequence_length"]
        self.pad_id = model_args["pad_token_id"]
        self.eos_id = model_args["eos_token_id"]
        self.is_decoder_only_model = model_args["is_decoder_only_model"]
        # watermark config
        self.ngram = ngram
        self.salt_key = salt_key
        self.seed = seed
        self.hashtable = torch.randperm(1000003)
        self.seeding = seeding 
        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)
        self.payload = payload
        self.eta = eta
        self.autoeta = autoeta
        self.gamma = gamma
        self.b = b
        self.h_func = h_func
        self.init_lambda = init_lambda
        

    def hashint(self, integer_tensor: torch.LongTensor) -> torch.LongTensor:
        """Adapted from https://github.com/jwkirchenbauer/lm-watermarking"""
        return self.hashtable[integer_tensor.cpu() % len(self.hashtable)] 
    
    def get_seed_rng(
        self, 
        input_ids: torch.LongTensor
    ) -> int:
        """
        Seed RNG with hash of input_ids.
        Adapted from https://github.com/jwkirchenbauer/lm-watermarking
        """
        if self.seeding == 'hash':
            seed = self.seed
            for i in input_ids:
                seed = (seed * self.salt_key + i.item()) % (2 ** 64 - 1)
        elif self.seeding == 'additive':
            seed = self.salt_key * torch.sum(input_ids).item()
            seed = self.hashint(seed)
        elif self.seeding == 'skip':
            seed = self.salt_key * input_ids[0].item()
            seed = self.hashint(seed)
        elif self.seeding == 'min':
            seed = self.hashint(self.salt_key * input_ids)
            seed = torch.min(seed).item()
        return seed


    @torch.no_grad()
    def generate_metric(
        self,
        model,
        index_list,
        prompt_batch,
        mask_batch,
        max_gen_len: int,
        temperature: float = 1.,
        top_p: float = 0.95,
        label = "generated_str"
    ) -> List[str]:
        """
        Generate text from prompts. 
        Adapted from https://github.com/facebookresearch/llama/
        """
        
        assert prompt_batch.shape == mask_batch.shape
        assert torch.min(mask_batch).item() < 1e-3 and torch.max(mask_batch).item() > 1-1e-3

        dic_list = []

        bsz = len(prompt_batch)

        lens = mask_batch.sum(axis=1)
    
        min_prompt_size = torch.min(lens).item()
        max_prompt_size = prompt_batch.shape[1]

        
        total_len = min(self.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.pad_id).to(model.device).long()
        input_text_mask = torch.zeros((bsz, total_len)).to(model.device).long()

        for k, t in enumerate(prompt_batch):
            tokens[k, : len(t)] = t.long().clone().detach()

        input_text_mask[:mask_batch.shape[0], : mask_batch.shape[1]] = mask_batch
        input_text_mask = input_text_mask.bool()

        start_pos = min_prompt_size
        prev_pos = 0

        entropy_list = np.zeros((bsz, total_len))
        loglike_list = np.zeros((bsz, total_len))
        ppllike_list = np.zeros((bsz, total_len))
        green_probs = np.zeros((bsz, total_len))
        green_probs_increases = np.zeros((bsz, total_len))
        conditional_kl = np.zeros((bsz, total_len))

        budget = self.b * max_gen_len

        ###################################################
        lambda_record = self.init_lambda * np.ones((bsz, total_len))
        budget_record = budget * np.ones((bsz, total_len))
        delta_record = np.zeros((bsz, total_len))
        ###################################################

        for cur_pos in range(start_pos, total_len):
            outputs = model.forward(
                tokens[:, prev_pos:cur_pos], use_cache=True, past_key_values=outputs.past_key_values if prev_pos > 0 else None
            )

            ngram_tokens = tokens[:, cur_pos-self.ngram:cur_pos]

            current_lambda = lambda_record[:, cur_pos]
            current_budget = budget_record[:, cur_pos]
            
            next_toks, entropy, loglike, ppllike, GreenP, GreenP_increase, kl, lamb_next, budget_next, delta = \
                self.sample_next(outputs.logits[:, -1, :], ngram_tokens, temperature, top_p, \
                                 current_lambda, current_budget, metric = True, cur_pos = cur_pos)

            tokens[:, cur_pos] = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_toks)
            
            np_mask = input_text_mask[:, cur_pos].detach().cpu().numpy()


            entropy_list[:, cur_pos] = np.where(np_mask, 0, entropy)
            loglike_list[:, cur_pos] = np.where(np_mask, 0, loglike)
            ppllike_list[:, cur_pos] = np.where(np_mask, 0, ppllike)
            green_probs[:, cur_pos] = np.where(np_mask, 0, GreenP)
            green_probs_increases[:, cur_pos] = np.where(np_mask, 0, GreenP_increase)
            conditional_kl[:, cur_pos] = np.where(np_mask, 0, kl)
            delta_record[:, cur_pos] = np.where(np_mask, 0, delta)

            if cur_pos < total_len -1:

                np_mask_next = input_text_mask[:, cur_pos+1].detach().cpu().numpy()
                lambda_record[:, cur_pos+1] = np.where(np_mask_next, lambda_record[:, cur_pos], lamb_next)
                budget_record[:, cur_pos+1] = np.where(np_mask_next, budget_record[:, cur_pos], budget_next)
            

            prev_pos = cur_pos



        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: lens[i].item() + max_gen_len]
            entropy_i = entropy_list[i][: lens[i].item() + max_gen_len]
            loglike_i = loglike_list[i][: lens[i].item() + max_gen_len]
            ppllike_i = ppllike_list[i][: lens[i].item() + max_gen_len]
            greep_i = green_probs[i][: lens[i].item() + max_gen_len]
            greep_increase_i = green_probs_increases[i][: lens[i].item() + max_gen_len]
            ckl_i = conditional_kl[i][: lens[i].item() + max_gen_len]
            lambda_i = lambda_record[i][: lens[i].item() + max_gen_len]
            budget_i = budget_record[i][: lens[i].item() + max_gen_len]
            delta_i = delta_record[i][: lens[i].item() + max_gen_len]

            # cut to eos tok if any
            try:

                entropy_i = entropy_i[: t.index(self.eos_id)]
                loglike_i = loglike_i[: t.index(self.eos_id)]
                ppllike_i = ppllike_i[: t.index(self.eos_id)]
                greep_i = greep_i[: t.index(self.eos_id)]
                greep_increase_i = greep_increase_i[: t.index(self.eos_id)]
                ckl_i = ckl_i[: t.index(self.eos_id)]
                lambda_i = lambda_i[: t.index(self.eos_id)]
                budget_i = budget_i[: t.index(self.eos_id)]
                delta_i = delta_i[: t.index(self.eos_id)]


                t = t[: t.index(self.eos_id)]

            except ValueError:
                pass

            if self.is_decoder_only_model:

                entropy_i = entropy_i[lens[i].item() : ]
                loglike_i = loglike_i[lens[i].item() : ]
                ppllike_i = ppllike_i[lens[i].item() : ]
                greep_i = greep_i[lens[i].item() : ]
                greep_increase_i = greep_increase_i[lens[i].item() : ]
                ckl_i = ckl_i[lens[i].item() : ]
                lambda_i = lambda_i[lens[i].item() : ]
                budget_i = budget_i[lens[i].item() : ]
                delta_i = delta_i[lens[i].item() : ]

            
                t = t[lens[i].item() : ]

            output_str = self.tokenizer.decode(t)

            dic_list.append({
                "hidden_index": index_list[i].item(),
                label: output_str,
                "output_token_num": len(t),
                "entropy_list": entropy_i.tolist(),
                "loglike_list": loglike_i.tolist(),
                "ppllike_list": ppllike_i.tolist(),
                "green_probs": greep_i.tolist(),
                "green_probs_increases": greep_increase_i.tolist(),
                "conditional_kl": ckl_i.tolist(),
                "dual_lambda": lambda_i.tolist(),
                "kl_budget": budget_i.tolist(),
                "adaptive_delta": delta_i.tolist()
            })
        
        return dic_list

    
    def sample_next(
        self,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        ngram_tokens: torch.LongTensor, # (bsz, ngram): tokens to consider when seeding
        temperature: float = 1., # temperature for sampling
        top_p: float = 0.95, # top p for sampling
        current_lambda = None, 
        current_budget = None,
        metric = False,
        cur_pos = None
    ) -> torch.LongTensor:
        
        original_logits = logits

        bsz, vocab_size = logits.shape

        logits = logits.clone()

        # delta_base_table = torch.Tensor(np.arange(0, 10, 0.1)).to(logits.device)

        ################# prepare ##################

        current_lambda = torch.Tensor(current_lambda).to(logits.device)
        current_budget = torch.Tensor(current_budget).to(logits.device)

        # lambda_chose_chocolate = lambda_chose.view(-1, 1).repeat(1, len(delta_base_table))

        # delta_base_chocolate = delta_base_table.view(1, -1).repeat(bsz, 1)

        ## get green list indexes

        green_indicator = torch.zeros((bsz, vocab_size)).to(logits.device)

        for ii in range(ngram_tokens.shape[0]): # batch of texts
            seed = self.get_seed_rng(ngram_tokens[ii])
            self.rng.manual_seed(seed)
            vocab_permutation = torch.randperm(vocab_size, generator=self.rng)
            greenlist = vocab_permutation[:int(self.gamma * vocab_size)] # gamma * n

            green_indicator[ii][greenlist] = 1

        ## calculate g and delta
            
        prob = torch.softmax(original_logits, dim=-1)

        g = (prob * green_indicator).sum(dim=-1)


        # g_chocolate = g.view(-1, 1).repeat(1, len(delta_base_table))

        # choose_argmax_delta_chocolate = obj_func(delta_base_chocolate, lambda_chose_chocolate, g_chocolate)

        # argmax_index = torch.argmax(choose_argmax_delta_chocolate, dim = -1)

        # delta_tilde = torch.gather(delta_base_table, 0, index=argmax_index)

        delta_tilde = torch.clip(1 / current_lambda, min=0, max=10)

        delta_harvest = copy.deepcopy(delta_tilde)

        ## use remaining budget to mask out deltas

        delta_harvest[torch.where(current_budget <= 1e-3)[0]] = 0

        # with this delta, we can modify logits

        bias_placement = delta_harvest.view(-1, 1) * green_indicator

        logits += bias_placement

        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            next_token = torch.multinomial(probs_sort, num_samples=1) # one hot of next token, ordered by original probs
            next_token = torch.gather(probs_idx, -1, next_token) # one hot of next token, ordered by vocab
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)

        if metric:

            entropy, loglike, ppllike, GreenP, GreenP_increase, kl = self.retrieve_metric(original_logits, logits, next_token)

            ## budget update

            new_budget = torch.clamp(current_budget - torch.Tensor(kl).to(logits.device), min=0, max=1e4)

            # update lambda

            dgt = - KLt(delta_tilde, g) + self.b

            if self.eta > 1e-4:
                use_eta = self.eta
            else:
                use_eta = self.autoeta / np.sqrt(cur_pos)

            if self.h_func == "2x2":

                new_lam = torch.clamp(current_lambda - use_eta * dgt, min=0, max=1e4)

            elif self.h_func == "xlnx":

                new_lam = current_lambda * torch.exp( - use_eta * dgt)

            else:

                raise NotImplementedError


            return next_token, entropy, loglike, ppllike, GreenP, GreenP_increase, kl \
                , new_lam.cpu().numpy(), new_budget.cpu().numpy(), delta_harvest.cpu().numpy()

        else:
            return next_token






    def retrieve_metric(self, original_logits, logits, next_token):

        assert torch.all(logits >= original_logits)

        green_mask = (logits > original_logits + 1e-3)

        original_prob = torch.softmax(original_logits, dim=-1)
        prob = torch.softmax(logits, dim=-1)

        ppllike = (prob * torch.log(original_prob)).sum(dim=-1).detach().cpu().numpy()

        entropy = (original_prob * torch.log(original_prob)).sum(dim=-1).detach().cpu().numpy()

        GreenP = (original_prob * green_mask).sum(dim=-1).detach().cpu().numpy()

        GreenP_increase = (prob * green_mask).sum(dim=-1).detach().cpu().numpy() - GreenP

        kl = (prob * (torch.log(prob) - torch.log(original_prob))).sum(dim=-1).detach().cpu().numpy()

        loglike = torch.log(torch.gather(original_prob, -1, next_token.reshape(-1, 1))).reshape(-1).detach().cpu().numpy()

        return entropy, loglike, ppllike, GreenP, GreenP_increase, kl
    



class DualGDGenerator_KL():
    def __init__(self, 
            tokenizer: LlamaTokenizer, 
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317,
            payload: int = 0,
            eta: float = 0.1,
            autoeta: int = 1,
            gamma: float = 0.5,
            D: float = 0.2,
            h_func: str = "2x2",
            init_lambda_KL: str = 3,
            model_args = None,
            **kwargs
        ):
        # model config
        self.tokenizer = tokenizer
        
        self.max_seq_len = model_args["max_sequence_length"]
        self.pad_id = model_args["pad_token_id"]
        self.eos_id = model_args["eos_token_id"]
        self.is_decoder_only_model = model_args["is_decoder_only_model"]
        # watermark config
        self.ngram = ngram
        self.salt_key = salt_key
        self.seed = seed
        self.hashtable = torch.randperm(1000003)
        self.seeding = seeding 
        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)
        self.payload = payload
        self.eta = eta
        self.autoeta = autoeta
        self.gamma = gamma
        self.D = D
        self.h_func = h_func
        self.init_lambda_KL = init_lambda_KL
        

    def hashint(self, integer_tensor: torch.LongTensor) -> torch.LongTensor:
        """Adapted from https://github.com/jwkirchenbauer/lm-watermarking"""
        return self.hashtable[integer_tensor.cpu() % len(self.hashtable)] 
    
    def get_seed_rng(
        self, 
        input_ids: torch.LongTensor
    ) -> int:
        """
        Seed RNG with hash of input_ids.
        Adapted from https://github.com/jwkirchenbauer/lm-watermarking
        """
        if self.seeding == 'hash':
            seed = self.seed
            for i in input_ids:
                seed = (seed * self.salt_key + i.item()) % (2 ** 64 - 1)
        elif self.seeding == 'additive':
            seed = self.salt_key * torch.sum(input_ids).item()
            seed = self.hashint(seed)
        elif self.seeding == 'skip':
            seed = self.salt_key * input_ids[0].item()
            seed = self.hashint(seed)
        elif self.seeding == 'min':
            seed = self.hashint(self.salt_key * input_ids)
            seed = torch.min(seed).item()
        return seed


    @torch.no_grad()
    def generate_metric(
        self,
        model,
        index_list,
        prompt_batch,
        mask_batch,
        max_gen_len: int,
        temperature: float = 1.,
        top_p: float = 0.95,
        label = "generated_str"
    ) -> List[str]:
        """
        Generate text from prompts. 
        Adapted from https://github.com/facebookresearch/llama/
        """
        
        assert prompt_batch.shape == mask_batch.shape
        assert torch.min(mask_batch).item() < 1e-3 and torch.max(mask_batch).item() > 1-1e-3

        dic_list = []

        bsz = len(prompt_batch)

        lens = mask_batch.sum(axis=1)
    
        min_prompt_size = torch.min(lens).item()
        max_prompt_size = prompt_batch.shape[1]

        
        total_len = min(self.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.pad_id).to(model.device).long()
        input_text_mask = torch.zeros((bsz, total_len)).to(model.device).long()

        for k, t in enumerate(prompt_batch):
            tokens[k, : len(t)] = t.long().clone().detach()

        input_text_mask[:mask_batch.shape[0], : mask_batch.shape[1]] = mask_batch
        input_text_mask = input_text_mask.bool()

        start_pos = min_prompt_size
        prev_pos = 0

        entropy_list = np.zeros((bsz, total_len))
        loglike_list = np.zeros((bsz, total_len))
        ppllike_list = np.zeros((bsz, total_len))
        green_probs = np.zeros((bsz, total_len))
        green_probs_increases = np.zeros((bsz, total_len))
        conditional_kl = np.zeros((bsz, total_len))

        ###################################################
        lambda_record = self.init_lambda_KL * np.ones((bsz, total_len))
        
        delta_record = np.zeros((bsz, total_len))
        ###################################################

        for cur_pos in range(start_pos, total_len):
            outputs = model.forward(
                tokens[:, prev_pos:cur_pos], use_cache=True, past_key_values=outputs.past_key_values if prev_pos > 0 else None
            )

            ngram_tokens = tokens[:, cur_pos-self.ngram:cur_pos]

            current_lambda = lambda_record[:, cur_pos]
            
            next_toks, entropy, loglike, ppllike, GreenP, GreenP_increase, kl, lamb_next, delta = \
                self.sample_next(outputs.logits[:, -1, :], ngram_tokens, temperature, top_p, \
                                 current_lambda, metric = True, cur_pos = cur_pos)

            tokens[:, cur_pos] = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_toks)
            
            np_mask = input_text_mask[:, cur_pos].detach().cpu().numpy()


            entropy_list[:, cur_pos] = np.where(np_mask, 0, entropy)
            loglike_list[:, cur_pos] = np.where(np_mask, 0, loglike)
            ppllike_list[:, cur_pos] = np.where(np_mask, 0, ppllike)
            green_probs[:, cur_pos] = np.where(np_mask, 0, GreenP)
            green_probs_increases[:, cur_pos] = np.where(np_mask, 0, GreenP_increase)
            conditional_kl[:, cur_pos] = np.where(np_mask, 0, kl)
            delta_record[:, cur_pos] = np.where(np_mask, 0, delta)

            if cur_pos < total_len -1:

                np_mask_next = input_text_mask[:, cur_pos+1].detach().cpu().numpy()
                lambda_record[:, cur_pos+1] = np.where(np_mask_next, lambda_record[:, cur_pos], lamb_next)

            prev_pos = cur_pos



        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: lens[i].item() + max_gen_len]
            entropy_i = entropy_list[i][: lens[i].item() + max_gen_len]
            loglike_i = loglike_list[i][: lens[i].item() + max_gen_len]
            ppllike_i = ppllike_list[i][: lens[i].item() + max_gen_len]
            greep_i = green_probs[i][: lens[i].item() + max_gen_len]
            greep_increase_i = green_probs_increases[i][: lens[i].item() + max_gen_len]
            ckl_i = conditional_kl[i][: lens[i].item() + max_gen_len]
            lambda_i = lambda_record[i][: lens[i].item() + max_gen_len]
            delta_i = delta_record[i][: lens[i].item() + max_gen_len]

            # cut to eos tok if any
            try:

                entropy_i = entropy_i[: t.index(self.eos_id)]
                loglike_i = loglike_i[: t.index(self.eos_id)]
                ppllike_i = ppllike_i[: t.index(self.eos_id)]
                greep_i = greep_i[: t.index(self.eos_id)]
                greep_increase_i = greep_increase_i[: t.index(self.eos_id)]
                ckl_i = ckl_i[: t.index(self.eos_id)]
                lambda_i = lambda_i[: t.index(self.eos_id)]
                delta_i = delta_i[: t.index(self.eos_id)]


                t = t[: t.index(self.eos_id)]

            except ValueError:
                pass

            if self.is_decoder_only_model:

                entropy_i = entropy_i[lens[i].item() : ]
                loglike_i = loglike_i[lens[i].item() : ]
                ppllike_i = ppllike_i[lens[i].item() : ]
                greep_i = greep_i[lens[i].item() : ]
                greep_increase_i = greep_increase_i[lens[i].item() : ]
                ckl_i = ckl_i[lens[i].item() : ]
                lambda_i = lambda_i[lens[i].item() : ]
                delta_i = delta_i[lens[i].item() : ]

            
                t = t[lens[i].item() : ]

            output_str = self.tokenizer.decode(t)

            dic_list.append({
                "hidden_index": index_list[i].item(),
                label: output_str,
                "output_token_num": len(t),
                "entropy_list": entropy_i.tolist(),
                "loglike_list": loglike_i.tolist(),
                "ppllike_list": ppllike_i.tolist(),
                "green_probs": greep_i.tolist(),
                "green_probs_increases": greep_increase_i.tolist(),
                "conditional_kl": ckl_i.tolist(),
                "dual_lambda": lambda_i.tolist(),
                "adaptive_delta": delta_i.tolist()
            })
        
        return dic_list

    
    def sample_next(
        self,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        ngram_tokens: torch.LongTensor, # (bsz, ngram): tokens to consider when seeding
        temperature: float = 1., # temperature for sampling
        top_p: float = 0.95, # top p for sampling
        current_lambda = None, 
        metric = False,
        cur_pos = None
    ) -> torch.LongTensor:
        
        original_logits = logits

        bsz, vocab_size = logits.shape

        logits = logits.clone()

        # delta_base_table = torch.Tensor(np.arange(0, 10, 0.1)).to(logits.device)

        ################# prepare ##################

        current_lambda = torch.Tensor(current_lambda).to(logits.device)
    
        # lambda_chose_chocolate = lambda_chose.view(-1, 1).repeat(1, len(delta_base_table))

        # delta_base_chocolate = delta_base_table.view(1, -1).repeat(bsz, 1)

        ## get green list indexes

        green_indicator = torch.zeros((bsz, vocab_size)).to(logits.device)

        for ii in range(ngram_tokens.shape[0]): # batch of texts
            seed = self.get_seed_rng(ngram_tokens[ii])
            self.rng.manual_seed(seed)
            vocab_permutation = torch.randperm(vocab_size, generator=self.rng)
            greenlist = vocab_permutation[:int(self.gamma * vocab_size)] # gamma * n

            green_indicator[ii][greenlist] = 1

        ## calculate g and delta
            
        prob = torch.softmax(original_logits, dim=-1)

        g = (prob * green_indicator).sum(dim=-1)


        # g_chocolate = g.view(-1, 1).repeat(1, len(delta_base_table))

        # choose_argmax_delta_chocolate = obj_func(delta_base_chocolate, lambda_chose_chocolate, g_chocolate)

        # argmax_index = torch.argmax(choose_argmax_delta_chocolate, dim = -1)

        # delta_tilde = torch.gather(delta_base_table, 0, index=argmax_index)

        delta_tilde = torch.clip(current_lambda, min=0, max=15)

        delta_harvest = copy.deepcopy(delta_tilde)

        ## use remaining budget to mask out deltas

        # with this delta, we can modify logits

        bias_placement = delta_harvest.view(-1, 1) * green_indicator

        logits += bias_placement

        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            next_token = torch.multinomial(probs_sort, num_samples=1) # one hot of next token, ordered by original probs
            next_token = torch.gather(probs_idx, -1, next_token) # one hot of next token, ordered by vocab
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)

        if metric:

            entropy, loglike, ppllike, GreenP, GreenP_increase, kl = self.retrieve_metric(original_logits, logits, next_token)


            # update lambda

            dgt = - DGt(delta_tilde, g) + self.D


            if self.eta > 1e-4:
                use_eta = self.eta
            else:
                use_eta = self.autoeta / np.sqrt(cur_pos)


            if self.h_func == "2x2":

                new_lam = torch.clamp(current_lambda + use_eta * dgt, min=0, max=15)

            elif self.h_func == "xlnx":

                raise NotImplementedError

            else:

                raise NotImplementedError


            return next_token, entropy, loglike, ppllike, GreenP, GreenP_increase, kl \
                , new_lam.cpu().numpy(), delta_harvest.cpu().numpy()

        else:
            return next_token






    def retrieve_metric(self, original_logits, logits, next_token):

        assert torch.all(logits >= original_logits)

        green_mask = (logits > original_logits + 1e-3)

        original_prob = torch.softmax(original_logits, dim=-1)
        prob = torch.softmax(logits, dim=-1)

        ppllike = (prob * torch.log(original_prob)).sum(dim=-1).detach().cpu().numpy()

        entropy = (original_prob * torch.log(original_prob)).sum(dim=-1).detach().cpu().numpy()

        GreenP = (original_prob * green_mask).sum(dim=-1).detach().cpu().numpy()

        GreenP_increase = (prob * green_mask).sum(dim=-1).detach().cpu().numpy() - GreenP

        kl = (prob * (torch.log(prob) - torch.log(original_prob))).sum(dim=-1).detach().cpu().numpy()

        loglike = torch.log(torch.gather(original_prob, -1, next_token.reshape(-1, 1))).reshape(-1).detach().cpu().numpy()

        return entropy, loglike, ppllike, GreenP, GreenP_increase, kl
    

# as its properties will not be accessible in DDP setting
class MinACFGenerator():
    def __init__(self, 
            tokenizer: LlamaTokenizer, 
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317,
            payload: int = 0,
            eta: float = 5.,
            autoeta: int = 1,
            gamma: float = 0.5,
            b: float = 2.0,
            h_func: str = "2x2",
            init_lambda: str = 0.2,
            model_args = None,
            **kwargs
        ):
        # model config
        self.tokenizer = tokenizer
        
        self.max_seq_len = model_args["max_sequence_length"]
        self.pad_id = model_args["pad_token_id"]
        self.eos_id = model_args["eos_token_id"]
        self.is_decoder_only_model = model_args["is_decoder_only_model"]
        # watermark config
        self.ngram = ngram
        self.salt_key = salt_key
        self.seed = seed
        self.hashtable = torch.randperm(1000003)
        self.seeding = seeding 
        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)
        self.payload = payload
        self.eta = eta
        self.autoeta = autoeta
        self.gamma = gamma
        self.b = b
        self.h_func = h_func
        self.init_lambda = init_lambda
        

    def hashint(self, integer_tensor: torch.LongTensor) -> torch.LongTensor:
        """Adapted from https://github.com/jwkirchenbauer/lm-watermarking"""
        return self.hashtable[integer_tensor.cpu() % len(self.hashtable)] 
    
    def get_seed_rng(
        self, 
        input_ids: torch.LongTensor
    ) -> int:
        """
        Seed RNG with hash of input_ids.
        Adapted from https://github.com/jwkirchenbauer/lm-watermarking
        """
        if self.seeding == 'hash':
            seed = self.seed
            for i in input_ids:
                seed = (seed * self.salt_key + i.item()) % (2 ** 64 - 1)
        elif self.seeding == 'additive':
            seed = self.salt_key * torch.sum(input_ids).item()
            seed = self.hashint(seed)
        elif self.seeding == 'skip':
            seed = self.salt_key * input_ids[0].item()
            seed = self.hashint(seed)
        elif self.seeding == 'min':
            seed = self.hashint(self.salt_key * input_ids)
            seed = torch.min(seed).item()
        return seed


    @torch.no_grad()
    def generate_metric(
        self,
        model,
        index_list,
        prompt_batch,
        mask_batch,
        max_gen_len: int,
        temperature: float = 1.,
        top_p: float = 0.95,
        label = "generated_str"
    ) -> List[str]:
        """
        Generate text from prompts. 
        Adapted from https://github.com/facebookresearch/llama/
        """
        
        assert prompt_batch.shape == mask_batch.shape
        assert torch.min(mask_batch).item() < 1e-3 and torch.max(mask_batch).item() > 1-1e-3

        dic_list = []

        bsz = len(prompt_batch)

        lens = mask_batch.sum(axis=1)
    
        min_prompt_size = torch.min(lens).item()
        max_prompt_size = prompt_batch.shape[1]

        
        total_len = min(self.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.pad_id).to(model.device).long()
        input_text_mask = torch.zeros((bsz, total_len)).to(model.device).long()

        for k, t in enumerate(prompt_batch):
            tokens[k, : len(t)] = t.long().clone().detach()

        input_text_mask[:mask_batch.shape[0], : mask_batch.shape[1]] = mask_batch
        input_text_mask = input_text_mask.bool()

        start_pos = min_prompt_size
        prev_pos = 0

        entropy_list = np.zeros((bsz, total_len))
        loglike_list = np.zeros((bsz, total_len))
        ppllike_list = np.zeros((bsz, total_len))
        green_probs = np.zeros((bsz, total_len))
        green_probs_increases = np.zeros((bsz, total_len))
        conditional_kl = np.zeros((bsz, total_len))

        budget = self.b * max_gen_len

        ###################################################
        lambda_record = self.init_lambda * np.ones((bsz, total_len))
        budget_record = budget * np.ones((bsz, total_len))
        delta_record = np.zeros((bsz, total_len))
        ###################################################

        # keep track of the color of previous token
        prev_color_list = ['r'] * bsz

        for cur_pos in range(start_pos, total_len):
            outputs = model.forward(
                tokens[:, prev_pos:cur_pos], use_cache=True, past_key_values=outputs.past_key_values if prev_pos > 0 else None
            )

            ngram_tokens = tokens[:, cur_pos-self.ngram:cur_pos]

            current_lambda = lambda_record[:, cur_pos]
            current_budget = budget_record[:, cur_pos]
            
            next_toks, entropy, loglike, ppllike, addi_P, addi_P_increase, kl, lamb_next, budget_next, delta, cur_color_list = \
                self.sample_next(outputs.logits[:, -1, :], ngram_tokens, temperature, top_p, \
                                 current_lambda, current_budget, metric = True, cur_pos = cur_pos, prev_color_list = prev_color_list)
            
            prev_color_list = cur_color_list

            tokens[:, cur_pos] = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_toks)
            
            np_mask = input_text_mask[:, cur_pos].detach().cpu().numpy()


            entropy_list[:, cur_pos] = np.where(np_mask, 0, entropy)
            loglike_list[:, cur_pos] = np.where(np_mask, 0, loglike)
            ppllike_list[:, cur_pos] = np.where(np_mask, 0, ppllike)
            green_probs[:, cur_pos] = np.where(np_mask, 0, addi_P)
            green_probs_increases[:, cur_pos] = np.where(np_mask, 0, addi_P_increase)
            conditional_kl[:, cur_pos] = np.where(np_mask, 0, kl)
            delta_record[:, cur_pos] = np.where(np_mask, 0, delta)

            if cur_pos < total_len -1:

                np_mask_next = input_text_mask[:, cur_pos+1].detach().cpu().numpy()
                lambda_record[:, cur_pos+1] = np.where(np_mask_next, lambda_record[:, cur_pos], lamb_next)
                budget_record[:, cur_pos+1] = np.where(np_mask_next, budget_record[:, cur_pos], budget_next)
            

            prev_pos = cur_pos



        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: lens[i].item() + max_gen_len]
            entropy_i = entropy_list[i][: lens[i].item() + max_gen_len]
            loglike_i = loglike_list[i][: lens[i].item() + max_gen_len]
            ppllike_i = ppllike_list[i][: lens[i].item() + max_gen_len]
            greep_i = green_probs[i][: lens[i].item() + max_gen_len]
            greep_increase_i = green_probs_increases[i][: lens[i].item() + max_gen_len]
            ckl_i = conditional_kl[i][: lens[i].item() + max_gen_len]
            lambda_i = lambda_record[i][: lens[i].item() + max_gen_len]
            budget_i = budget_record[i][: lens[i].item() + max_gen_len]
            delta_i = delta_record[i][: lens[i].item() + max_gen_len]

            # cut to eos tok if any
            try:

                entropy_i = entropy_i[: t.index(self.eos_id)]
                loglike_i = loglike_i[: t.index(self.eos_id)]
                ppllike_i = ppllike_i[: t.index(self.eos_id)]
                greep_i = greep_i[: t.index(self.eos_id)]
                greep_increase_i = greep_increase_i[: t.index(self.eos_id)]
                ckl_i = ckl_i[: t.index(self.eos_id)]
                lambda_i = lambda_i[: t.index(self.eos_id)]
                budget_i = budget_i[: t.index(self.eos_id)]
                delta_i = delta_i[: t.index(self.eos_id)]


                t = t[: t.index(self.eos_id)]

            except ValueError:
                pass

            if self.is_decoder_only_model:

                entropy_i = entropy_i[lens[i].item() : ]
                loglike_i = loglike_i[lens[i].item() : ]
                ppllike_i = ppllike_i[lens[i].item() : ]
                greep_i = greep_i[lens[i].item() : ]
                greep_increase_i = greep_increase_i[lens[i].item() : ]
                ckl_i = ckl_i[lens[i].item() : ]
                lambda_i = lambda_i[lens[i].item() : ]
                budget_i = budget_i[lens[i].item() : ]
                delta_i = delta_i[lens[i].item() : ]

            
                t = t[lens[i].item() : ]

            output_str = self.tokenizer.decode(t)

            dic_list.append({
                "hidden_index": index_list[i].item(),
                label: output_str,
                "output_token_num": len(t),
                "entropy_list": entropy_i.tolist(),
                "loglike_list": loglike_i.tolist(),
                "ppllike_list": ppllike_i.tolist(),
                "added_probs": greep_i.tolist(),
                "added_probs_increases": greep_increase_i.tolist(),
                "conditional_kl": ckl_i.tolist(),
                "dual_lambda": lambda_i.tolist(),
                "kl_budget": budget_i.tolist(),
                "adaptive_delta": delta_i.tolist()
            })
        
        return dic_list

    
    def sample_next(
        self,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        ngram_tokens: torch.LongTensor, # (bsz, ngram): tokens to consider when seeding
        temperature: float = 1., # temperature for sampling
        top_p: float = 0.95, # top p for sampling
        current_lambda = None, 
        current_budget = None,
        metric = False,
        cur_pos = None,
        prev_color_list = None,
    ) -> torch.LongTensor:
        
        original_logits = logits

        bsz, vocab_size = logits.shape

        logits = logits.clone()

        # delta_base_table = torch.Tensor(np.arange(0, 10, 0.1)).to(logits.device)

        ################# prepare ##################

        current_lambda = torch.Tensor(current_lambda).to(logits.device)
        current_budget = torch.Tensor(current_budget).to(logits.device)

        # lambda_chose_chocolate = lambda_chose.view(-1, 1).repeat(1, len(delta_base_table))

        # delta_base_chocolate = delta_base_table.view(1, -1).repeat(bsz, 1)

        ## get green list indexes

        addi_indicator = torch.zeros((bsz, vocab_size)).to(logits.device)

        for ii in range(ngram_tokens.shape[0]): # batch of texts
            seed = self.get_seed_rng(ngram_tokens[ii])
            self.rng.manual_seed(seed)
            vocab_permutation = torch.randperm(vocab_size, generator=self.rng)
            greenlist = vocab_permutation[:int(self.gamma * vocab_size)] # gamma * n
            redlist = vocab_permutation[int(self.gamma * vocab_size):]

            if prev_color_list[ii] == 'r':
                addi_indicator[ii][greenlist] = 1
            elif prev_color_list[ii] == 'g':
                addi_indicator[ii][redlist] = 1
            else:
                raise NotImplementedError

        

        ## calculate g and delta
            
        prob = torch.softmax(original_logits, dim=-1)

        g = (prob * addi_indicator).sum(dim=-1)


        # g_chocolate = g.view(-1, 1).repeat(1, len(delta_base_table))

        # choose_argmax_delta_chocolate = obj_func(delta_base_chocolate, lambda_chose_chocolate, g_chocolate)

        # argmax_index = torch.argmax(choose_argmax_delta_chocolate, dim = -1)

        # delta_tilde = torch.gather(delta_base_table, 0, index=argmax_index)

        delta_tilde = torch.clip(1 / current_lambda, min=0, max=15)

        delta_harvest = copy.deepcopy(delta_tilde)

        ## use remaining budget to mask out deltas

        delta_harvest[torch.where(current_budget <= 1e-3)[0]] = 0

        # with this delta, we can modify logits

        bias_placement = delta_harvest.view(-1, 1) * addi_indicator

        logits += bias_placement

        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            next_token = torch.multinomial(probs_sort, num_samples=1) # one hot of next token, ordered by original probs
            next_token = torch.gather(probs_idx, -1, next_token) # one hot of next token, ordered by vocab
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)

        cur_color_list = []

        for i in range(len(prev_color_list)):

            if addi_indicator[i][int(next_token[i].item())] > 1-1e-3:

                # make color switch
                if prev_color_list[i] == 'r':
                    cur_color_list.append('g')
                    
                else:
                    cur_color_list.append('r')

            elif addi_indicator[i][int(next_token[i].item())] < 1e-3:

                if prev_color_list[i] == 'r':
                    cur_color_list.append('r')
                else:
                    cur_color_list.append('g')
                    

            else:

                raise ValueError
        

        if metric:

            entropy, loglike, ppllike, GreenP, GreenP_increase, kl = self.retrieve_metric(original_logits, logits, next_token)

            ## budget update

            new_budget = torch.clamp(current_budget - torch.Tensor(kl).to(logits.device), min=0, max=1e4)

            # update lambda

            dgt = - KLt(delta_tilde, g) + self.b

            
            use_eta = self.eta

            if self.h_func == "2x2":

                new_lam = torch.clamp(current_lambda - use_eta * dgt, min=0.1, max=1)

            elif self.h_func == "xlnx":

                new_lam = current_lambda * torch.exp( - use_eta * dgt)

            else:

                raise NotImplementedError


            return next_token, entropy, loglike, ppllike, GreenP, GreenP_increase, kl \
                , new_lam.cpu().numpy(), new_budget.cpu().numpy(), delta_harvest.cpu().numpy(), cur_color_list

        else:
            return next_token






    def retrieve_metric(self, original_logits, logits, next_token):

        assert torch.all(logits >= original_logits)

        green_mask = (logits > original_logits + 1e-3)

        original_prob = torch.softmax(original_logits, dim=-1)
        prob = torch.softmax(logits, dim=-1)

        ppllike = (prob * torch.log(original_prob)).sum(dim=-1).detach().cpu().numpy()

        entropy = (original_prob * torch.log(original_prob)).sum(dim=-1).detach().cpu().numpy()

        GreenP = (original_prob * green_mask).sum(dim=-1).detach().cpu().numpy()

        GreenP_increase = (prob * green_mask).sum(dim=-1).detach().cpu().numpy() - GreenP

        kl = (prob * (torch.log(prob) - torch.log(original_prob))).sum(dim=-1).detach().cpu().numpy()

        loglike = torch.log(torch.gather(original_prob, -1, next_token.reshape(-1, 1))).reshape(-1).detach().cpu().numpy()

        return entropy, loglike, ppllike, GreenP, GreenP_increase, kl
    
