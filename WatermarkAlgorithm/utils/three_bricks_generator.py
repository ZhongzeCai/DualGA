# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import numpy as np


# the generator can not take as input the model
# as its properties will not be accessible in DDP setting
class WmGenerator():
    def __init__(self, 
            tokenizer: LlamaTokenizer, 
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317,
            payload: int = 0,
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
    def generate(
        self,
        model,
        index_list,
        prompt_batch,
        mask_batch,
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        """
        Generate text from prompts. 
        Adapted from https://github.com/facebookresearch/llama/
        """
        
        assert prompt_batch.shape == mask_batch.shape
        assert torch.min(mask_batch).item() < 1e-3 and torch.max(mask_batch).item() > 1-1e-3

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
        for cur_pos in range(start_pos, total_len):
            outputs = model.forward(
                tokens[:, prev_pos:cur_pos], use_cache=True, past_key_values=outputs.past_key_values if prev_pos > 0 else None
            )
            ngram_tokens = tokens[:, cur_pos-self.ngram:cur_pos]
            next_toks = self.sample_next(outputs.logits[:, -1, :], ngram_tokens, temperature, top_p)
            tokens[:, cur_pos] = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_toks)
            prev_pos = cur_pos


        # attach at first position the hidden index, and second position the prompt len
        tokens = torch.cat((torch.Tensor(index_list).view(-1,1).to(model.device) , torch.Tensor(lens).view(-1, 1).to(model.device) , tokens), axis=1).long()
        
        return tokens

        
    def decode_generated_token(self, tokens, max_gen_len, label = "gen_str"):
        index_list = tokens[:, 0]
        lens = tokens[:, 1]
        tokens = tokens[:, 2:]

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: lens[i].item() + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.eos_id)]
            except ValueError:
                pass

            if self.is_decoder_only_model:

                t = t[lens[i].item() : ]

            output_str = self.tokenizer.decode(t)

            decoded.append({
                "hidden_index": index_list[i].item(),
                label: output_str,
                "output_token_num": len(t)
            })

        return decoded

    @torch.no_grad()
    def generate_metric(
        self,
        model,
        index_list,
        prompt_batch,
        mask_batch,
        max_gen_len: int,
        temperature: float = 0.8,
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

        for cur_pos in range(start_pos, total_len):
            outputs = model.forward(
                tokens[:, prev_pos:cur_pos], use_cache=True, past_key_values=outputs.past_key_values if prev_pos > 0 else None
            )

            ngram_tokens = tokens[:, cur_pos-self.ngram:cur_pos]
            
            next_toks, entropy, loglike, ppllike, GreenP, GreenP_increase, kl,  = self.sample_next(outputs.logits[:, -1, :], ngram_tokens, temperature, top_p, metric = True)
            tokens[:, cur_pos] = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_toks)
            
            np_mask = input_text_mask[:, cur_pos].detach().cpu().numpy()

            entropy_list[:, cur_pos] = np.where(np_mask, 0, entropy)
            loglike_list[:, cur_pos] = np.where(np_mask, 0, loglike)
            ppllike_list[:, cur_pos] = np.where(np_mask, 0, ppllike)
            green_probs[:, cur_pos] = np.where(np_mask, 0, GreenP)
            green_probs_increases[:, cur_pos] = np.where(np_mask, 0, GreenP_increase)
            conditional_kl[:, cur_pos] = np.where(np_mask, 0, kl)

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

            # cut to eos tok if any
            try:

                entropy_i = entropy_i[: t.index(self.eos_id)]
                loglike_i = loglike_i[: t.index(self.eos_id)]
                ppllike_i = ppllike_i[: t.index(self.eos_id)]
                greep_i = greep_i[: t.index(self.eos_id)]
                greep_increase_i = greep_increase_i[: t.index(self.eos_id)]
                ckl_i = ckl_i[: t.index(self.eos_id)]

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
                "conditional_kl": ckl_i.tolist()
            })
        
        return dic_list

    
    def sample_next(
        self,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        ngram_tokens: torch.LongTensor, # (bsz, ngram): tokens to consider when seeding
        temperature: float = 0.8, # temperature for sampling
        top_p: float = 0.95, # top p for sampling
        metric = False
    ) -> torch.LongTensor:
        """ Vanilla sampling with temperature and top p."""
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

            entropy, loglike, ppllike, GreenP, GreenP_increase, kl = self.retrieve_metric(logits, logits, next_token)

            return next_token, entropy, loglike, ppllike, GreenP, GreenP_increase, kl

        else:
            return next_token






    def retrieve_metric(self, original_logits, logits, next_token):

        # assert torch.all(logits >= original_logits)

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
    

class OpenaiGenerator(WmGenerator):
    """ Generate text using LLaMA and Aaronson's watermarking method. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        

    def sample_next(
        self,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        ngram_tokens: torch.LongTensor, # (bsz, ngram): tokens to consider when seeding
        temperature: float = 0.8, # temperature for sampling
        top_p: float = 0.95, # top p for sampling
        metric = True
    ) -> torch.LongTensor:
        """
        From ngram tokens, select the next token based on the following:
        - hash the ngram tokens and get a seed
        - use the seed to generate V random number r between [0,1]
        - select argmax ( r^(1/p) )
        payload (the message) is encoded by shifting the secret vector r by `payload`.
        """
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            for ii in range(ngram_tokens.shape[0]): # batch of texts
                # seed with hash of ngram tokens
                seed = self.get_seed_rng(ngram_tokens[ii])
                self.rng.manual_seed(seed)
                # generate rs randomly between [0,1]
                rs = torch.rand(self.tokenizer.vocab_size, generator=self.rng) # n
                rs = rs.roll(-self.payload)
                rs = torch.Tensor(rs).to(probs_sort.device)
                rs = rs[probs_idx[ii]] 
                # compute r^(1/p)
                probs_sort[ii] = torch.pow(rs, 1/probs_sort[ii])
            # select argmax ( r^(1/p) )
            next_token = torch.argmax(probs_sort, dim=-1, keepdim=True)
            next_token = torch.gather(probs_idx, -1, next_token)
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)

        if not metric:
            return next_token
        else:

            new_logits = torch.zeros(logits.shape).to(logits.device)
            for i in range(logits.shape[0]):
                new_logits[i, next_token[i]] = 100

            entropy, loglike, ppllike, GreenP, GreenP_increase, kl = self.retrieve_metric(logits, new_logits, next_token)

            return next_token, entropy, loglike, ppllike, GreenP, GreenP_increase, kl
        

class MarylandGenerator(WmGenerator):
    """ Generate text using LLaMA and Maryland's watemrarking method. """
    def __init__(self, 
            *args, 
            gamma: float = 0.5,
            delta: float = 1.0,
            **kwargs
        ):
        super().__init__(*args, **kwargs)        
        self.gamma = gamma
        self.delta = delta

    def sample_next(
        self,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        ngram_tokens: torch.LongTensor, # (bsz, ngram): tokens to consider when seeding
        temperature: float = 0.8, # temperature for sampling
        top_p: float = 0.95, # top p for sampling
        metric = False
    ) -> torch.LongTensor:
        """
        From ngram tokens, select the next token based on the following:
        - hash the ngram tokens and get a seed
        - use the seed to partition the vocabulary into greenlist (gamma*V words) and blacklist 
        - add delta to greenlist words' logits
        payload (the message) is encoded by shifting the secret vector r by `payload`.
        """

        original_logits = logits
        logits = self.logits_processor(logits, ngram_tokens)



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

            return next_token, entropy, loglike, ppllike, GreenP, GreenP_increase, kl

        else:
            return next_token

    def logits_processor(self, logits, ngram_tokens):
        """Process logits to mask out words in greenlist."""
        bsz, vocab_size = logits.shape
        logits = logits.clone()
        
        for ii in range(ngram_tokens.shape[0]): # batch of texts
            seed = self.get_seed_rng(ngram_tokens[ii])
            self.rng.manual_seed(seed)
            vocab_permutation = torch.randperm(vocab_size, generator=self.rng)
            greenlist = vocab_permutation[:int(self.gamma * vocab_size)] # gamma * n
            bias = torch.zeros(vocab_size).to(logits.device) # n
            bias[greenlist] = self.delta
            bias = bias.roll(-self.payload)
            logits[ii] += bias # add bias to greenlist words
        return logits