from .watermark_processor import WatermarkLogitsProcessor
import torch
from transformers import LogitsProcessorList, DataCollatorWithPadding



## maryland basic version

"""
gen_kwargs = dict(max_new_tokens=args.max_new_tokens)

    
if args.use_sampling:
    gen_kwargs.update(
        dict(
            do_sample=True,
            top_k=args.top_k,
            top_p=args.top_p,
            typical_p=args.typical_p,
            temperature=args.sampling_temp,
        )
    )
else:
    gen_kwargs.update(dict(num_beams=args.num_beams))

"""
"""
watermark_algo_args.keys:

    gamma, delta, seeding_scheme

model_args:

    is_decoder_only_model, use_sampling, max_new_tokens, top_k, top_p, typical_p, sampling_temp, num_beams


"""
def gen_watermarked_n_unwatermarked(
        input_ids, gen_model, tokenizer,
        extract_hidden_index=True,
        generation_seed = None,
        watermark_algo_args = None,
        model_args = None):
    
    if extract_hidden_index:

        hidden_indices = input_ids[:,0]

        input_ids = input_ids[:,1:].to(gen_model.device)

    else:

        input_ids = input_ids.to(gen_model.device)

    output_dics = []

    watermark_processor = WatermarkLogitsProcessor(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=watermark_algo_args.gamma,
        delta=watermark_algo_args.delta,
        seeding_scheme=watermark_algo_args.seeding_scheme,
        store_spike_ents=False,
        select_green_tokens=True,
    )

    gen_kwargs = dict(max_new_tokens=model_args.max_new_tokens)

    
    if model_args.use_sampling:
        gen_kwargs.update(
            dict(
                do_sample=True,
                top_k=model_args.top_k,
                top_p=model_args.top_p,
                typical_p=model_args.typical_p,
                temperature=model_args.sampling_temp,
            )
        )
    else:
        gen_kwargs.update(dict(num_beams=model_args.num_beams))

    with torch.no_grad():
        if generation_seed is not None:
            torch.manual_seed(generation_seed)
        output_without_watermark = gen_model.generate(input_ids=input_ids, **gen_kwargs)

        if generation_seed is not None:
            torch.manual_seed(generation_seed)
        output_with_watermark = gen_model.generate(input_ids=input_ids, logits_processor=LogitsProcessorList([watermark_processor]), **gen_kwargs)

    if model_args.is_decoder_only_model:
        # need to isolate the newly generated tokens
        output_without_watermark = output_without_watermark[:, input_ids.shape[-1] :]
        output_with_watermark = output_with_watermark[:, input_ids.shape[-1] :]

    decoded_output_without_watermark = tokenizer.batch_decode(
        output_without_watermark, skip_special_tokens=True
    )
    decoded_output_with_watermark = tokenizer.batch_decode(
        output_with_watermark, skip_special_tokens=True
    )

    for i in range(len(input_ids)):

        dict_harvest = {
            "no_wm_output": decoded_output_without_watermark[i],
            "w_wm_output": decoded_output_with_watermark[i],
            "no_wm_output_length": (output_without_watermark[i] != (tokenizer.pad_token_id).sum()).item(),
            "w_wm_output_length": (output_with_watermark[i] != (tokenizer.pad_token_id).sum()).item()
        }

        if extract_hidden_index:

            dict_harvest["hidden_index"] = int(hidden_indices[i])

        # possibly add other metrics
        # ...............
            
        output_dics.append(dict_harvest)

    return output_dics





