from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    DataCollatorWithPadding,
)
import torch

def load_model(args):
    """Load and return the model and tokenizer"""

    args.is_seq2seq_model = any(
        [(model_type in args.model_name_or_path) for model_type in ["t5", "T0"]]
    )
    args.is_decoder_only_model = any(
        [(model_type in args.model_name_or_path) for model_type in ["gpt", "opt", "bloom", "llama"]]
    )
    if args.is_seq2seq_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    elif args.is_decoder_only_model:
        if args.load_fp16:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path, torch_dtype=torch.float16
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(f"Unknown model type: {args.model_name_or_path}")

    if args.use_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.load_fp16:
            pass
        else:
            model = model.to(device)
    else:
        device = "cpu"
    model.eval()

    if args.is_decoder_only_model:
        padding_side = "left"
    else:
        raise NotImplementedError(
            "Need to check how to handle padding for seq2seq models when calling generate"
        )

    if "llama" in args.model_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(
            args.model_name_or_path, padding_side=padding_side
        )
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, padding_side=padding_side
        )

    args.model_max_length = model.config.max_position_embeddings

    return model, tokenizer, device