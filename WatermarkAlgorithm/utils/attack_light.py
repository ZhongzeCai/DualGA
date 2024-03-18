import torch


# vocab_size = model.get_output_embeddings().weight.shape[0] - 8

def substitution_attack(tokens,p,vocab_size,distribution=None):
    if distribution is None:
        distribution = lambda x : torch.ones(size=(len(tokens),vocab_size)) / vocab_size
    idx = torch.randperm(len(tokens))[:int(p*len(tokens))]
    
    new_probs = distribution(tokens)
    samples = torch.multinomial(new_probs,1).flatten()
    tokens[idx] = samples[idx]
    
    return tokens

def deletion_attack(tokens,p):
    idx = torch.randperm(len(tokens))[:int(p*len(tokens))]
    
    keep = torch.ones(len(tokens),dtype=torch.bool)
    keep[idx] = False
    tokens = tokens[keep == True]
        
    return tokens

def insertion_attack(tokens,p,vocab_size,distribution=None):
    if distribution is None:
        distribution = lambda x : torch.ones(size=(len(tokens),vocab_size)) / vocab_size
    idx = torch.randperm(len(tokens))[:int(p*len(tokens))]
    
    new_probs = distribution(tokens)
    samples = torch.multinomial(new_probs,1)
    for i in idx.sort(descending=True).values:
        tokens = torch.cat([tokens[:i],samples[i],tokens[i:]])
        tokens[i] = samples[i]
        
    return tokens


# vocab_size = model.get_output_embeddings().weight.shape[0]
# eff_vocab_size = vocab_size - args.truncate_vocab
# args.truncate_vocab = 8
def corrupt(text, tokenizer, deletion_eps, insertion_eps, substitution_eps):
    eff_vocab_size = tokenizer.vocab_size - 8
    tokens = torch.Tensor(tokenizer.encode(text, add_special_tokens=False)).long()
    tokens = deletion_attack(tokens, deletion_eps)
    tokens = insertion_attack(tokens,insertion_eps,eff_vocab_size)
    tokens = substitution_attack(tokens,substitution_eps,eff_vocab_size)
    attacked_text = tokenizer.decode(tokens)

    return attacked_text

