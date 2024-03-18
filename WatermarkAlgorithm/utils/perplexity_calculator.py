from .load_model import load_model
import torch


def batch_ppl(whole_dicLists, args=None):
    
    model, tokenizer, _ = load_model(args)

    modelName = args.modelName

    for dicList in whole_dicLists:

        ppl_for_wm_gened(dicList, model, tokenizer, modelName)

        
def ppl_for_wm_gened(dicList, model, tokenizer, modelName):

    strKeys = ['text', 'baseline_completion']

    for key in dicList[0].keys():

        if "generated_str" in key:

            strKeys.append(key)

    ppl_for_dicList(dicList, model, tokenizer, modelName, strKeys)


def ppl_for_dicList(dicList, model, tokenizer, modelName, strKeys):

    for dic in dicList:

        for key in strKeys:

            dic[key + "_"+modelName+"_ppl"] = calculate_perplexity(dic[key], model, tokenizer)





def calculate_perplexity(text, model, tokenizer):
    
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors = "pt")
        loss = model(input_ids = inputs["input_ids"].to(model.device), labels = inputs["input_ids"]).loss
        return loss.item()  



