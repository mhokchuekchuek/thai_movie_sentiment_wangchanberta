import sys
sys.path.append("/code/notebook")
from script.training import model, _tokenizer
import torch

dict_ja = {'pos': 0, 'neu': 1, 'neg': 2}
def get_keys(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key
        
def predict_unseen(unseen_text, model_ja, tokenizer):
    input_ids = tokenizer.encode(unseen_text, add_special_tokens=True)
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    labels = torch.tensor([1]).unsqueeze(0)
    outputs = model_ja(input_ids, labels=labels)
    logits = outputs[0]
    return get_keys(logits.argmax().item(), dict_ja)