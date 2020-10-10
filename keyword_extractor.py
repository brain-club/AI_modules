# Modified from http://github.com/ibtra/BERT-Keyword-Extractor

from pytorch_pretrained_bert import BertTokenizer, BertForTokenClassification

import torch
import numpy as np


def keywordextract(sentence, model_path='./pretrained/keyword_extraction_pretrained.pt'):
    # returns a single keyword of given sentence
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=3)

    text = sentence
    tkns = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tkns)
    segments_ids = [0] * len(tkns)
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)
    segments_tensors = torch.tensor([segments_ids]).to(device)
    model = torch.load(model_path)
    model.eval()
    prediction = []
    logit = model(tokens_tensor, token_type_ids=None,
                                  attention_mask=segments_tensors)
    logit = logit.detach().cpu().numpy()
    prediction.extend([list(p) for p in np.argmax(logit, axis=2)])

    keyword = None
    for k, j in enumerate(prediction[0]):
        if j==1 or j==0:
            # print(tokenizer.convert_ids_to_tokens(tokens_tensor[0].to('cpu').numpy())[k])
            keyword = tokenizer.convert_ids_to_tokens(tokens_tensor[0].to('cpu').numpy())[k]

    if "#" in keyword:
        keyword = keyword.replace("#", "")
        for word in sentence.split():
            if keyword in word:
                keyword = word.lower()

    return keyword    


if __name__ == "__main__":
    
    sentence = "Chatbot service - where newly joined company member can get the answers via text"
    
    print(keywordextract(sentence))
