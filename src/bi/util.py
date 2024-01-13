import torch
import json
from torch import Tensor as T
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer

def get_tokenizer(model_checkpoint):
    """
    Get tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    return tokenizer

def query_trans(text, tokenizer):
    return "câu hỏi " + tokenizer.sep_token + " " + text
def context_trans(text, tokenizer):
    return "đoạn văn " + tokenizer.sep_token + " " + text

def build_dpr_traindata(corpus, df, tokenizer, q_len, ctx_len, batch_size, no_hard, shuffle = False, all_data=False):
    """
    This funtion builds train and val data loader for biencoder training
    """
    tokenized_questions = [query_trans(x, tokenizer) for x in df["tokenized_question"].tolist()]
    questions = []
    positives = []
    negatives = []
    ans_ids = df["best_ans_id"].tolist()
    neg_ids = df["neg_ids"].tolist()

    for i in range(len(df)):
        #positive_ids = [int(x) for x in str(ans_ids[i][1:-1]).split(", ")]
        positive_ids = json.loads(str(ans_ids[i]))
        poss = [context_trans(corpus[j], tokenizer) for j in positive_ids]
        if no_hard != 0:
            #negative_ids = [int(y) for y in neg_ids[i][1:-1].split(", ")[:no_hard]]
            negative_ids = json.loads(str(neg_ids[i]))[:no_hard]
            negs = [context_trans(corpus[j], tokenizer) for j in negative_ids]

        if all_data:
            for pos in poss:
                questions.append(tokenized_questions[i])
                positives.append(pos)
                if no_hard != 0:
                    negatives += negs
        else:
            questions.append(tokenized_questions[i])
            positives.append(poss[0])
            if no_hard != 0:
                negatives += negs

    Q = tokenizer.batch_encode_plus(questions, padding='max_length', truncation=True, max_length=q_len, return_tensors='pt')
    P = tokenizer.batch_encode_plus(positives, padding='max_length', truncation=True, max_length=ctx_len, return_tensors='pt')
    if no_hard != 0:
        N = tokenizer.batch_encode_plus(negatives, padding='max_length', truncation=True, max_length=ctx_len, return_tensors='pt')
        N_ids = N['input_ids'].view(-1,no_hard,ctx_len)
        N_attn = N['attention_mask'].view(-1,no_hard,ctx_len)
        data_tensor = TensorDataset(Q['input_ids'], Q['attention_mask'], P['input_ids'], P['attention_mask'], N_ids, N_attn)
    else:
        data_tensor = TensorDataset(Q['input_ids'], Q['attention_mask'], P['input_ids'], P['attention_mask'])
    data_loader = DataLoader(data_tensor, batch_size=batch_size, shuffle=shuffle)
    return data_loader