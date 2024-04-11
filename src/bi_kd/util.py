import torch
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
    #return "câu hỏi " + tokenizer.sep_token + " " + text
    return text

def context_trans(text, tokenizer):
    #return "đoạn văn " + tokenizer.sep_token + " " + text
    return text

def build_dpr_traindata(corpus, dataset, tokenizer, q_len, ctx_len, batch_size, no_hard, shuffle = False):
    """
    This funtion builds train and val data loader for biencoder training
    """
    questions = []
    positives = []
    negatives = []
    scores = []
    #ans_ids = df["best_ans_id"].tolist()
    #if no_hard != 0:
    #    neg_ids = df["neg_ids"].tolist()

    for i in range(len(dataset)):
        positive = dataset['positives'][i]
        max_score = max(positive['score'])
        pos_id = positive['doc_id'][positive['score'].index(max_score)]
        score = [max_score]
        pos = context_trans(corpus[pos_id], tokenizer)
        if len(negative['doc_id']) >= no_hard:
            negative = dataset['negatives'][i]
            neg_ids = negative['doc_id'][:no_hard]
            negs = [context_trans(corpus[j], tokenizer) for j in neg_ids]
            score += negative['score'][:no_hard]
            negatives += negs
        else:
            continue
        
        questions.append(dataset['query'][i])
        positives.append(pos)
        scores.append(score)

    Q = tokenizer.batch_encode_plus(questions, padding='max_length', truncation=True, max_length=q_len, return_tensors='pt')
    P = tokenizer.batch_encode_plus(positives, padding='max_length', truncation=True, max_length=ctx_len, return_tensors='pt')
    scores = torch.tensor(scores, dtype=torch.float)
    if no_hard != 0:
        N = tokenizer.batch_encode_plus(negatives, padding='max_length', truncation=True, max_length=ctx_len, return_tensors='pt')
        N_ids = N['input_ids'].view(-1,no_hard,ctx_len)
        N_attn = N['attention_mask'].view(-1,no_hard,ctx_len)
        data_tensor = TensorDataset(Q['input_ids'], Q['attention_mask'], P['input_ids'], P['attention_mask'], N_ids, N_attn, scores)
    else:
        data_tensor = TensorDataset(Q['input_ids'], Q['attention_mask'], P['input_ids'], P['attention_mask'], scores)
    data_loader = DataLoader(data_tensor, batch_size=batch_size, shuffle=shuffle)
    return data_loader