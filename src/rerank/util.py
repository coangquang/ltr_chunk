import json
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

def build_cross_sub_dataloader(dscorpus, json_file, csv_file, tokenizer, text_len, batch_size, no_negs=30, shuffle=False):
    """
    This function builds train, val, test dataframe for training and evaluating cross-encoder
    """
    sep_token = " . " + tokenizer.sep_token + " " 
    texts, labels = [], []
    with open(json_file, 'r') as f:
        retrieved_sub_list = json.load(f)
    df = pd.read_csv(csv_file)
    
    for i in range(len(df)):
        tokenized_question = df['tokenized_question'][i]
        retrieved_sub_ids = retrieved_sub_list[i]
        ans_ids = str(df['ans_id'][i])
        ans_ids = [int(x) for x in ans_ids.split(", ")]
        ans_sub_ids = df['ans_sub_id'][i][1:-1]
        ans_sub_ids = [int(x) for x in ans_sub_ids.split(", ")]
        for a_sub_id in ans_sub_ids:
            tokenized_text = dscorpus['tokenized_text'][a_sub_id]
            text = tokenized_question + sep_token + tokenized_text
            
            for j in range(no_negs-1):
                labels.append(0)
                texts.append(text)
                
        neg_ids = [x for x in retrieved_sub_ids if dscorpus['id'][x] not in ans_ids]
        neg_ids = neg_ids[:no_negs]
        for neg_id in neg_ids:
            tokenized_text = dscorpus['tokenized_text'][neg_id]
            text = tokenized_question + sep_token + tokenized_text
            labels.append(1)
            texts.append(text)    
            
    C = tokenizer.batch_encode_plus(texts, padding='max_length', truncation=True, max_length=text_len, return_tensors='pt')
    labels = torch.tensor(labels, dtype=torch.long)
    data_tensor = TensorDataset(C['input_ids'], C['attention_mask'], labels)
    data_loader = DataLoader(data_tensor, batch_size=batch_size, shuffle=shuffle)
    return data_loader  

def build_cross_dataloader(dcorpus, json_file, csv_file, tokenizer, text_len, batch_size, no_negs=30, shuffle=False):
    """
    This function builds train, val, test dataframe for training and evaluating cross-encoder
    """
    sep_token = " . " + tokenizer.sep_token + " " 
    texts, labels = [], []
    with open(json_file, 'r') as f:
        retrieved_list = json.load(f)
    df = pd.read_csv(csv_file)
    
    for i in range(len(df)):
        tokenized_question = df['tokenized_question'][i]
        retrieved_ids = retrieved_list[i]
        ans_ids = str(df['ans_id'][i])
        ans_ids = [int(x) for x in ans_ids.split(", ")]
        for a_id in ans_ids:
            tokenized_text = dcorpus['tokenized_text'][a_id]
            text = tokenized_question + sep_token + tokenized_text
            
            for j in range(no_negs-1):
                labels.append(0)
                texts.append(text)
                
        neg_ids = [x for x in retrieved_ids if x not in ans_ids]
        neg_ids = neg_ids[:no_negs]
        for neg_id in neg_ids:
            tokenized_text = dcorpus['tokenized_text'][neg_id]
            text = tokenized_question + sep_token + tokenized_text
            labels.append(1)
            texts.append(text)    
            
    C = tokenizer.batch_encode_plus(texts, padding='max_length', truncation=True, max_length=text_len, return_tensors='pt')
    labels = torch.tensor(labels, dtype=torch.long)
    data_tensor = TensorDataset(C['input_ids'], C['attention_mask'], labels)
    data_loader = DataLoader(data_tensor, batch_size=batch_size, shuffle=shuffle)
    return data_loader             