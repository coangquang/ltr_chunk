import json
import pandas as pd
from sentence_transformers import InputExample

def build_cross_sub_data(dscorpus, json_file, csv_file, no_negs= 30):
    """
    This function builds train, val, test dataframe for training and evaluating cross-encoder
    """
    with open(json_file, 'r') as f:
        retrieved_sub_list = json.load(f)
    df = pd.read_csv(csv_file)
    data = []
    for i in range(len(df)):
        tokenized_question = df['tokenized_question'][i]
        retrieved_sub_ids = retrieved_sub_list[i]
        ans_ids = str(df['ans_id'][i])
        ans_ids = [int(x) for x in ans_ids.split(", ")]
        ans_sub_ids = df['ans_sub_id'][i][1:-1]
        ans_sub_ids = [int(x) for x in ans_sub_ids.split(", ")]
        for a_sub_id in ans_sub_ids:
            tokenized_text = dscorpus['tokenized_text'][a_sub_id]
            example = InputExample(texts=[tokenized_question, tokenized_text], label=1)
            for j in range(no_negs):
                data.append(example)
                
        neg_ids = [x for x in retrieved_sub_ids if dscorpus['id'][x] not in ans_ids]
        neg_ids = neg_ids[:no_negs]
        for neg_id in neg_ids:
            tokenized_text = dscorpus['tokenized_text'][neg_id]
            example = InputExample(texts=[tokenized_question, tokenized_text], label=0)
            data.append(example)           
    return data

def build_cross_data(dcorpus, json_file, csv_file, no_negs= 30):
    """
    This function builds train, val, test dataframe for training and evaluating cross-encoder
    """
    with open(json_file, 'r') as f:
        retrieved_list = json.load(f)
    df = pd.read_csv(csv_file)
    data = []
    for i in range(len(df)):
        tokenized_question = df['tokenized_question'][i]
        retrieved_ids = retrieved_list[i]
        ans_ids = str(df['ans_id'][i])
        ans_ids = [int(x) for x in ans_ids.split(", ")]
        for a_id in ans_ids:
            tokenized_text = dcorpus['tokenized_text'][a_id]
            example = InputExample(texts=[tokenized_question, tokenized_text], label=1)
            for j in range(no_negs):
                data.append(example)
                
        neg_ids = [x for x in retrieved_ids if x not in ans_ids]
        neg_ids = neg_ids[:no_negs]
        for neg_id in neg_ids:
            tokenized_text = dcorpus['tokenized_text'][neg_id]
            example = InputExample(texts=[tokenized_question, tokenized_text], label=0)
            data.append(example)           
    return data
