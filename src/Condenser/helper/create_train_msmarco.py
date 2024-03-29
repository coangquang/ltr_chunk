# Copyright 2021 Condenser Author All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import json
from multiprocessing import Pool
import random
import datasets
from tqdm import tqdm
from transformers import AutoTokenizer
import nltk
nltk.download("punkt", quiet=True)
import itertools

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str)
parser.add_argument('--save_to', type=str)
parser.add_argument('--column', type=int, help="take specified column")
parser.add_argument('--tokenizer_name', required=True)
parser.add_argument('--max_len', type=int, default=512)
parser.add_argument('--chunksize', type=int, default=500)
parser.add_argument('--short_sentence_prob', type=float, default=0.1)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
file_name = "condenser_corpus_vectors"

target_length = args.max_len - tokenizer.num_special_tokens_to_add(pair=False)

'''
def encoder_one_article(text: str):
    ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
'''    
    
def encode_one_line(text: str):
    if args.column is not None:
        text = text.split('\t')[args.column]
    blocks = []
    sentences = nltk.sent_tokenize(text)
    ids = [
        tokenizer(
            sent,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"] for sent in sentences
    ]
    curr_len = 0
    curr_block = []

    curr_tgt_len = target_length if random.random() > args.short_sentence_prob else random.randint(1, target_length)

    for sent in ids:
        if curr_len + len(sent) > curr_tgt_len and curr_len > 0:
            blocks.append(curr_block)
            curr_block = []
            curr_len = 0
            curr_tgt_len = target_length if random.random() > args.short_sentence_prob \
                else random.randint(1, target_length)
        curr_len += len(sent)
        curr_block.extend(sent)
    if len(curr_block) > 0:
        blocks.append(curr_block)
    return blocks


#with open(args.file, 'r') as corpus_file:
#    #lines = corpus_file.readlines()
#    lines = json.load(corpus_file)
    

data = datasets.load_dataset('sentence-transformers/embedding-training-data', data_files='msmarco-triplets.jsonl.gz', split='train')
eval_data = data.select(range(50000))
lines = list(itertools.chain.from_iterable(eval_data['pos'] + eval_data['neg'])) 


with open(os.path.join(args.save_to, file_name + '.json'), 'w') as tokenized_file:
    with Pool() as p:
        all_blocks = p.imap_unordered(
            encode_one_line,
            tqdm(lines),
            chunksize=args.chunksize,
        )
        for blocks in all_blocks:
            for block in blocks:
                tokenized_file.write(json.dumps({'text': block}) + '\n')