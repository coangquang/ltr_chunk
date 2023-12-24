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
from transformers import AutoTokenizer
from multiprocessing import Pool
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--corpus_path', required=True)
parser.add_argument('--file', required=True)
parser.add_argument('--save_to', required=True)
parser.add_argument('--tokenizer', required=True)
parser.add_argument('--save_dir', required=True)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)


def encode_one(line):
    # item = json.loads(line)
    # spans = item['spans']
    spans = line.split("#")
    if len(spans) <= 1:
        return None
    tokenized = [
        tokenizer(
            s,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"] for s in spans
    ]
    return json.dumps({'spans': tokenized})

data = json.load(open(args.corpus_path))
co_f = open(os.path.join(args.save_dir, "cocondenser_data.json"), "w")
count = 0
for article_full in tqdm(data):
    article_full = article_full.replace("\n", " ")    
    # Save data for cocondenser 
    spans = []
    #passages = re.split(r"\n[0-9]+\. |1\. ", article_text)
    passages = article_full.split(".")
    passages = [x for x in passages if (len(x) > 0 and x!="" and x!=" ")]
    length = len(passages)
    if length > 1:
        passage1 = ".".join(passages[:(length//2)])
        passage2 = ".".join(passages[(length//2):])
        new_passages = [passage1, passage2]
        for idx, p in enumerate(new_passages):
            spans.append(p)
        co_f.write("#".join(spans) + "\n")    
co_f.close()

with open(args.save_to, 'w') as f:
    with Pool() as p:
        all_tokenized = p.imap_unordered(
            encode_one,
            #tqdm(data),
            tqdm(open(args.file)),
            chunksize=500,
        )
        for x in all_tokenized:
            if x is None:
                continue
            f.write(x + '\n')
