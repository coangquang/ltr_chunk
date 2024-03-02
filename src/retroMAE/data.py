import os
import random
from copy import deepcopy
from dataclasses import dataclass

import torch.utils.data.dataset
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import DataCollatorForWholeWordMask

from utils import tensorize_batch
from typing import List, Dict
import itertools

class DatasetForPretraining(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        if os.path.isdir(data_dir):
            datasets = []
            for file in os.listdir(data_dir):
                print(f"Loading {file}")
                file = os.path.join(data_dir, file)
                datasets.append(self.load_dataset(file))
            self.dataset = concatenate_datasets(datasets)
        else:
            print(f"Loading {data_dir}")
            self.dataset = self.load_datasett()

    def load_dataset(self, file):
        if file.endswith('.jsonl') or file.endswith('.json'):
            return load_dataset('json', data_files=file)['train']
        elif os.path.isdir(file):
            return Dataset.load_from_disk(file)
        else:
            raise NotImplementedError(f"Not support this file format:{file}")
        
    def load_datasett(self):
        data = load_dataset('sentence-transformers/embedding-training-data', data_files='msmarco-triplets.jsonl.gz', split='train').select(range(50000))
        return list(itertools.chain.from_iterable(data['pos']))# + data['neg'])) 
        
    def __getitem__(self, item):
        return self.dataset[item]#['pos']

    def __len__(self):
        return len(self.dataset)


@dataclass
class RetroMAECollator(DataCollatorForWholeWordMask):
    max_seq_length: int = 512
    encoder_mlm_probability: float = 0.15
    decoder_mlm_probability: float = 0.15
    
    def __post_init__(self):
        super(RetroMAECollator, self).__post_init__()

        from transformers import BertTokenizer, BertTokenizerFast, PhobertTokenizer
        from transformers import RobertaTokenizer, RobertaTokenizerFast
        from transformers import DebertaV2Tokenizer
        if isinstance(self.tokenizer, (BertTokenizer, BertTokenizerFast)):
            self.whole_word_cand_indexes = self._whole_word_cand_indexes_bert
        elif isinstance(self.tokenizer, (RobertaTokenizer, RobertaTokenizerFast)):
            self.whole_word_cand_indexes = self. _whole_word_cand_indexes_roberta
        elif isinstance(self.tokenizer, DebertaV2Tokenizer):
            self.whole_word_cand_indexes = self._whole_word_cand_indexes_deberta_v2
        else:
            self.whole_word_cand_indexes = self._whole_word_cand_indexes_deberta_v2

        self.specials = self.tokenizer.all_special_tokens

    def _whole_word_cand_indexes_bert(self, input_tokens: List[str]):
        cand_indexes = []
        for (i, token) in enumerate(input_tokens):
            if token in self.specials:
                continue

            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])
        return cand_indexes

    def _whole_word_cand_indexes_roberta(self, input_tokens: List[str]):
        cand_indexes = []
        for (i, token) in enumerate(input_tokens):
            if token in self.specials:
                continue

            if i == 0:
                cand_indexes.append([0])
            elif not token.startswith('\u0120'):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])
        return cand_indexes
    
    def _whole_word_cand_indexes_deberta_v2(self, input_tokens:List[str]):
        cand_indexes = []
        for (i, token) in enumerate(input_tokens):
            if token in self.specials:
                continue
            else:
                cand_indexes.append([i])
        return cand_indexes

    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """

        cand_indexes = self.whole_word_cand_indexes(input_tokens)

        random.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        assert len(covered_indexes) == len(masked_lms)
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels

    def __call__(self, examples):
        input_ids_batch = []
        attention_mask_batch = []
        encoder_mlm_mask_batch = []
        decoder_labels_batch = []
        decoder_matrix_attention_mask_batch = []

        for e in examples:

            e_trunc = self.tokenizer.encode(e, max_length=self.max_seq_length, truncation=True)
            tokens = [self.tokenizer._convert_id_to_token(tid) for tid in e_trunc]

            self.mlm_probability = self.encoder_mlm_probability
            text_encoder_mlm_mask = self._whole_word_mask(tokens)

            self.mlm_probability = self.decoder_mlm_probability
            mask_set = []
            for _ in range(min(len(tokens), 128)):
                mask_set.append(self._whole_word_mask(tokens))

            text_matrix_attention_mask = []
            for i in range(len(tokens)):
                idx = random.randint(0, min(len(tokens), 128) - 1)
                text_decoder_mlm_mask = deepcopy(mask_set[idx])
                text_decoder_mlm_mask[i] = 1
                text_matrix_attention_mask.append(text_decoder_mlm_mask)

            input_ids_batch.append(torch.tensor(e_trunc))
            attention_mask_batch.append(torch.tensor([1] * len(e_trunc)))
            e_trunc[0] = -100
            e_trunc[-1] = -100
            decoder_labels_batch.append(torch.tensor(e_trunc))

            encoder_mlm_mask_batch.append(torch.tensor(text_encoder_mlm_mask))
            decoder_matrix_attention_mask_batch.append(1 - torch.tensor(text_matrix_attention_mask))

        input_ids_batch = tensorize_batch(input_ids_batch, self.tokenizer.pad_token_id)
        attention_mask_batch = tensorize_batch(attention_mask_batch, 0)
        origin_input_ids_batch = input_ids_batch.clone()
        encoder_mlm_mask_batch = tensorize_batch(encoder_mlm_mask_batch, 0)
        encoder_input_ids_batch, encoder_labels_batch = self.torch_mask_tokens(input_ids_batch, encoder_mlm_mask_batch)
        decoder_labels_batch = tensorize_batch(decoder_labels_batch, -100)
        matrix_attention_mask_batch = tensorize_batch(decoder_matrix_attention_mask_batch, 0)

        batch = {
            "encoder_input_ids": encoder_input_ids_batch,
            "encoder_attention_mask": attention_mask_batch,
            "encoder_labels": encoder_labels_batch,
            "decoder_input_ids": origin_input_ids_batch,
            "decoder_attention_mask": matrix_attention_mask_batch,  # [B,L,L]
            "decoder_labels": decoder_labels_batch,
        }

        return batch
