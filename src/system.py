import os
import time
import json
import torch
import pandas as pd
from dpr.util import get_tokenizer
from dpr.preprocess import tokenise, preprocess_question
from pyvi.ViTokenizer import tokenize
from dpr.retriever import DPRRetriever
from rerank.model import CrossEncoder

class Retriever():
    def __init__(self, 
                 args, 
                 q_encoder=None, 
                 ctx_encoder=None, 
                 biencoder=None, 
                 cross_encoder=None,
                 save_type="system"):
        start = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.save_type = save_type
        self.dpr_tokenizer = get_tokenizer(self.args.BE_checkpoint)
        
        if biencoder is not None:
            self.dpr = DPRRetriever(self.args, biencoder=biencoder, sub=True)
        elif q_encoder is not None and ctx_encoder is not None:
            self.dpr = DPRRetriever(self.args, q_encoder=q_encoder, ctx_encoder=ctx_encoder, sub=True)
        else:
            self.dpr = DPRRetriever(self.args, sub=True)
            
        #self.dpr.to(self.device)
        self.q_encoder, self.ctx_encoder = self.dpr.biencoder.get_models()
        self.corpus = self.dpr.corpus
        if cross_encoder is not None:
            self.cross_encoder = cross_encoder
        else:
            self.cross_encoder = CrossEncoder(model_checkpoint=self.args.cross_checkpoint,
                                            representation=self.args.cross_representation,
                                            dropout=self.args.cross_dropout,
                                            device=self.device)
        
        if self.args.cross_load_path:
            self.cross_encoder.load_state_dict(torch.load(self.args.cross_load_path))
        self.cross_encoder.to(self.device)
        
        end = time.time()
        print(end - start)
    
    def retrieve(self, question, top_k=30, segmented = True):
        start = time.time()
        self.q_encoder.to(self.device).eval()
        
        if segmented:
            tokenized_question = question
        else:
             tokenized_question = tokenise(preprocess_question(question, remove_end_phrase=False), tokenize)
        retrieved_ids, retrieved_sub_ids, dpr_scores = self.dpr.retrieve(tokenized_question, top_k=top_k, segmented=True)
        cross_samples = [(tokenized_question, self.corpus['tokenized_text'][retrieved_id]) for retrieved_id in retrieved_sub_ids]

        cross_scores = self.cross_encoder.predict(cross_samples, max_len=self.args.cross_len)
        
        rerank_list = [(retrieved_ids[i], retrieved_sub_ids[i], self.args.cross_ratio*float(cross_scores[i][0].item()) + float(self.args.dpr_ratio*dpr_scores[i]) ) for i in range(top_k)]
        sorted_rerank_list = sorted(rerank_list, key=lambda x: x[2], reverse=True)   
        rerank_ids = [x[0] for x in sorted_rerank_list]
        rerank_sub_ids = [x[1] for x in sorted_rerank_list]
        rerank_scores = [x[2] for x in sorted_rerank_list]
        end = time.time()
        #print(end - start)
        return rerank_ids, rerank_sub_ids, rerank_scores
    
    def test_on_data(self, top_k =[30], segmented = True, train= False, test= False, val= False):
        result = []
        if train:
            dtrain = pd.read_csv(os.path.join(self.args.data_dir, 'ttrain.csv'))
            train_retrieved, train_sub_retrieved = self.retrieve_on_data(dtrain, name = 'train', top_k= max(top_k),segmented=segmented)
        if test:
            dtest = pd.read_csv(os.path.join(self.args.data_dir, 'ttest.csv'))
            test_retrieved, test_sub_retrieved = self.retrieve_on_data(dtest, name = 'test', top_k= max(top_k), segmented=segmented)
        if val:
            dval = pd.read_csv(os.path.join(self.args.data_dir, 'tval.csv'))
            val_retrieved, val_sub_retrieved = self.retrieve_on_data(dval, name = 'val', top_k= max(top_k),segmented=segmented)
        
        
        for k in top_k:
            rlt = {}
            strk = str(k)
            rlt[strk] = {}
            print("Testing hit scores with top_{}:".format(k))
            if test:
                test_retrieved_k = [x[:k] for x in test_retrieved]
                test_hit_acc, test_all_acc = self.calculate_score(dtest, test_retrieved_k)
                rlt[strk]['test_hit'] = test_hit_acc
                rlt[strk]['test_all'] = test_all_acc
                print("\tTest hit acc: {:.4f}%".format(test_hit_acc*100))
                print("\tTest all acc: {:.4f}%".format(test_all_acc*100))
            if val:
                val_retrieved_k = [x[:k] for x in val_retrieved]
                val_hit_acc, val_all_acc = self.calculate_score(dval, val_retrieved_k)
                rlt[strk]['val_hit'] = val_hit_acc
                rlt[strk]['val_all'] = val_all_acc
                print("\tVal hit acc: {:.4f}%".format(val_hit_acc*100))
                print("\tVal all acc: {:.4f}%".format(val_all_acc*100))
            
            result.append(rlt)
        save_file = "outputs/testdpr_"+ self.save_type + ".json" 
        with open(save_file, 'w') as f:
            json.dump(result, f, ensure_ascii = False, indent =4)
        
    def retrieve_on_data(self, df, name, top_k = 30, segmented = False):
        count = 0
        acc = 0
        retrieved_list = []
        retrieved_sub_list = []
        if not segmented:
            tokenized_questions = []
            for i in range(len(df)):
                tokenized_question = tokenise(preprocess_question(df['question'][i], remove_end_phrase=False), tokenize)
                tokenized_questions.append(tokenized_question)
            df['tokenized_question'] = tokenized_questions
            
          
        for i in range(len(df)):
            tokenized_question = df['tokenized_question'][i]
            retrieved_ids, retrieved_sub_ids, _ = self.retrieve(tokenized_question, top_k, segmented=True)
            retrieved_list.append(retrieved_ids)
            retrieved_sub_list.append(retrieved_sub_ids)

        save_file = "outputs/" + self.save_type + "_" + name + "_retrieved.json" 
        sub_save_file = "outputs/" + self.save_type + "_" + name + "_sub_retrieved.json"
        with open(save_file, 'w') as f:
            json.dump(retrieved_list, f, ensure_ascii = False, indent =4)
        with open(sub_save_file, 'w') as f:
            json.dump(retrieved_sub_list, f, ensure_ascii = False, indent =4)
        return retrieved_list, retrieved_sub_list
    
    def calculate_score(self, df, retrieved_list):
        top_k = len(retrieved_list[0])
        all_count = 0
        hit_count = 0
        for i in range(len(df)):
            retrieved_ids = retrieved_list[i]
            ans_ids = [int(x) for x in str(df['ans_id'][i]).split(", ")]
            for a_id in ans_ids:
                if a_id in retrieved_ids:
                    retrieved_ids.remove(a_id)
            if len(retrieved_ids) == top_k - len(ans_ids):
                all_count += 1
            if len(retrieved_ids) < top_k:
                hit_count += 1
                
        all_acc = all_count/len(df)
        hit_acc = hit_count/len(df)
        return hit_acc, all_acc