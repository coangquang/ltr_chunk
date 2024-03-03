import os
import time
import json
import torch
import pandas as pd
import faiss
from datasets import load_dataset
from .model import SharedBiEncoder
from .util import get_tokenizer, query_trans, context_trans
from .preprocess import tokenise, preprocess_question
from pyvi.ViTokenizer import tokenize

class BiRetriever():
    def __init__(self, args, encoder=None, biencoder=None, save_type="dpr"):
        start = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        if self.args.new_data:
            self.train_file = "ttrain_all.csv"
            self.test_file = "ttest_all.csv"
            self.val_file = "tval_all.csv"
        else:
            self.train_file = "ttrain.csv"
            self.test_file = "ttest.csv"
            self.val_file = "tval.csv"
        self.save_type = save_type
        self.dpr_tokenizer = get_tokenizer(self.args.BE_checkpoint)
        if biencoder is not None:
            self.biencoder = biencoder
        elif encoder is not None:
            self.biencoder = SharedBiEncoder(model_checkpoint=self.args.BE_checkpoint,
                                       encoder=encoder,
                                       representation=self.args.BE_representation,
                                       fixed=self.args.bi_fixed)
        else:
            self.biencoder = SharedBiEncoder(model_checkpoint=self.args.biencoder_path,
                                            representation=self.args.BE_representation,
                                            fixed=self.args.bi_fixed)
            #self.biencoder.load_state_dict(torch.load(self.args.biencoder_path))
            
        self.biencoder.to(self.device)
        self.encoder = self.biencoder.get_model()
        self.corpus = load_dataset("csv", data_files=self.args.corpus_file, split = 'train')
        if self.args.index_path:
            self.corpus.load_faiss_index('embeddings', self.args.index_path)
        else:
            self.corpus = self.get_index()
        end = time.time()
        print(end - start)
        
    def get_index(self):
        self.encoder.to("cuda").eval()
        with torch.no_grad():
            corpus_with_embeddings = self.corpus.map(lambda example: {'embeddings': self.encoder.get_representation(self.dpr_tokenizer.encode_plus(context_trans(example["tokenized_text"], self.dpr_tokenizer),
                                                                                                                                                       padding='max_length',
                                                                                                                                                       truncation=True,
                                                                                                                                                       max_length=self.args.ctx_len,
                                                                                                                                                       return_tensors='pt')['input_ids'].to(self.device),
                                                                                                                        self.dpr_tokenizer.encode_plus(context_trans(example["tokenized_text"], self.dpr_tokenizer),
                                                                                                                                                       padding='max_length',
                                                                                                                                                       truncation=True,
                                                                                                                                                       max_length=self.args.ctx_len,
                                                                                                                                                       return_tensors='pt')['attention_mask'].to(self.device))[0].to('cpu').numpy()})
        corpus_with_embeddings.add_faiss_index(column='embeddings', metric_type=faiss.METRIC_INNER_PRODUCT)
        index_path = self.args.biencoder_path.split("/")[-1]
        index_path = "outputs/index/index_"+ self.save_type + ".faiss"
        corpus_with_embeddings.save_faiss_index('embeddings', index_path)
        return corpus_with_embeddings
    
    def retrieve(self, question, top_k=100, segmented = False):
        start = time.time()
        self.encoder.to(self.device).eval()
        
        if segmented:
            tokenized_question = query_trans(question, self.dpr_tokenizer)
        else:
             tokenized_question = query_trans(tokenise(preprocess_question(question, remove_end_phrase=False), tokenize), self.dpr_tokenizer)

        with torch.no_grad():
            Q = self.dpr_tokenizer.encode_plus(tokenized_question, padding='max_length', truncation=True, max_length=self.args.q_len, return_tensors='pt')
            question_embedding = self.encoder.get_representation(Q['input_ids'].to(self.device),
                                                                   Q['attention_mask'].to(self.device))[0].to('cpu').numpy()
            scores, retrieved_examples = self.corpus.get_nearest_examples('embeddings', question_embedding, k=top_k)
            retrieved_ids = retrieved_examples['id']   
        end = time.time()
        #print(end - start)
        return retrieved_ids, scores
    
    def test_on_data(self, top_k =[100], segmented = True, train= True):
        result = []  
        dtest = pd.read_csv(os.path.join(self.args.data_dir, self.test_file))
        dval = pd.read_csv(os.path.join(self.args.data_dir, self.val_file))
        if train:
            dtrain = pd.read_csv(os.path.join(self.args.data_dir, self.train_file))
            train_retrieved = self.retrieve_on_data(dtrain, name = 'train', top_k= max(top_k),segmented=segmented)
        test_retrieved = self.retrieve_on_data(dtest, name = 'test', top_k= max(top_k), segmented=segmented)
        val_retrieved = self.retrieve_on_data(dval, name = 'val', top_k= max(top_k),segmented=segmented)
        
        for k in top_k:
            rlt = {}
            strk = str(k)
            rlt[strk] = {}
            test_retrieved_k = [x[:k] for x in test_retrieved]
            val_retrieved_k = [x[:k] for x in val_retrieved]
            
            print("Testing hit scores with top_{}:".format(k))
            val_hit_acc, val_all_acc = self.calculate_score(dval, val_retrieved_k)
            rlt[strk]['val_hit'] = val_hit_acc
            rlt[strk]['val_all'] = val_all_acc
            print("\tVal hit acc: {:.4f}%".format(val_hit_acc*100))
            print("\tVal all acc: {:.4f}%".format(val_all_acc*100))
            test_hit_acc, test_all_acc = self.calculate_score(dtest, test_retrieved_k)
            rlt[strk]['test_hit'] = test_hit_acc
            rlt[strk]['test_all'] = test_all_acc
            print("\tTest hit acc: {:.4f}%".format(test_hit_acc*100))
            print("\tTest all acc: {:.4f}%".format(test_all_acc*100))
            result.append(rlt)
        #name = self.args.biencoder_path.split("/")
        save_file = "outputs/testdpr_"+ self.save_type + ".json" 
        with open(save_file, 'w') as f:
            json.dump(result, f, ensure_ascii = False, indent =4)
            
    def retrieve_with_result(self, df, name, top_k=[100], segmented=False):
        result = []
        retrieved = self.retrieve_on_data(df, name, top_k=max(top_k), segmented=segmented)
            
        for k in top_k:
            rlt = {}
            strk = str(k)
            rlt[strk] = {}
            retrieved_k = [x[:k] for x in retrieved]
            
            print("Testing hit scores with top_{}:".format(k))
            hit_acc, all_acc = self.calculate_score(df, retrieved_k)
            rlt[strk]['hit'] = hit_acc
            rlt[strk]['all'] = all_acc
            print("\tHit acc: {:.4f}%".format(hit_acc*100))
            print("\tAll acc: {:.4f}%".format(all_acc*100))
            result.append(rlt)     
              
    def retrieve_on_data(self, df, name, top_k = 100, segmented = False, saved=True):
        count = 0
        acc = 0
        retrieved_list = []
        #retrieved_sub_list = []
        if not segmented:
            tokenized_questions = []
            for i in range(len(df)):
                tokenized_question = tokenise(preprocess_question(df['question'][i], remove_end_phrase=False), tokenize)
                tokenized_questions.append(tokenized_question)
            df['tokenized_question'] = tokenized_questions
            
        for i in range(len(df)):
            tokenized_question = df['tokenized_question'][i]
            retrieved_ids, _ = self.retrieve(tokenized_question, top_k, segmented=True)
            retrieved_list.append(retrieved_ids)

        if saved:
            save_file = "outputs/" + self.save_type + "_" + name + "_retrieved.json" 
            with open(save_file, 'w') as f:
                json.dump(retrieved_list, f, ensure_ascii = False, indent =4)
        return retrieved_list
    
    def find_neg(self, df, name, no_negs=3, segmented=True):
        retrieved_list = self.retrieve_on_data(df, name, 100, segmented, saved=False)
        
        ttokenized_ques = df['tokenized_question'].tolist()
        tans_id = df['ans_id'].tolist()
        tnew_neg = []
        tbest_ans_id = df['best_ans_id'].tolist()
        
        nbest_ans_id = []
        
        for i in range(len(df)):
            retrieved_ids = retrieved_list[i]
            ans_idss = json.loads(tans_id[i])
            tbest_ans_idss = json.loads(tbest_ans_id[i])
            ans_ids = []
            nbest_ans_ids = []
            for j in range(len(ans_idss)):
                a_ids = ans_idss[j]
                tbest_a_id = tbest_ans_idss[j]
                ans_ids += a_ids
                found = True
                ij = 0
                while (found and ij < 100):
                    if retrieved_ids[ij] in a_ids:
                        nbest_ans_ids.append(retrieved_ids[ij]) 
                        found = False
                    ij += 1
                if found:
                    nbest_ans_ids.append(tbest_a_id)
                        
            new_neg_ids = [x for x in retrieved_ids if x not in ans_ids]# and x not in kept_neg_ids]
            new_neg_ids = new_neg_ids[:no_negs]
            nbest_ans_id.append(nbest_ans_ids)
            tnew_neg.append(new_neg_ids) 
            
        dn = pd.DataFrame()
        dn['tokenized_question'] = ttokenized_ques
        dn['ans_id'] = tans_id
        dn['best_ans_id'] = nbest_ans_id
        dn['neg_ids'] = tnew_neg
        
        dt = pd.DataFrame()
        dt['tokenized_question'] = ttokenized_ques
        dt['ans_id'] = tans_id
        dt['best_ans_id'] = tbest_ans_id
        dt['neg_ids'] = tnew_neg
        return dt, dn
    
    def increase_neg(self, no_negs=3, segmented=True):
        dtrain = pd.read_csv(os.path.join(self.args.data_dir, self.train_file))
        dval = pd.read_csv(os.path.join(self.args.data_dir, self.val_file))
        dtest = pd.read_csv(os.path.join(self.args.data_dir, self.test_file))
        
        dttrain, dntrain = self.find_neg(dtrain, "train", no_negs, segmented)
        dtval, dnval = self.find_neg(dval, "val", no_negs, segmented)
        dttest, dntest = self.find_neg(dtest, "test", no_negs, segmented)
        
        dttrain.to_csv("outputs/data/{}/old/{}".format(self.save_type, self.train_file), index=False)
        dtval.to_csv("outputs/data/{}/old/{}".format(self.save_type, self.val_file), index=False)
        dttest.to_csv("outputs/data/{}/old/{}".format(self.save_type, self.test_file), index=False)
        
        dntrain.to_csv("outputs/data/{}/new/{}".format(self.save_type, self.train_file), index=False)
        dnval.to_csv("outputs/data/{}/new/{}".format(self.save_type, self.val_file), index=False)
        dntest.to_csv("outputs/data/{}/new/{}".format(self.save_type, self.test_file), index=False)
    
    def calculate_score(self, df, retrieved_list):
        top_k = len(retrieved_list[0])
        all_count = 0
        hit_count = 0
        for i in range(len(df)):
            all_check = True
            hit_check = False
            retrieved_ids = retrieved_list[i]
            ans_ids = json.loads(df['ans_id'][i])
            for a_ids in ans_ids:
                com = [a_id for a_id in a_ids if a_id in retrieved_ids]
                if len(com) > 0:
                    hit_check = True
                else:
                    all_check = False
            
            if hit_check:
                hit_count += 1
            if all_check:
                all_count += 1
                
        all_acc = all_count/len(df)
        hit_acc = hit_count/len(df)
        return hit_acc, all_acc
