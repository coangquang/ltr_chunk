import faiss
import torch
import logging
import datasets
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from transformers import AutoTokenizer
from model import SharedBiEncoder
#from src.process import process_query, process_text, concat_str
import itertools
#from pandarallel import pandarallel

logger = logging.getLogger(__name__)


@dataclass
class Args:
    encoder: str = field(
        default="vinai/phobert-base-v2",
        metadata={'help': 'The encoder name or path.'}
    )
    tokenizer: str = field(
        default=None,
        metadata={'help': 'The encoder name or path.'}
    )
    sentence_pooling_method: str = field(
        default="cls",
        metadata={'help': 'Embedding method'}
    )
    fp16: bool = field(
        default=False,
        metadata={'help': 'Use fp16 in inference?'}
    )
    max_query_length: int = field(
        default=32,
        metadata={'help': 'Max query length.'}
    )
    max_passage_length: int = field(
        default=256,
        metadata={'help': 'Max passage length.'}
    )
    batch_size: int = field(
        default=128,
        metadata={'help': 'Inference batch size.'}
    )
    index_factory: str = field(
        default="Flat",
        metadata={'help': 'Faiss index factory.'}
    )
    k: int = field(
        default=1000,
        metadata={'help': 'How many neighbors to retrieve?'}
    )
    data_path: str = field(
        default="/kaggle/input/zalo-data",
        metadata={'help': 'Path to zalo data.'}
    )
    data_type: str = field(
        default="test",
        metadata={'help': 'Type data to test'}
    )
    save_embedding: bool = field(
        default=False,
        metadata={'help': 'Save embeddings in memmap at save_dir?'}
    )
    load_embedding: bool = field(
        default=False,
        metadata={'help': 'Load embeddings from save_dir?'}
    )
    save_path: str = field(
        default="embeddings.memmap",
        metadata={'help': 'Path to save embeddings.'}
    )

def index(model: SharedBiEncoder, tokenizer:AutoTokenizer, corpus, batch_size: int = 16, max_length: int=512, index_factory: str = "Flat", save_path: str = None, save_embedding: bool = False, load_embedding: bool = False):
    """
    1. Encode the entire corpus into dense embeddings; 
    2. Create faiss index; 
    3. Optionally save embeddings.
    """
    if load_embedding:
        test = model.encode("test")
        dtype = test.dtype
        dim = len(test)

        corpus_embeddings = np.memmap(
            save_path,
            mode="r",
            dtype=dtype
        ).reshape(-1, dim)
    
    else:
        #df_corpus = pd.DataFrame()
        #df_corpus['text'] = corpus
        #pandarallel.initialize(progress_bar=True, use_memory_fs=False, nb_workers=12)
        #df_corpus['processed_text'] = df_corpus['text'].parallel_apply(process_text)
        #processed_corpus = df_corpus['processed_text'].tolist()
        #model.to('cuda')
        all_embeddings = []
        for start_index in tqdm(range(0, len(corpus), batch_size), desc="Inference Embeddings",
                                disable=len(corpus) < batch_size):
            passages_batch = corpus[start_index:start_index + batch_size]
            d_collated = tokenizer(
                    passages_batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to('cuda')

            with torch.no_grad():
                corpus_embeddings = model.encoder.get_representation(d_collated['input_ids'], d_collated['attention_mask']) 
            
            corpus_embeddings = corpus_embeddings.cpu().numpy()
            all_embeddings.append(corpus_embeddings)

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        dim = all_embeddings.shape[-1]
        
        if save_embedding:
            logger.info(f"saving embeddings at {save_path}...")
            memmap = np.memmap(
                save_path,
                shape=corpus_embeddings.shape,
                mode="w+",
                dtype=corpus_embeddings.dtype
            )

            length = corpus_embeddings.shape[0]
            # add in batch
            save_batch_size = 10000
            if length > save_batch_size:
                for i in tqdm(range(0, length, save_batch_size), leave=False, desc="Saving Embeddings"):
                    j = min(i + save_batch_size, length)
                    memmap[i: j] = corpus_embeddings[i: j]
            else:
                memmap[:] = corpus_embeddings
    # create faiss index
    faiss_index = faiss.index_factory(dim, index_factory, faiss.METRIC_INNER_PRODUCT)

    #if model.device == torch.device("cuda"):
    if True:
        co = faiss.GpuClonerOptions()
        #co = faiss.GpuMultipleClonerOptions()
        #co.useFloat16 = True
        faiss_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss_index, co)
        #faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co)

    # NOTE: faiss only accepts float32
    logger.info("Adding embeddings...")
    all_embeddings = all_embeddings.astype(np.float32)
    #print(all_embeddings[0])
    faiss_index.train(all_embeddings)
    faiss_index.add(all_embeddings)
    return faiss_index


def search(model: SharedBiEncoder, tokenizer:AutoTokenizer, queries: pd.DataFrame, faiss_index: faiss.Index, k:int = 100, batch_size: int = 256, max_length: int=128):
    """
    1. Encode queries into dense embeddings;
    2. Search through faiss index
    """
    #model.to('cuda')
    q_embeddings = []
    questions = queries['tokenized_question'].tolist()
    #questions = [process_query(x) for x in questions]
    for start_index in tqdm(range(0, len(questions), batch_size), desc="Inference Embeddings",
                            disable=len(questions) < batch_size):
                    
        q_collated = tokenizer(
                    questions[start_index: start_index + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                ).to('cuda')

        with torch.no_grad():
            query_embeddings = model.encoder.get_representation(q_collated['input_ids'], q_collated['attention_mask'])
        query_embeddings = query_embeddings.cpu().numpy()
        q_embeddings.append(query_embeddings)
    
    q_embeddings = np.concatenate(q_embeddings, axis=0)
    query_size = q_embeddings.shape[0]
    all_scores = []
    all_indices = []

    for i in tqdm(range(0, query_size, batch_size), desc="Searching"):
        j = min(i + batch_size, query_size)
        q_embedding = q_embeddings[i: j]
        score, indice = faiss_index.search(q_embedding.astype(np.float32), k=k)
        all_scores.append(score)
        all_indices.append(indice)
    
    all_scores = np.concatenate(all_scores, axis=0)
    all_indices = np.concatenate(all_indices, axis=0)
    return all_scores, all_indices
    
def evaluate(preds, labels, cutoffs=[1,5,10,30,100]):
    """
    Evaluate MRR and Recall at cutoffs.
    """
    metrics = {}
    
    # MRR
    mrrs = np.zeros(len(cutoffs))
    for pred, label in zip(preds, labels):
        jump = False
        for i, x in enumerate(pred, 1):
            if x in label:
                for k, cutoff in enumerate(cutoffs):
                    if i <= cutoff:
                        mrrs[k] += 1 / i
                jump = True
            if jump:
                break
    mrrs /= len(preds)
    for i, cutoff in enumerate(cutoffs):
        mrr = mrrs[i]
        metrics[f"MRR@{cutoff}"] = mrr

    # Recall
    recalls = np.zeros(len(cutoffs))
    for pred, label in zip(preds, labels):
        for k, cutoff in enumerate(cutoffs):
            recall = np.intersect1d(label, pred[:cutoff])
            recalls[k] += len(recall) / len(label)
    recalls /= len(preds)
    for i, cutoff in enumerate(cutoffs):
        recall = recalls[i]
        metrics[f"Recall@{cutoff}"] = recall

    return metrics

def calculate_score(df, retrieved_list):
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

def check(df, retrieved_list, cutoffs=[1,5,10,30,100]):
    metrics = {}
    for cutoff in cutoffs:
        retrieved_k = [x[:cutoff] for x in retrieved_list]
        hit_acc, all_acc = calculate_score(df, retrieved_k)
        metrics[f"All@{cutoff}"] = all_acc
        metrics[f"Hit@{cutoff}"] = hit_acc
    return metrics
    

def main():
    parser = HfArgumentParser([Args])
    args: Args = parser.parse_args_into_dataclasses()[0]
    print(args)
    model = SharedBiEncoder(model_checkpoint=args.encoder,
                            representation=args.sentence_pooling_method,
                            fixed=True)
    model.to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer if args.tokenizer else args.encoder)
    
    if args.data_type == 'eval':
        test_data = pd.read_csv(args.data_path + "/tval.csv") 
    elif args.data_type == 'train':
        test_data = pd.read_csv(args.data_path + "/ttrain.csv")
    elif args.data_type == 'all':
        data1 = pd.read_csv(args.data_path + "/ttrain.csv")
        data2 = pd.read_csv(args.data_path + "/ttest.csv")
        data3 = pd.read_csv(args.data_path + "/tval.csv")
        test_data = pd.concat([data1, data3, data2], ignore_index=True)
          
    else:
        test_data = pd.read_csv(args.data_path + "/ttest.csv")
    corpus_data = pd.read_csv(args.data_path + "/zalo_corpus.csv")
    #dcorpus = pd.DataFrame(corpus_data)
    #pandarallel.initialize(progress_bar=True, use_memory_fs=False, nb_workers=12)
    #dcorpus["full_text"] = dcorpus.parallel_apply(concat_str, axis=1)
    corpus = corpus_data['tokenized_text'].tolist()
    
    #ans_text_ids = [json.loads(test_data['best_ans_text_id'][i]) for i in range(len(test_data))]
    ans_text_ids = []
    for i in range(len(test_data)):
        print(test_data['best_ans_text_id'][i])
        ans_text_ids.append(json.loads(i))
    #ans_text_ids = [json.loads(sample) for sample in ans_text_ids]
    ground_truths = []
    for sample in ans_text_ids:
        temp = ["_".join(y.split("_")[:2]) for y in sample]
        ground_truths.append(temp)
    
    faiss_index = index(
        model=model, 
        tokenizer=tokenizer,
        corpus=corpus, 
        batch_size=args.batch_size,
        max_length=args.max_passage_length,
        index_factory=args.index_factory,
        save_path=args.save_path,
        save_embedding=args.save_embedding,
        load_embedding=args.load_embedding
    )
    
    scores, indices = search(
        model=model, 
        tokenizer=tokenizer,
        queries=test_data, 
        faiss_index=faiss_index, 
        k=args.k, 
        batch_size=args.batch_size, 
        max_length=args.max_query_length
    )

    #print(len(indices))

    retrieval_results, retrieval_ids = [], []
    for indice in indices:
        # filter invalid indices
        indice = indice[indice != -1].tolist()
        rst = []
        for x in indice:
            temp = corpus_data['law_id'][x] + "_" + corpus_data['article_id'][x]
            if temp not in rst:
                rst.append(temp)
        retrieval_results.append(rst)
        retrieval_ids.append(indice)
    
    metrics = check(test_data, retrieval_ids)
    print(metrics)
    metrics = evaluate(retrieval_results, ground_truths)
    print(metrics)


if __name__ == "__main__":
    main()