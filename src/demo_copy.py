import faiss
import torch
import logging
import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from transformers import AutoTokenizer
from bi.model import SharedBiEncoder
from bi.preprocess import preprocess_question
from cross_rerank.model import RerankerForInference
from pyvi.ViTokenizer import tokenize
#import gradio as gr
import streamlit as st
logger = logging.getLogger(__name__)


@dataclass
class Args:
    encoder: str = field(
        default="/kaggle/input/add3-kd8",
        metadata={'help': 'The encoder name or path.'}
    )
    cross_checkpoint: str = field(
        default="/kaggle/input/add3-cross1750",
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
    index_factory: str = field(
        default="Flat",
        metadata={'help': 'Faiss index factory.'}
    )
    k: int = field(
        default=30,
        metadata={'help': 'How many neighbors to retrieve?'}
    )
    top_k: int = field(
        default=10,
        metadata={'help': 'How many neighbors to retrieve?'}
    )
    
    batch_size: int = field(
        default=1024,
        metadata={'help': 'Inference batch size.'}
    )
    
    corpus_file: str = field(
        default="/kaggle/input/zalo-data/zalo_corpus.csv",
        metadata={'help': 'Path to zalo corpus.'}
    )
    
    save_embedding: bool = field(
        default=False,
        metadata={'help': 'Save embeddings in memmap at save_dir?'}
    )
    load_embedding: str = field(
        default='/kaggle/input/add3-kd8/embeddings.memmap',
        metadata={'help': 'Path to saved embeddings.'}
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
    if load_embedding != '':
        test_tokens = tokenizer(['test'],
                                padding=True,
                                truncation=True,
                                max_length=128,
                                return_tensors="pt").to('cuda')
        test = model.encoder.get_representation(test_tokens['input_ids'], test_tokens['attention_mask'])
        test = test.cpu().numpy()
        dtype = test.dtype
        dim = test.shape[-1]

        all_embeddings = np.memmap(
            load_embedding,
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
                shape=all_embeddings.shape,
                mode="w+",
                dtype=all_embeddings.dtype
            )

            length = all_embeddings.shape[0]
            # add in batch
            save_batch_size = 10000
            if length > save_batch_size:
                for i in tqdm(range(0, length, save_batch_size), leave=False, desc="Saving Embeddings"):
                    j = min(i + save_batch_size, length)
                    memmap[i: j] = all_embeddings[i: j]
            else:
                memmap[:] = all_embeddings
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


def search(model: SharedBiEncoder, tokenizer:AutoTokenizer, question, faiss_index: faiss.Index, k:int = 100, max_length: int=128):
    """
    1. Encode queries into dense embeddings;
    2. Search through faiss index
    """
    #model.to('cuda')
    q_embeddings = []
    #questions = queries['tokenized_question'].tolist()
    #questions = [process_query(x) for x in questions]
    #for start_index in tqdm(range(0, len(questions), batch_size), desc="Inference Embeddings",
    #                        disable=len(questions) < batch_size):
                    
    q_collated = tokenizer(
                [question],
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

    #for i in tqdm(range(0, query_size, batch_size), desc="Searching"):
    #    j = min(i + batch_size, query_size)
    #    q_embedding = q_embeddings[i: j]
    score, indice = faiss_index.search(q_embeddings.astype(np.float32), k=k)
    all_scores.append(score)
    all_indices.append(indice)
    
    all_scores = np.concatenate(all_scores, axis=0)
    all_indices = np.concatenate(all_indices, axis=0)
    return all_scores, all_indices

def rerank(reranker: SharedBiEncoder, tokenizer:AutoTokenizer, question, corpus, retrieved_ids, max_length = 256, top_k=30):
    eos = tokenizer.eos_token
    texts = []
    for j in range(top_k):
        texts.append(question + eos + eos + corpus[retrieved_ids[j]])
    collated = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to('cuda')
    reranked_scores = reranker(collated).logits
    reranked_scores = reranked_scores.view(-1,top_k).to('cpu').tolist()
    tuple_lst = [(retrieved_ids[n], reranked_scores[0][n]) for n in range(top_k)]
    tuple_lst.sort(key=lambda tup: tup[1], reverse=True)
    reranked_ids = [tup[0] for tup in tuple_lst]
    rerank_scores = [tup[1] for tup in tuple_lst]            
            
    return reranked_ids, rerank_scores

                
def app():
    parser = HfArgumentParser([Args])
    args: Args = parser.parse_args_into_dataclasses()[0]
    print(args)
    model = SharedBiEncoder(model_checkpoint=args.encoder,
                            representation=args.sentence_pooling_method,
                            fixed=True)
    model.to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer if args.tokenizer else args.encoder)
    corpus_data = pd.read_csv(args.corpus_file)
    corpus = corpus_data['tokenized_text'].tolist()
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
    reranker = RerankerForInference(model_checkpoint=args.cross_checkpoint)
    reranker.to('cuda')
    reranker_tokenizer = AutoTokenizer.from_pretrained(args.cross_checkpoint)
    
    st.header("Vietnamese Legal Retriever Web App")
    st.subheader("Powered by NguyenNhatQuang")
    option = st.selectbox(
    "Select your retrieval system?",
    ("Bi-encoder only", "Bi-encoder + Cross-encoder Re-ranker"))

    st.write("System selected:", option)
    user_input = st.text_area(
        "Enter your legal query/question below and click the button to submit."
    )
    def bi_answer(org_question):
        start = time.time()
        question = tokenize(preprocess_question(org_question, remove_end_phrase=False))
        scores, indices = search(
            model=model, 
            tokenizer=tokenizer,
            question=question, 
            faiss_index=faiss_index, 
            k=args.k, 
            max_length=args.max_query_length
        )
        
        indice = indices[0]
        score = scores[0]
        timee = time.time - start
        chunks = []
        for i in range(args.k):
            x = indice[i]
            chunk = {}
            chunk['bi_score'] = float(score[i])
            chunk['id'] = int(x)
            chunk['law_id'] = corpus_data['law_id'][x]
            chunk['article_id'] = int(corpus_data['article_id'][x])
            chunk['title'] = corpus_data['title'][x]
            chunk['text'] = corpus_data['text'][x]
            chunks.append(chunk)
        
        rst = {}
        rst['question'] = org_question
        rst['top_relevant_chunks'] = chunks

        with open("result-bi.json", 'w') as f:
            json.dump(rst, f, ensure_ascii=False, indent=4)
        return rst, timee
    
    def cross_answer(org_question, k=30, top_k=10):
        start = time.time()
        question = tokenize(preprocess_question(org_question, remove_end_phrase=False))
        scores, indices = search(
            model=model, 
            tokenizer=tokenizer,
            question=question, 
            faiss_index=faiss_index, 
            k=k, 
            max_length=256
        )
        indice = indices[0]
        score = scores[0]
        retrieval_ids = indice
        
        rerank_ids, rerank_scores = rerank(reranker, reranker_tokenizer, question, corpus, retrieval_ids, 256, top_k)
        timee = time.time -start
        indice = indice.tolist()
        chunks = []
        for i in range(top_k):
            x = rerank_ids[i]
            chunk = {}
            chunk['rerank_score'] = float(rerank_scores[i])
            chunk['bi_score'] = float(score[indice.index(x)])
            chunk['id'] = int(x)
            chunk['law_id'] = corpus_data['law_id'][x]
            chunk['article_id'] = int(corpus_data['article_id'][x])
            chunk['title'] = corpus_data['title'][x]
            chunk['text'] = corpus_data['text'][x]
            chunks.append(chunk)
        
        rst = {}
        rst['question'] = org_question
        rst['top_relevant_chunks'] = chunks

        with open("result-cross.json", 'w') as f:
            json.dump(rst, f, ensure_ascii=False, indent=4)
        return rst, timee

    with st.form("my_form"):
        submit = st.form_submit_button(label="Search")

    if submit:
        if option == 'Bi-encoder only':
            ans, timee = bi_answer(user_input)
        else:
            ans, timee = cross_answer(user_input)
        st.write("Retrieval Time:", timee, "s.")
        st.write(ans)

    

if __name__ == "__main__":
    app()