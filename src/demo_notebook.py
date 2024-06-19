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
from pyvi.ViTokenizer import tokenize
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
    index_factory: str = field(
        default="Flat",
        metadata={'help': 'Faiss index factory.'}
    )
    input_query: str = field(
        default="Thủ tục bầu Chủ tịch nước là gì?",
        metadata={'help': 'Faiss index factory.'}
    )
    
    k: int = field(
        default=1000,
        metadata={'help': 'How many neighbors to retrieve?'}
    )
    
    batch_size: int = field(
        default=128,
        metadata={'help': 'Inference batch size.'}
    )
    
    corpus_file: str = field(
        default="/kaggle/input/zalo-data",
        metadata={'help': 'Path to zalo corpus.'}
    )
    
    save_embedding: bool = field(
        default=False,
        metadata={'help': 'Save embeddings in memmap at save_dir?'}
    )
    load_embedding: str = field(
        default='',
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

                
def main():
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
    
    

    def greet(org_question):
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
        indice = indice[indice != -1].tolist()
        rst = []
        chunks = []
        for x in indice:
            temp = corpus_data['law_id'][x] + "_" + str(corpus_data['article_id'][x])
            chunks.append(corpus_data['text'][x])
            if temp not in rst:
                rst.append(temp)
        #retrieval_results = rst
        #retrieval_ids = indice
        return chunks

    chunks = greet(args.input_query)
    for chunk in chunks:
        print()
        print(chunk)
    #demo = gr.Interface(fn=greet, inputs="text", outputs="text")

    #demo.launch(share=True)

if __name__ == "__main__":
    main()