import faiss
import torch
import logging
import datasets
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from transformers import AutoTokenizer
from model.nor_modeling import BiEncoderModel
from src.process import process_query, process_text, concat_str
import itertools
from pandarallel import pandarallel

logger = logging.getLogger(__name__)


@dataclass
class Args:
    encoder: str = field(
        default="output",
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
    normalized: bool = field(
        default=False,
        metadata={'help': 'Use cosine similarity of not'}
    )
    add_instruction: bool = field(
        default=False,
        metadata={'help': 'Add query-side instruction?'}
    )
    
    max_query_length: int = field(
        default=128,
        metadata={'help': 'Max query length.'}
    )
    max_passage_length: int = field(
        default=512,
        metadata={'help': 'Max passage length.'}
    )
    batch_size: int = field(
        default=64,
        metadata={'help': 'Inference batch size.'}
    )
    index_factory: str = field(
        default="Flat",
        metadata={'help': 'Faiss index factory.'}
    )
    k: int = field(
        default=100,
        metadata={'help': 'How many neighbors to retrieve?'}
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

def index(model: BiEncoderModel, tokenizer:AutoTokenizer, corpus: datasets.Dataset, batch_size: int = 16, max_length: int=512, index_factory: str = "Flat", save_path: str = None, save_embedding: bool = False, load_embedding: bool = False):
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
        df_corpus = pd.DataFrame()
        df_corpus['text'] = corpus
        pandarallel.initialize(progress_bar=True, use_memory_fs=False, nb_workers=12)
        df_corpus['processed_text'] = df_corpus['text'].parallel_apply(process_text)
        processed_corpus = df_corpus['processed_text'].tolist()
        model.to('cuda')
        all_embeddings = []
        for start_index in tqdm(range(0, len(corpus), batch_size), desc="Inference Embeddings",
                                disable=len(corpus) < 256):
            passages_batch = processed_corpus[start_index:start_index + batch_size]
            d_collated = tokenizer(
                    passages_batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to('cuda')

            with torch.no_grad():
                corpus_embeddings = model.encode(d_collated) 
            
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
        # co = faiss.GpuClonerOptions()
        co = faiss.GpuMultipleClonerOptions()
        co.useFloat16 = True
        # faiss_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss_index, co)
        faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co)

    # NOTE: faiss only accepts float32
    logger.info("Adding embeddings...")
    all_embeddings = all_embeddings.astype(np.float32)
    #print(all_embeddings[0])
    faiss_index.train(all_embeddings)
    faiss_index.add(all_embeddings)
    return faiss_index


def search(model: BiEncoderModel, tokenizer:AutoTokenizer, queries: datasets, faiss_index: faiss.Index, k:int = 100, batch_size: int = 256, max_length: int=128):
    """
    1. Encode queries into dense embeddings;
    2. Search through faiss index
    """
    model.to('cuda')
    q_embeddings = []
    questions = queries['query']
    questions = [process_query(x) for x in questions]
    for start_index in tqdm(range(0, len(questions), batch_size), desc="Inference Embeddings",
                            disable=len(questions) < 256):
                    
        q_collated = tokenizer(
                    questions[start_index: start_index + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                ).to('cuda')

        with torch.no_grad():
            query_embeddings = model.encode(q_collated)
        query_embeddings = query_embeddings.cpu().numpy()
        q_embeddings.append(query_embeddings)


    
    q_embeddings = np.concatenate(q_embeddings, axis=0)
    query_size = q_embeddings.shape[0]
    #print(q_embeddings.shape)
    #print(q_embeddings[0])
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
    
    
def evaluate(preds, labels, cutoffs=[1,10,100]):
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


def main():
    parser = HfArgumentParser([Args])
    args: Args = parser.parse_args_into_dataclasses()[0]
    print(args)
    model = BiEncoderModel(
        model_name=args.encoder,
        sentence_pooling_method=args.sentence_pooling_method,
        normlized=args.normalized
    )
    #model.to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer if args.tokenizer else args.encoder)
    eval_data = datasets.load_dataset("json", data_files="data/test.jsonl", split="train")
    corpus_data = datasets.load_dataset("json", data_files='data/corpus.jsonl', split='train')
    dcorpus = pd.DataFrame(corpus_data)
    pandarallel.initialize(progress_bar=True, use_memory_fs=False, nb_workers=12)
    dcorpus["full_text"] = dcorpus.parallel_apply(concat_str, axis=1)
    corpus = dcorpus['full_text'].tolist()
    
    #eval_data = datasets.load_dataset("namespace-Pt/msmarco", split="dev")
    #corpus = datasets.load_dataset("namespace-Pt/msmarco-corpus", split="train")
    #data = datasets.load_dataset('sentence-transformers/embedding-training-data', data_files='msmarco-triplets.jsonl.gz', split='train')
    #eval_data = data.select(range(50000))
    #print(eval_data)
    #print(eval_data['pos'])
    #questions = list(itertools.chain.from_iterable(eval_data['query'])) 
    #corpus = list(itertools.chain.from_iterable(eval_data['pos'])) 
    #corpus = []
    #ground_truths = []
    #for sample in eval_data:
    #    ground_truths.append(sample["pos"])
    #    for x in sample["pos"]:  
    #        corpus.append(x)
        
    #model = FlagModel(
    #    args.encoder, 
    #    query_instruction_for_retrieval="Represent this sentence for searching relevant passages: " if args.add_instruction else None,
    #    use_fp16=args.fp16
    #)
    
    
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
        queries=eval_data, 
        faiss_index=faiss_index, 
        k=args.k, 
        batch_size=args.batch_size, 
        max_length=args.max_query_length
    )

    #print(len(indices))

    retrieval_results = []
    for indice in indices:
        # filter invalid indices
        indice = indice[indice != -1].tolist()
        #print(indice)
        rst = [corpus[i] for i in indice]
        retrieval_results.append(rst)
        #retrieval_results.append(corpus[indice])

    ground_truths = []
    for sample in eval_data:
        ground_truths.append(sample["pos"])

    #for i in range(10):
    #    print(i, eval_data['query'][i])
    #    print(corpus.index(ground_truths[i][0]))
    #    print(indices[i][:10])
    #    print(scores[i][:10])
    #    print(ground_truths[i])
    #for j in indices[i][:10]:
    #    print(j, corpus[j])
    #print()
    #print(retrieval_results[i])
    #    print(scores[i][:10])
    #    print(retrieval_results[i][:10])  
    #from FlagEmbedding.llm_embedder.src.utils import save_json

    metrics = evaluate(retrieval_results, ground_truths)

    print(metrics)


if __name__ == "__main__":
    main()