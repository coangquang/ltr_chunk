import os
import argparse
import pandas as pd
import torch
from dpr.util import get_tokenizer, build_dpr_traindata#, embed_corpus
from dpr.trainer import DPRTrainer
from dpr.retriever import DPRRetriever

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_file", default=None, type=str,
                        help="Directory of the corpus file (corpus.json)")
    parser.add_argument("--data_dir", default=None, type=str,
                        help="Directory of the data folder (containing train, test, val, ttrain, ttest, tval file)")
    parser.add_argument("--BE_checkpoint", default="vinai/phobert-base-v2", type=str,
                        help="Name or directory of pretrained model for encoder")
    parser.add_argument("--BE_representation", default=0, type=int,
                        help="Type of encoder representation (-10 for avg, -100 for pooled-output)")
    parser.add_argument("--BE_score", default="dot", type=str,
                        help="Type of similarity score")
    parser.add_argument("--load_path", default=None, type=str,
                        help="Biencoder saved state for multi-stage training")
    parser.add_argument("--q_fixed", default=False, type=bool,
                        help="To fix question encoder during training stage or not")
    parser.add_argument("--ctx_fixed", default=False, type=bool,
                        help="To fix context encoder during training stage or not")
    parser.add_argument("--BE_num_epochs", default=2, type=int,
                        help="Number of training epochs for biencoder")
    parser.add_argument("--grad_cache", default=False, type=bool,
                        help="To use contrasive learning gradient scaling or not")
    parser.add_argument("--q_len", default=32, type=int,
                        help="Maximum token length for question")
    parser.add_argument("--ctx_len", default=256, type=int,
                        help="Maximum token length for context")
    parser.add_argument("--BE_train_batch_size", default=1, type=int,
                        help="Biencoder training batch size (sum in all gpus)")
    parser.add_argument("--BE_val_batch_size", default=1, type=int,
                        help="Biencoder validation batch size")
    parser.add_argument("--q_chunk_size", default=1, type=int,
                        help="Question chunk size when using grad_cache")
    parser.add_argument("--ctx_chunk_size", default=1, type=int,
                        help="Context chunk size when using grad_cache")
    parser.add_argument("--BE_lr", default=0.00001, type=float,
                        help="Biencoder training learning rate")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--BE_loss", default=1.0, type=float,
                        help="Biencoder loss alpha parameter, 1: mean use all (hard) negatives")
    parser.add_argument("--no_hard", default=0, type=int,
                        help="Number of hard negatives using for each question")
    parser.add_argument("--patience", default=10, type=int,
                        help="When using validattion result to save model")
    parser.add_argument("--final_path", default=None, type=str,
                        help="Path to save the final state")
    parser.add_argument("--biencoder_path", default=None, type=str,
                        help="Path to save the state with highest validation result.")
    parser.add_argument("--index_path", default=None, type=str,
                        help="Path to save the index")
    parser.add_argument("--new_data", default=False, type=bool,
                        help="To use crawl data or not")
    parser.add_argument("--all_data", default=False, type=bool,
                        help="To use all positives or not")
    
    args = parser.parse_args()
    dcorpus = pd.read_csv(args.corpus_file)
    if args.new_data:
        print("Training with new data.")
        dtrain = pd.read_csv(os.path.join(args.data_dir, 'ttrain_all.csv'))
        dval = pd.read_csv(os.path.join(args.data_dir, 'tval_all.csv'))
        #dtest = pd.read_csv(os.path.join(args.data_dir, 'ttest_all.csv'))
    else:
        dtrain = pd.read_csv(os.path.join(args.data_dir, 'ttrain.csv'))
        dval = pd.read_csv(os.path.join(args.data_dir, 'tval.csv'))
        #dtest = pd.read_csv(os.path.join(args.data_dir, 'ttest.csv'))
    corpus_tokenized = dcorpus['tokenized_text'].tolist()
    tokenizer = get_tokenizer(args.BE_checkpoint)
    print("\t* Loading data...")
    print(args)
    val_loader = build_dpr_traindata(corpus=corpus_tokenized, 
                                     df=dval, 
                                     tokenizer=tokenizer, 
                                     q_len=args.q_len, 
                                     ctx_len=args.ctx_len,
                                     batch_size=args.BE_val_batch_size, 
                                     no_hard=args.no_hard, 
                                     shuffle=True,
                                     all_data=args.all_data)

    train_loader = build_dpr_traindata(corpus=corpus_tokenized, 
                                       df=dtrain, 
                                       tokenizer=tokenizer, 
                                       q_len=args.q_len, 
                                       ctx_len=args.ctx_len, 
                                       batch_size=args.BE_train_batch_size, 
                                       no_hard=args.no_hard, 
                                       shuffle=True,
                                       all_data=args.all_data)

    dpr_trainer = DPRTrainer(args=args,
                            train_loader=train_loader,
                            val_loader=val_loader)
    
    q_encoder, ctx_encoder = dpr_trainer.train_biencoder()
    torch.cuda.empty_cache()
    print("Check with the final state:")
    dpr_retriever = DPRRetriever(args, q_encoder, ctx_encoder, save_type="final")
    dpr_retriever.test_on_data(top_k = [1,5,10,30,100])
    dpr_retriever.increase_neg(no_negs=15, segmented=True)
    print("Check with the best state:")
    torch.cuda.empty_cache()
    dpr_retriever = DPRRetriever(args, save_type="best")
    dpr_retriever.test_on_data(top_k = [1,5,10,30,100])
    dpr_retriever.increase_neg(no_negs=15, segmented=True)
    
    
if __name__ == "__main__":
    main()