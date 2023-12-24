import os
import math
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from util import build_cross_sub_data, build_cross_data
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryAccuracyEvaluator, CEBinaryClassificationEvaluator 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir", default=None, type=str,
                        help="Directory of the (sub) corpus file (scorpus.csv)")
    parser.add_argument("--file_dir", default=None, type=str,
                        help="Directory of the files to prepare data for training cross-encoder")
    parser.add_argument("--json_dir", default=None, type=str,
                        help="Directory of the files to prepare data for training cross-encoder")
    parser.add_argument("--cross_checkpoint", default="vinai/phobert-base-v2", type=str,
                        help="Name or directory of pretrained model for cross-encoder")
    parser.add_argument("--cross_num_epochs", default=2, type=int,
                        help="Number of training epochs for cross-encoder")
    parser.add_argument("--cross_max_len", default=256, type=int,
                        help="Maximum token length for cross-encoder input")
    parser.add_argument("--cross_batch_size", default=1, type=int,
                        help="Cross-encoder training batch size (sum in all gpus)")
    parser.add_argument("--cross_lr", default=0.00001, type=float,
                        help="cross-encoder training learning rate")
    parser.add_argument("--cross_threshold", default=0.5, type=float,
                        help="cross-encoder threshold for classification")
    parser.add_argument("--cross_no_negs", default=1, type=int,
                        help="Number of top negatives using")
    parser.add_argument("--cross_eval_steps", default=1000, type=int)
    parser.add_argument("--cross_save_path", default=None, type=str,
                        help="Path to save the best state")
    parser.add_argument("--cross_train_type", default="all", type=str,
                        help="To train cross encoder on chunks or not")
    
    args = parser.parse_args()
    
    if args.cross_train_type == "sub":
        dscorpus = pd.read_csv(os.path.join(args.corpus_dir, "scorpus.csv"))
        train_samples = build_cross_sub_data(dscorpus=dscorpus,
                                            json_file=os.path.join(args.json_dir, "dpr_train_sub_retrieved.json"),
                                            csv_file=os.path.join(args.file_dir, "ttrain.csv"),
                                            no_negs=args.cross_no_negs)
        val_samples = build_cross_sub_data(dscorpus=dscorpus,
                                            json_file=os.path.join(args.json_dir, "dpr_val_sub_retrieved.json"),
                                            csv_file=os.path.join(args.file_dir, "tval.csv"),
                                            no_negs=args.cross_no_negs)
        test_samples = build_cross_sub_data(dscorpus=dscorpus,
                                            json_file=os.path.join(args.json_dir, "dpr_test_sub_retrieved.json"),
                                            csv_file=os.path.join(args.file_dir, "ttest.csv"),
                                            no_negs=args.cross_no_negs)
    else:
        dcorpus = pd.read_csv(os.path.join(args.corpus_dir, "corpus.csv"))
        train_samples = build_cross_data(dcorpus=dcorpus,
                                        json_file=os.path.join(args.json_dir, "dpr_train_retrieved.json"),
                                        csv_file=os.path.join(args.file_dir, "ttrain.csv"),
                                        no_negs=args.cross_no_negs)
        val_samples = build_cross_data(dcorpus=dcorpus,
                                        json_file=os.path.join(args.json_dir, "dpr_val_retrieved.json"),
                                        csv_file=os.path.join(args.file_dir, "tval.csv"),
                                        no_negs=args.cross_no_negs)
        test_samples = build_cross_data(dcorpus=dcorpus,
                                        json_file=os.path.join(args.json_dir, "dpr_test_retrieved.json"),
                                        csv_file=os.path.join(args.file_dir, "ttest.csv"),
                                        no_negs=args.cross_no_negs)
    
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.cross_batch_size)
    
    model = CrossEncoder(args.cross_checkpoint, num_labels=1, max_length=args.cross_max_len)
    evaluator = CEBinaryAccuracyEvaluator.from_input_examples(val_samples, name='cross-val', threshold= args.cross_threshold)
    warmup_steps = math.ceil(len(train_dataloader) * args.cross_num_epochs * 0.1)
    
    # Train the model
    model.fit(train_dataloader=train_dataloader,
              evaluator=evaluator,
              epochs=args.cross_num_epochs,
              warmup_steps=warmup_steps,
              evaluation_steps=args.cross_eval_steps,
              output_path=args.cross_save_path)

    torch.cuda.empty_cache()
    ##### Load model and eval on test set
    model = CrossEncoder(args.cross_save_path)

    evaluator = CEBinaryAccuracyEvaluator.from_input_examples(test_samples, name='cross-test')
    evaluator(model)
    
if __name__ == "__main__":
    main()