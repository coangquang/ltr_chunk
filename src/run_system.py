import argparse
from system import Retriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_file", default=None, type=str,
                        help="Directory of the subcorpus file (scorpus.json)")
    parser.add_argument("--data_dir", default=None, type=str,
                        help="Directory of the data folder (containing train, test, val, ttrain, ttest, tval file)")
    parser.add_argument("--BE_checkpoint", default="vinai/phobert-base-v2", type=str,
                        help="Name or directory of pretrained model for encoder")
    parser.add_argument("--BE_representation", default=0, type=int,
                        help="Type of encoder representation (-10 for avg, -100 for pooled-output)")
    parser.add_argument("--BE_score", default="dot", type=str,
                        help="Type of similarity score")
    parser.add_argument("--q_fixed", default=False, type=bool,
                        help="To fix question encoder during training stage or not")
    parser.add_argument("--ctx_fixed", default=False, type=bool,
                        help="To fix context encoder during training stage or not")
    parser.add_argument("--q_len", default=32, type=int,
                        help="Maximum token length for question")
    parser.add_argument("--ctx_len", default=256, type=int,
                        help="Maximum token length for context")
    parser.add_argument("--biencoder_path", default=None, type=str,
                        help="Path to save the state with highest validation result.")
    parser.add_argument("--index_path", default=None, type=str,
                        help="Path to save the index")
    parser.add_argument("--cross_checkpoint", default="vinai/phobert-base-v2", type=str,
                        help="Name or directory of pretrained model for cross-encoder")
    parser.add_argument("--cross_representation", default=0, type=int,
                        help="Type of encoder representation (-10 for avg, -100 for pooled-output)")
    parser.add_argument("--cross_load_path", default=None, type=str,
                        help="Path to load state")
    parser.add_argument("--cross_len", default=256, type=int,
                        help="Maximum token length for cross-text")
    parser.add_argument("--cross_dropout", default=0.1, type=float,
                        help="cross-encoder threshold for classification")
    parser.add_argument("--cross_ratio", default=1.0, type=float,
                        help="ratio for cross-encoder score")
    parser.add_argument("--dpr_ratio", default=0.0, type=float,
                        help="ratio for dpr score")
    
    args = parser.parse_args()
    system = Retriever(args)
    system.test_on_data(top_k = [1,5,10,30], test=True, val=True)
    
if __name__ == "__main__":
    main()