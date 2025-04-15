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
from prediction import bi_answer, cross_answer
            
def app():
    #parser = HfArgumentParser([Args])
    #args: Args = parser.parse_args_into_dataclasses()[0]
    #print(args)
    #model = SharedBiEncoder(model_checkpoint=args.encoder,
    #                        representation=args.sentence_pooling_method,
    #                        fixed=True)
    #model.to('cuda')
    #tokenizer = AutoTokenizer.from_pretrained(args.tokenizer if args.tokenizer else args.encoder)
    #corpus_data = pd.read_csv(args.corpus_file)
    #corpus = corpus_data['tokenized_text'].tolist()
    #faiss_index = index(
    #    model=model, 
    #    tokenizer=tokenizer,
    #    corpus=corpus, 
    #    batch_size=args.batch_size,
    #    max_length=args.max_passage_length,
    #    index_factory=args.index_factory,
    #    save_path=args.save_path,
    #    save_embedding=args.save_embedding,
    #    load_embedding=args.load_embedding
    #)
    #reranker = RerankerForInference(model_checkpoint=args.cross_checkpoint)
    #reranker.to('cuda')
    #reranker_tokenizer = AutoTokenizer.from_pretrained(args.cross_checkpoint)
    
    st.header("Vietnamese Legal Retriever Web App")
    #st.subheader("")
    option = st.selectbox(
    "Select your retrieval system",
    ("Bi-encoder only", "Bi-encoder + Cross-encoder Re-ranker"))

    st.write("System selected:", option)
    user_input = st.text_area(
        "Enter your legal query/question below and click the button to submit."
    )

    with st.form("my_form"):
        submit = st.form_submit_button(label="Search")

    if submit:
        if option == 'Bi-encoder only':
            ans, timee = bi_answer(user_input)
        else:
            ans, timee = cross_answer(user_input)
        #st.write("Retrieval Time:", timee, "s.")
        st.write(ans)

    

if __name__ == "__main__":
    app()