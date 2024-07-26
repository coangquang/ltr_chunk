import os
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
logger = logging.getLogger(__name__)
from flask import Flask, request, redirect, jsonify
from flask_ngrok import run_with_ngrok
import prediction

app = Flask(__name__)
run_with_ngrok(app)

@app.route("/")
def hello():
  return "Hello World!! from anywhere in the world!"

if __name__ == '__main__':
      app.run()


            

    

if __name__ == "__main__":
    app()