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
from flask import Flask, request, redirect
from flask_restful import Resource, Api
from flask_cors import CORS
import prediction

app = Flask(__name__)
cors =  CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)

class Test(Resource):
    def get(self):
        return 'Welcome to, Test App API!'

    def post(self):
        try:
            value = request.get_json()
            if(value):
                return {'Post Values': value}, 201

            return {"error":"Invalid format."}

        except Exception as error:
            return {'error': error}

class GetPredictionOutput(Resource):
    def get(self):
        return {"error":"Invalid Method."}

    def post(self):
        try:
            data = request.get_json()
            predict = prediction.bi_answer(data['query'])
            predictOutput = predict
            return predictOutput

        except Exception as error:
            return {'error': error}

api.add_resource(Test,'/')
api.add_resource(GetPredictionOutput,'/getPredictionOutput')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)



            

    

if __name__ == "__main__":
    app()