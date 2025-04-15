import requests
import json
url = 'http://127.0.0.1:5000/getPredictionOutput'
data = {
  "query": ["thủ tục ly hôn là gì?"]
}
response = requests.post(url, json=data)
try:
    result = response.json()
    print("Prediction output:", result)
except ValueError:
    print("Response is not valid JSON.")