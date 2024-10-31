import os
import urllib

import certifi
from fastapi import FastAPI
from pymongo import MongoClient
from pymongo.server_api import ServerApi
# dotenv 라이브러리 import
from dotenv import load_dotenv
# .env 파일 로드
load_dotenv()

MAPPING_EN2KO = {
    "hangover": "해장",
    "diet": "다이어트"
}
MAPPING_KO2EN = {v: k for k, v in MAPPING_EN2KO.items()}

app = FastAPI()

mongodb_id = os.getenv("mongodb-id")
mongodb_pw = os.getenv("mongodb-pw")
uri = f"mongodb+srv://{mongodb_id}:{mongodb_pw}@devdoe.7ca7a.mongodb.net/?retryWrites=true&w=majority&appName=devdoe"
client = MongoClient(uri, server_api=ServerApi('1'), tlsCAFile=certifi.where())
db = client.recommendations_db
collection = db.recommendations


@app.get("/health")
def health():
    return "OK"


@app.get("/recommend/{query_en}")
def recommend(query_en: str = "hangover"):
    query_ko = MAPPING_EN2KO[query_en]
    data = list(collection.find({"_id": query_ko}, {'_id': 0}))
    return data