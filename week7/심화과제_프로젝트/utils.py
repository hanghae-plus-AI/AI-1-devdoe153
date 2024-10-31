import os

import numpy as np
from openai import OpenAI

# dotenv 라이브러리 import
from dotenv import load_dotenv
# .env 파일 로드
load_dotenv()

client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')  # .env 파일에서 키를 가져옴
)

KEYWORDS_BLACKLIST = ['리뷰', 'zㅣ쀼', 'ZI쀼', 'Zl쀼', '리쀼', '찜', '이벤트', '추가', '소스']
KEYWORDS_CONTEXT = [
    '해장', '숙취',
    '다이어트'
]


def get_embedding(text, model='text-embedding-3-small'):
    
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding


def get_embeddings(text, model='text-embedding-3-small'):
    
    response = client.embeddings.create(
        input=text,
        model=model
    )
    output = []
    for i in range(len(response.data)):
        output.append(response.data[i].embedding)
    return output


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_most_relevant_indices(query_embedding, context_embeddings):
    query = np.array(query_embedding)
    context = np.array(context_embeddings)
    
    similarities = np.array([cosine_similarity(query, ctx) for ctx in context])
    
    sorted_indices = np.argsort(similarities)[::-1].tolist()
    
    return sorted_indices, similarities


def extract_keywords(review_text):
    keywords = []

    for word in review_text.split():
        if any(keyword in word for keyword in KEYWORDS_CONTEXT):
            keywords.append(word)
    return keywords


def is_valid_menu(menu_name):
    return True if not any(keyword in menu_name for keyword in KEYWORDS_BLACKLIST) else False


def call_openai(prompt, temperature=0.0, model='gpt-4o-2024-08-06'):
    
    completion = client.chat.completions.create(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        temperature=temperature
    )

    return completion.choices[0].message.content


import os
import time
import urllib.parse

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo import MongoClient
from collections.abc import MutableMapping
import certifi



def fetch_restaurant_info():
    # 환경 변수 사용
    mongodb_id = os.getenv("mongodb-id")
    mongodb_pw = os.getenv("mongodb-pw")
    uri = f"mongodb+srv://{mongodb_id}:{mongodb_pw}@devdoe.7ca7a.mongodb.net/?retryWrites=true&w=majority&appName=devdoe"
    client = MongoClient(uri, server_api=ServerApi('1'), tlsCAFile=certifi.where())
    db = client.restaurant_db
    collection = db.restaurant_info

    restaurants_info = list(collection.find({}, {'_id': False}))
    return restaurants_info