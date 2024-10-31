import openai
import os
from dotenv import load_dotenv

# API 키 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embeddings(text, model='text-embedding-ada-002'):
    response = openai.Embedding.create(
        input=text,  # 단일 텍스트 또는 텍스트 리스트
        model=model
    )
    return response['data'][0]['embedding']

# 단일 문자열 테스트
embedding = get_embeddings("This is a test.")
print(embedding)

# 여러 문자열을 리스트로 전달
embedding_multiple = get_embeddings(["This is a test.", "Another test sentence."])
print(embedding_multiple)
