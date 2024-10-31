from openai import OpenAI
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# 방법 1: 환경 변수를 통한 API 키 설정
client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')  # .env 파일에서 키를 가져옴
)

print("os.getenv('OPENAI_API_KEY') ->",os.getenv('OPENAI_API_KEY'))
# 방법 2: 직접 API 키 설정 (테스트용, 실제 프로덕션에서는 권장하지 않음)
# client = OpenAI(
#     api_key='your-api-key-here'
# )

def test_openai_connection():
    try:
        # API 연결 테스트
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Hello!"}
            ]
        )
        print("API 연결 성공:", response.choices[0].message.content)
    except Exception as e:
        print("에러 발생:", str(e))

if __name__ == "__main__":
    test_openai_connection()