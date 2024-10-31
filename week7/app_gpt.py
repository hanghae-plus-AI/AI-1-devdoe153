import streamlit as st
import os
from openai import OpenAI

from dotenv import load_dotenv

# API 키 로드
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"
    st.session_state["max_tokens"] = 256  # 최대 출력 token 수
    st.session_state["temperature"] = 0.1  # sampling할 때 사용하는 temperature
    st.session_state["frequency_penalty"] = 0.0  # 반복해서 나오는 token들을 조절하는 인자


st.title("GPT Bot")

# Session state 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 만약 app이 rerun하면 message들을 다시 UI에 띄우기
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],  # 기존의 메시지들을 모두 이어붙여서 prompt로 보냅니다.
            max_tokens=st.session_state["max_tokens"],
            temperature=st.session_state["temperature"],
            frequency_penalty=st.session_state["frequency_penalty"],
            stream=True,
        )
        response = st.write_stream(stream)

    st.session_state.messages.append({
        "role": "assistant", 
        "content": response
    })