import streamlit as st
import os
from dotenv import load_dotenv
# API 키 로드
load_dotenv()

from huggingface_hub import login
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace


st.title("HF Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


@st.cache_resource
def get_model():
    hf_token = os.getenv("HF_TOKEN")
    login(hf_token)

    llm = HuggingFacePipeline.from_model_id(
        model_id="google/gemma-2b-it",
        task="text-generation",
        device=0,  # 0번 GPU에 load
        pipeline_kwargs={
            "max_new_tokens": 256,  # 최대 256개의 token 생성
            "do_sample": False  # deterministic하게 답변 결정
        }
    )
    model = ChatHuggingFace(llm=llm)
    return model

model = get_model()
if prompt := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        messages = []
        # 기존의 message들을 모두 포함하여 prompt 준비
        for m in st.session_state.messages:
            if m["role"] == "user":
                messages.append(HumanMessage(content=m["content"]))
            else:
                messages.append(AIMessage(content=m["content"]))

        result = model.invoke(messages)
        response = result.content.split('<start_of_turn>')[-1]
        st.markdown(response)
        
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response
    })