import streamlit as st
import os
from dotenv import load_dotenv
# API 키 로드
load_dotenv()

from huggingface_hub import login
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


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

		# Custom하게 huggingface pipeline을 구성합니다.
    model_id = "sangjib/gpt-finetuned"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=256, 
        do_sample=False, 
        device=0
    )
    
    # 이전과 똑같이 ChatHuggingFace instance를 만듭니다.
    llm = HuggingFacePipeline(pipeline=pipe)
    model = ChatHuggingFace(llm=llm)
    
    # Tokenizer에 pad_token_id와 chat_template을 설정해줍시다.
    model.tokenizer.pad_token_id = tokenizer.eos_token_id
    model.tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"

    return model


model = get_model()  # 모델 초기화
result = model.invoke(messages)
response = result.content.split("<|im_end|>\n")[-1]
st.markdown(response)