import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from dotenv import load_dotenv
import bs4
import chromadb

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

st.set_page_config(page_title="ALL-in 코딩 공모전 챗봇", page_icon="🏆")

def init_session():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chain" not in st.session_state:
        st.session_state.chain = None

def create_vectorstore():
    """Create vector store from web content"""
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        
        loader = WebBaseLoader(
            web_paths=["https://spartacodingclub.kr/blog/all-in-challenge_winner"],
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("editedContent", "my-callout")
                )
            )
        )
        documents = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
        splits = splitter.split_documents(documents)
        
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            client=client,
            collection_name="coding_contest"
        )
        
        return vectorstore
    except Exception as e:
        st.error(f"벡터 스토어 생성 중 오류 발생: {str(e)}")
        return None

def setup_chain():
    """Set up the RAG chain"""
    if not OPENAI_API_KEY:
        st.error("OpenAI API 키가 필요합니다.")
        return None
        
    try:
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.1,
            openai_api_key=OPENAI_API_KEY
        )
        
        vectorstore = create_vectorstore()
        if vectorstore is None:
            return None
            
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}  
        )
        
        template = """주어진 컨텍스트를 기반으로 질문에 상세히 답변해주세요.
        
        컨텍스트: {context}
        
        질문: {question}
        
        답변:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        chain = (
            {"context": retriever, "question": lambda x: x}
            | prompt
            | llm
        )
        
        return chain
    except Exception as e:
        st.error(f"체인 설정 중 오류 발생: {str(e)}")
        return None

def main():
    st.title("🏆 'ALL-in 코딩 공모전' 수상작 소개 Bot")
    
    init_session()
    
    if st.session_state.chain is None:
        with st.spinner("시스템을 초기화하는 중..."):
            st.session_state.chain = setup_chain()
            if st.session_state.chain is None:
                st.error("시스템 초기화 실패")
                return
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle user input
    if prompt := st.chat_input("궁금한 점을 물어보세요!"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                try:
                    response = st.session_state.chain.invoke(prompt)
                    st.markdown(response.content)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.content
                    })
                except Exception as e:
                    st.error(f"답변 생성 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()