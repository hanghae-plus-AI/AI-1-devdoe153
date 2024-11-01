## streamlit 관련 패키지
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

# langchain 관련 패키지
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from typing import List
import os
import fitz 
import re

# 환경변수 블러오기
from dotenv import load_dotenv, dotenv_values
load_dotenv()

# 1. 단계
def save_uploaded_file(uploaded_file: UploadedFile) -> str:
    temp_dir = "files"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    return file_path

# 2. 저장된 PDF 파일을 Document 형태로 변환
def pdf_to_documents(pdf_path:str) -> List[Document]:
    documents = []
    loader = PyMuPDFLoader(pdf_path)
    doc = loader.load()
    for d in doc:
        d.metadata['file_path'] = pdf_path
    documents.extend(doc)
    return documents

# 3. Document를 더 작은 document로 변환
def chunk_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return text_splitter.split_documents(documents)

# 4. Document를 벡터 DB에 저장
def save_to_vector_db(documents: List[Document]) -> None:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(documents, embedding=embeddings)
    vectorstore.save_local("faiss_index")

# ========= 2 단계 : PDF 문서를 벡터 DB에 저장하는 함수들 =========

## 사용자 질문에 대한 RAG 처리
@st.cache_data
def process_question(user_question):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 벡터DB 호출
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # 관련 문서 3개를 호출하는 Retriever 생성
    retriever = new_db.as_retriever(search_kwargs={"k":3})

    # 사용자 질문을 기반으로 관련 문서 3개 검색
    retrieve_docs : List[Document] = retriever.invoke(user_question)

    # RAG 체인선언
    chain = get_rag_chain()

    # 결과 호출
    response = chain.invoke({"question": user_question, "context": retrieve_docs})

    return response, retrieve_docs

def get_rag_chain() -> Runnable:
    template = """
    다음의 컨텍스트를 활용해서 질문에 답변해줘
    - 질문에 대한 응답을 해줘
    - 간결하게 5줄 이내로 해줘
    - 곧바로 응답 결과를 말해줘

    컨텍스트 : {context}
    질문 : {question}
    응답 : 
    """

    custom_rag_prompt = PromptTemplate.from_template(template)
    # 모델 선언 변경 가능
    # ex) LLama3, Claude, Mistral, Gemini, etc.
    model = ChatOpenAI(model="gpt-4o-mini")

    return custom_rag_prompt | model | StrOutputParser()

# ========= 3 단계 : 챗봇 화면 구성 =========
@st.cache_data(show_spinner=False)
def convert_pdf_to_image(pdf_path: str, dpi: int = 250) -> List[str]:
    doc = fitz.open(pdf_path) # 문서 열기
    image_paths = []

    # 이미지 저장 폴더 생성
    output_folder = "pdf_images"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for page_num in range(len(doc)):
        page = doc.load_page(page_num) # 페이지 로드

        zoom = dpi / 72 # 확대/축소 비율
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        image_path = os.path.join(output_folder, f"page_{page_num + 1}.png")
        pix.save(image_path)
        image_paths.append(image_path)

    return image_paths

def display_pdf_page(image_path: str, page_number: int) -> None:
    image_bytes = open(image_path, "rb").read() # 파일에서 이미지 인식
    st.image(image_bytes, caption=f"페이지 {page_number}", output_format="PNG", width=600)


def main():
    st.set_page_config("청약 FAQ 챗봇", layout="wide")
    left_col, right_col = st.columns([1,1])

    with left_col:
        st.header("청약 FAQ 챗봇")
        st.text("PDF 파일을 업로드하고, 질문을 입력하면 답변을 얻을 수 있습니다.")

        pdf_doc = st.file_uploader("PDF 파일 업로드", type="pdf")
        button = st.button("PDF 업로드하기")

        if pdf_doc and button:
            with st.spinner("PDF 파일을 벡터 DB에 저장하는 중..."):
                st.text("PDF 파일 업로드 완료")
                pdf_path = save_uploaded_file(pdf_doc)
                pdf_document = pdf_to_documents(pdf_path)
                smaller_documents = chunk_documents(pdf_document)
                save_to_vector_db(smaller_documents)

        user_question = st.text_input("PDF 문서에 대해서 질문해 주세요", placeholder="무순위 청약 시에도 부부 중복신청이 가능한가요?")

        if user_question:
            with st.spinner("답변을 생성하는 중..."):
                answer, context = process_question(user_question)
                st.write(answer)
                i = 0
                for document in context:
                    with st.expander("관련 문서"):
                        st.write(document.page_content)
                        file_path = document.metadata.get("source", "")
                        page_number = document.metadata.get("page", 0) + 1
                        button_key = f"link_{file_path}_{page_number}"
                        reference_button = st.button(f" {os.path.basename(file_path)} 페이지 {page_number}로 이동", key=button_key)
                        if reference_button:
                            st.session_state["page_number"] = str(page_number)
                        i = i + 1

    with right_col:
        page_number = st.session_state.get("page_number")
        if page_number:
            page_number = int(page_number)
            image_folder = "pdf_images"
            images = sorted(os.listdir(image_folder), key=natural_sort_key)
            print(images)
            image_paths = [os.path.join(image_folder, image) for image in images]
            print(page_number)
            print(image_paths[page_number - 1])
            display_pdf_page(image_paths[page_number - 1], page_number)
            

def display_pdf_page(image_path: str, page_number: int) -> None:
    image_bytes = open(image_path, "rb").read() # 파일에서 이미지 인식
    st.image(image_bytes, caption=f"페이지 {page_number}", output_format="PNG", width=600)

def natural_sort_key(s: str) -> List[int]:
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]
if __name__ == "__main__":
    main()

