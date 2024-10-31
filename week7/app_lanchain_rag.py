import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
from dotenv import load_dotenv
# API 키 로드
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=OpenAIEmbeddings()
)

retriever = vectorstore.as_retriever()
print(retriever.invoke("What is Task Decomposition?"))

user_msg = "What is Task Decomposition?"
retrieved_docs = retriever.invoke(user_msg)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
prompt = hub.pull("rlm/rag-prompt")

user_prompt = prompt.invoke({"context": format_docs(retrieved_docs), "question": user_msg})
print("\n\n user_prompt >> ", user_prompt)

response = llm.invoke(user_prompt)
print("\n\n response.content >> ",response.content)


from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print(rag_chain.invoke(user_msg))