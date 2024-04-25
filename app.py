import streamlit as st
import os
import tempfile
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import HNSW
from langchain_community.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from streamlit_chat import message

def preprocess_pdfs(pdf_paths):
    documents = []
    embeddings = []
    for file_path in pdf_paths:
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            for page_number in reader.pages:
                text_in_page = page_number.extract_text()
                document = Document(page_content=text_in_page, metadata={'page': page_number})
                documents.append(document)
  
    splitter = RecursiveCharacterTextSplitter(chunk_size=50000)
    chunks = splitter.split_documents(documents)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    for chunk in chunks:
        embeddings.append(embedding_model.encode(chunk))
  
    return documents, embeddings

def load_preprocessed_data():
    if os.path.exists("preprocessed_data.pkl"):
        import pickle
        with open("preprocessed_data.pkl", "rb") as f:
            documents, embeddings = pickle.load(f)
        return documents, embeddings
    else:
        return None, None

def session_init():
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Ask me about your pdfs!"]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hi!"]

def chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

def main():
    session_init()
    st.title("Real-Time Retrieval Augmented Generation on PDF Documents")
    st.sidebar.title("Document Processing")
    uploaded_pdfs = st.sidebar.file_uploader("Upload PDFs", accept_multiple_files=True)

    if uploaded_pdfs:
        documents, embeddings = preprocess_pdfs([f.name for f in uploaded_pdfs])
    else:
        cached_documents, cached_embeddings = load_preprocessed_data()
        if cached_documents and cached_embeddings:
            documents, embeddings = cached_documents, cached_embeddings
        else:
            st.error("Please upload PDFs or pre-process them beforehand.")
            return

    vector_db = HNSW.from_documents(embeddings)
    llm = LlamaCpp(streaming=True,
                   model_path="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                   temperature=0.75,
                   top_p=1,
                   verbose=True,
                   n_ctx=4096)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                  retriever=vector_db.as_retriever(search_kwargs={"k": 1}),
                                                  memory=memory)

    reply = st.container()
    container = st.container()

    with container:
        user_input = st.text_input("Ask me about your PDFs!", key="user_input")

