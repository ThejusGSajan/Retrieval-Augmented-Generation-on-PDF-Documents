import streamlit as st
import os
import tempfile
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from streamlit_chat import message

def session_init():
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Ask me about your pdfs!"]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hi!"]

def chat(query,chain,history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

def main():
    session_init()
    st.title("Real-Time Retrieval Augmented Generation on PDF Documents")
    st.sidebar.title("Document Processing")
    uploaded_pdfs = st.sidebar.file_uploader("Upload PDFs", accept_multiple_files=True)

    if uploaded_pdfs:
        documents = []
        for file in uploaded_pdfs:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_path = temp_file.name

            extension = os.path.splitext(file.name)[1]
            reader = None
            if extension == ".pdf":
                reader = PdfReader(temp_path)
                for page_number in reader.pages:
                    text_in_page = page_number.extract_text()
                    document = Document(
                        page_content=text_in_page,
                        metadata={'page': page_number})
                    documents.append(document)
            os.remove(temp_path)
    
        # print(text)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=20)
        chunks = splitter.split_documents(documents)
        # print(chunks)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                           model_kwargs={'device': 'cpu'})
        vector_db = FAISS.from_documents(chunks, embedding=embeddings)

        llm = LlamaCpp(streaming = True,
                       model_path="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                       temperature=0.75,
                       top_p=1,
                       verbose=True,
                       n_ctx=4096)
    
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                 retriever=vector_db.as_retriever(search_kwargs={"k": 2}),
                                                 memory=memory)

        # display chat interface
        reply = st.container()
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                input = st.text_input("Question:", placeholder="Ask me something about your PDF!",
                                      key='input')
                submit_button = st.form_submit_button(label='Generate')

            if submit_button and input:
                with st.spinner('Finding answers...'):
                    output = chat(input, chain, st.session_state['history'])
                st.session_state['past'].append(input)
                st.session_state['generated'].append(output)
        if st.session_state['generated']:
            with reply:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state['past'][i], is_user=True, key=str(i)+'_user',avatar_style='avataaars')
                    message(st.session_state['generated'][i], key=str(i), avatar_style='bottts')

if __name__ == "__main__":
    main()