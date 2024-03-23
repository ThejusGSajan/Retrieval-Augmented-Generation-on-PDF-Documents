import streamlit as st
import os
import tempfile
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def session_init():
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Ask me about your pdfs!"]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hi!"]

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
        print(chunks)

if __name__ == "__main__":
    main()