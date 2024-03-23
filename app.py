import streamlit as st
import os
import tempfile
# from langchain.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader

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
        text = []
        for file in uploaded_pdfs:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_path = temp_file.name

            extension = os.path.splitext(file.name)[1]
            reader = None
            if extension == ".pdf":
                reader = PdfReader(temp_path)
                for page_number in reader.pages:
                    text.append(page_number.extract_text())
            os.remove(temp_path)
    
        print(text)

if __name__ == "__main__":
    main()