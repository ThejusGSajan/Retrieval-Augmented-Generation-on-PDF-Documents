import streamlit as st

def main():
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Ask me about your pdfs!"]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hi!"]
    
    st.title("Real-Time Retrieval Augmented Generation on PDF Documents")
    st.sidebar.title("Document processing")

if __name__ == "__main__":
    main()