import streamlit as st

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

if __name__ == "__main__":
    main()