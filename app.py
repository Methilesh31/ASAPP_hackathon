import streamlit as st
from rag_system import RAGSystem

# Initialize RAG system (load index from file)
rag_system = RAGSystem(papers_folder=None)  # No need to specify papers folder as we are loading from the saved index

st.title("Research Paper QA Bot")

query = st.text_input("Enter your question:")

if query:
    response = rag_system.generate_response(query)
    
    st.subheader("Answer:")
    st.write(response)

