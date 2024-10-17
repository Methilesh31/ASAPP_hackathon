import streamlit as st
from rag_system import RAGSystem


rag_system = RAGSystem(papers_folder=None)  
st.title("Research Paper QA Bot")
query = st.text_input("Enter your question:")
if query:
    response = rag_system.generate_response(query)
    
    st.subheader("Answer:")
    st.write(response)

