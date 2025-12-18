from langchain_community.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

import streamlit as st
from langchain_openai import ChatOpenAI

# Function to Load PDF and Create Retrieval Chain
@st.cache_resource
def load_pdf_chain(pdf_path):
    loader = PyPDFLoader(pdf_path)
    
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5"),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    ).from_loaders([loader])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(
            openai_api_base="http://localhost:1234/v1",
            openai_api_key="not-needed",
            model_name="llama-3.2-3b-instruct"
        ),
        retriever=index.vectorstore.as_retriever()
    )
    
    return qa_chain

# Streamlit UI
st.title("📄 Chat with Your PDF")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        qa_chain = load_pdf_chain("temp.pdf")
        
        st.success("PDF loaded successfully! You can now ask questions about it.")
        
        # ✅ Initialize session state properly
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        # Display existing messages
        for message in st.session_state["messages"]:
            st.chat_message(message["role"]).markdown(message["content"])

        # Get user prompt
        prompt = st.chat_input("Ask a question about the PDF:")

        if prompt:
            st.chat_message("user").markdown(prompt)
            st.session_state["messages"].append({"role": "user", "content": prompt})

            response = qa_chain.run(prompt)

            st.chat_message("assistant").markdown(response)
            st.session_state["messages"].append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"Error loading PDF or processing: {e}")
