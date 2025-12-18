from langchain_community.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

import streamlit as st
from langchain_openai import ChatOpenAI

# Set LM Studio API as OpenAI-compatible endpoint
llm = ChatOpenAI(
    openai_api_base="http://localhost:1234/v1",
    openai_api_key="not-needed",  # LM Studio doesn't require a real key, but this is mandatory to set
    model_name="llama-3.2-3b-instruct",  # Replace with your actual model name if needed
)

st.title('Ask ChatBot 🤖')

# Initialize chat history in session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display all historical messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Prompt for new user input
prompt = st.chat_input('Pass Your Prompt Here')

if prompt:
    st.chat_message('user').markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    # Send prompt to LLM (LM Studio)
    response = llm.invoke(prompt)

    # Show the LLM response
    st.chat_message('assistant').markdown(response.content)

    # Store the LLM response in chat history
    st.session_state.messages.append({'role': 'assistant', 'content': response.content})
