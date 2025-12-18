🤖 Streamlit LLM Chatbot & PDF Q&A (LM Studio Powered)

Large Language Models (LLMs) are a class of deep learning models trained on massive text corpora to understand, generate, and reason with human language. At their core, LLMs are probabilistic next-token predictors: given a sequence of words (tokens), the model estimates the probability distribution of the next possible token and samples from it to generate coherent text.

Modern LLMs such as LLaMA, GPT, and Mistral are based on the Transformer architecture, which uses self-attention mechanisms to capture long-range dependencies between words.

This project is a Streamlit-based AI application that works in two modes:

Real-Time Chatbot – Talk directly with a locally hosted Large Language Model (LLM)

Chat with PDF – Upload a PDF and ask questions using Retrieval-Augmented Generation (RAG)

The LLM is served locally using LM Studio, making this project fully offline, private, and cost-free.

🚀 Features

✅ Real-time chatbot using a local LLM

✅ PDF-based question answering (RAG)

✅ No OpenAI API key required

✅ Chat history preserved using Streamlit session state

✅ Uses modern embeddings (BGE) for document search

✅ Beginner-friendly and easy to extend

🧠 Tech Stack

Python 3.9+

Streamlit – UI framework

LangChain – LLM orchestration

LM Studio – Local OpenAI-compatible LLM server

LLaMA 3.2 (3B Instruct) – Language model

FAISS – Vector database

HuggingFace BGE Embeddings – Semantic search

📁 Project Structure
project-root/
│── app.py
│── temp.pdf
│── README.md
│── requirements.txt

🧠 Setting Up LM Studio (IMPORTANT)

This project does NOT use OpenAI cloud APIs. Instead, it uses LM Studio as a local OpenAI-compatible server.

Steps:

Download LM Studio

Download model: llama-3.2-3b-instruct

Go to Local Server tab

Start server with:

Port: 1234

OpenAI-compatible API: Enabled

Once running, the API will be available at:

http://localhost:1234/v1

No API key is required.

🧪 Running the Application
streamlit run app.py

Then open:
