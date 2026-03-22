from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai
import streamlit as st
import os

# 🔐 API key
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

DB_PATH = "db/faiss_index"

# ✅ Create or load FAISS
if os.path.exists(DB_PATH):
    db = FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    from langchain_community.document_loaders import TextLoader

    loader = TextLoader("data/company.txt")
    docs = loader.load()

    db = FAISS.from_documents(docs, embeddings)
    db.save_local(DB_PATH)

# Retriever
retriever = db.as_retriever(search_kwargs={"k": 1})

# Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")


def get_answer(query):
    docs = retriever.invoke(query)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
Answer in ONE short sentence.

Context:
{context}

Question:
{query}
"""

    response = model.generate_content(prompt)

    return response.text.strip()