from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai
import streamlit as st
import os

# 🔐 Configure API key safely
if "GOOGLE_API_KEY" not in st.secrets:
    raise ValueError("GOOGLE_API_KEY not found in Streamlit secrets")

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# ✅ Use stable Gemini model
model = genai.GenerativeModel("gemini-pro")

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

DB_PATH = "db/faiss_index"

def load_or_create_db():
    if os.path.exists(DB_PATH):
        return FAISS.load_local(
            DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        from langchain_community.document_loaders import TextLoader

        if not os.path.exists("data/company.txt"):
            raise FileNotFoundError("data/company.txt not found")

        loader = TextLoader("data/company.txt")
        docs = loader.load()

        db = FAISS.from_documents(docs, embeddings)
        db.save_local(DB_PATH)
        return db

# Load DB
db = load_or_create_db()
retriever = db.as_retriever(search_kwargs={"k": 1})


def get_answer(query):
    try:
        docs = retriever.invoke(query)

        if not docs:
            return "I don't have enough information."

        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""
Answer in ONE short sentence.

Context:
{context}

Question:
{query}
"""

        response = model.generate_content(prompt)

        if not response or not response.text:
            return "No response generated."

        return response.text.strip()

    except Exception as e:
        return f"Error: {str(e)}"