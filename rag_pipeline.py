from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai
import streamlit as st

# 🔐 Load API key from Streamlit secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector DB
db = FAISS.load_local(
    "db/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Retriever (fast)
retriever = db.as_retriever(search_kwargs={"k": 1})

# Gemini model (FAST)
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