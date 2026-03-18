from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

def create_vector_db():
    loader = TextLoader("data/company.txt")
    docs = loader.load()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(docs, embeddings)

    os.makedirs("db", exist_ok=True)
    db.save_local("db/faiss_index")

    print("✅ Vector DB created successfully!")

if __name__ == "__main__":
    create_vector_db()