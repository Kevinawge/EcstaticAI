import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from typing import List

# --- Set your OpenAI API key here ---
os.environ["OPENAI_API_KEY"] = ""


class PDFEmbedder:
    def __init__(self, pdf_folder: str = "./finance_books"):
        self.pdf_folder = pdf_folder
        self.vector_store = None

    def load_pdfs(self) -> List:
        pdf_paths = glob.glob(f"{self.pdf_folder}/*.pdf")
        documents = []
        print(f"[Loading PDFs from {self.pdf_folder}]")
        for path in pdf_paths:
            print(f" - Loading: {os.path.basename(path)}")
            loader = PyPDFLoader(path)
            docs = loader.load()
            documents.extend(docs)
        return documents

    def chunk_documents(self, documents: List) -> List:
        print(f"[Splitting {len(documents)} documents into chunks...]")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        return splitter.split_documents(documents)

    def embed_and_store(self, chunks: List, save_path: str = "faiss_index"):
        print("[Embedding chunks and saving FAISS index...]")
        embeddings = OpenAIEmbeddings()
        self.vector_store = FAISS.from_documents(chunks, embeddings)
        self.vector_store.save_local(save_path)

    def load_index(self, save_path: str = "faiss_index"):
        embeddings = OpenAIEmbeddings()
        self.vector_store = FAISS.load_local(save_path, embeddings)


# Test Block
if __name__ == "__main__":
    print("[Testing PDFEmbedder...]\n")

    embedder = PDFEmbedder(pdf_folder="./finance_books")
    docs = embedder.load_pdfs()
    chunks = embedder.chunk_documents(docs)
    embedder.embed_and_store(chunks, save_path="faiss_index")

    print("\n[FAISS index created and stored at ./faiss_index]")
