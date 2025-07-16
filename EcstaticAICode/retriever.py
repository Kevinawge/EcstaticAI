import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from typing import List, Optional, Any

# Set your API key securely
os.environ["OPENAI_API_KEY"] = ""


class PDFRetriever:
    def __init__(self, faiss_path: str = "faiss_index", embeddings: Optional[Any] = None):
        print(f"[Retriever] Loading FAISS index from '{faiss_path}'...")
        if embeddings is None:
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings()
        self.vector_store = FAISS.load_local(
            faiss_path, embeddings, allow_dangerous_deserialization=True
        )
        self.retriever = self.vector_store.as_retriever()

    def search(self, query: str, k: int = 5) -> List[Document]:
        print(f"\n[Retriever] Searching top {k} documents for query: {query}")
        return self.retriever.invoke(query, k=k)

    def pretty_print_results(self, docs: List[Document]):
        for i, doc in enumerate(docs):
            print(f"\n--- Result #{i+1} ---")
            print(doc.page_content[:1000])
            if doc.metadata:
                print("\n[Metadata]", doc.metadata)


# Test block
if __name__ == "__main__":
    print("[Testing PDFRetriever...]")

    retriever = PDFRetriever(index_path="faiss_index")

    sample_question = "What is the Black-Scholes formula for option pricing?"
    docs = retriever.search(sample_question, k=3)

    retriever.pretty_print_results(docs)
