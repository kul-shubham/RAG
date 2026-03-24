from langchain_chroma import Chroma
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from typing import List, Dict, Any

class OnnxEmbeddingWrapper:
    """Wraps Chroma's ONNX DefaultEmbeddingFunction so it can be used as a LangChain-compatible embedding_function."""
    def __init__(self):
        self._fn = DefaultEmbeddingFunction()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._fn(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._fn([text])[0]

class VectorStore:
    def __init__(self, persist_directory="chroma_db"):
        self.persist_directory = persist_directory
        self.vectorstore = None

    def create(self, documents, embedding):
        self.vectorstore = Chroma.from_documents(
            documents,
            embedding,
            persist_directory=self.persist_directory
        )

    def load(self, embedding):
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=embedding
        )

class RAGRetriever:
    def __init__(self, vector_store: VectorStore, embedding):
        self.vector_store = vector_store
        self.embedding = embedding

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        print(f"Retrieving documents for query: '{query}'")
        try:
            results = self.vector_store.vectorstore.similarity_search_with_score(query, k=top_k)
            retrieved_docs = []
            for rank, (doc, score) in enumerate(results):
                similarity_score = 1 / (1 + score)
                if similarity_score >= score_threshold:
                    retrieved_docs.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": score,
                        "similarity_score": similarity_score,
                        "rank": rank + 1
                    })
            return retrieved_docs
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
