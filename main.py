import yaml
from src.ingest import process_all_pdfs, split_documents
from src.vectorstore import OnnxEmbeddingWrapper, VectorStore, RAGRetriever
from src.llm import get_llm, rag_advanced

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def build_vector_store():
    config = load_config()
    documents = process_all_pdfs(config["data_dir"])
    chunks = split_documents(
        documents,
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"]
    )
    
    print("\nCreating ONNX embedding function...")
    embedding = OnnxEmbeddingWrapper()

    print("Building vector store...")
    vectorstore = VectorStore(persist_directory=config["persist_directory"])
    vectorstore.create(chunks, embedding)
    print("Vector store build complete.")

def run_query(query: str):
    config = load_config()
    embedding = OnnxEmbeddingWrapper()
    vectorstore = VectorStore(persist_directory=config["persist_directory"])
    vectorstore.load(embedding)
    
    rag_retriever = RAGRetriever(vectorstore, embedding)
    llm = get_llm(
        model_name=config["model_name"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"]
    )
    
    result = rag_advanced(
        query,
        rag_retriever,
        llm,
        top_k=3,
        min_score=0.1,
        return_context=True
    )
    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAG Pipeline Module")
    parser.add_argument("--build", action="store_true", help="Build the vector store from data_dir")
    parser.add_argument("--query", type=str, help="Query the RAG pipeline")
    args = parser.parse_args()

    if args.build:
        build_vector_store()
    
    if args.query:
        ans = run_query(args.query)
        print("\n=== ANSWER ===")
        print(ans['answer'])
        print("\n=== SOURCES ===")
        import json
        print(json.dumps(ans['sources'], indent=2))
    
    if not args.build and not args.query:
        print("Please provide --build or --query 'your question'")
