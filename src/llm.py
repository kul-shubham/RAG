from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

def get_llm(model_name: str, temperature: float, max_tokens: int):
    load_dotenv()
    # Ensure groq_api_key is available in the environment
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable not found. Please set it in .env")
    
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return llm

def rag_advanced(query, retriever, llm, top_k=5, min_score=0.2, return_context=False):
    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)
    if not results:
        return {'answer': 'No relevant context found.',
                'sources': [], 'confidence': 0.0, 'context': ''}

    context = "\n\n".join([doc['content'] for doc in results])

    sources = [{
        'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),
        'page': doc['metadata'].get('page', 'unknown'),
        'score': doc['similarity_score'],
        'preview': doc['content'][:300] + '...'
    } for doc in results]

    confidence = max([doc['similarity_score'] for doc in results])

    prompt = (f"Use the following context to answer concisely.\n"
              f"Context:\n{context}\n\nQuestion:{query}\n\nAnswer: ")
    response = llm.invoke([prompt])

    output = {
        'answer': response.content,
        'sources': sources,
        'confidence': confidence
    }
    if return_context:
        output['context'] = context
    return output
