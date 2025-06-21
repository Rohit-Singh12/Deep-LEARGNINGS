from RAG import RAGGraph, RAGState
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG Graph")
    parser.add_argument("--llm_url", type=str, required=True, help="URL of the LLM service")
    parser.add_argument("--client_url", type=str, required=True, help="URL of the Qdrant client")
    parser.add_argument("--embedding_model_name", type=str, default="all-MiniLM-L6-v2", help="Name of the embedding model")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the knowledge base file")
    parser.add_argument("--sample_query", type=str, default=None, help="Sample query to test the knowledge base")
    parser.add_argument("--collection_name", type=str, required=True, help="Name of the Qdrant collection")
    parser.add_argument("--query", type=str, required=True, help="Query to run against the RAG graph")
    parser.add_argument("--return_context", action='store_true', help="Return context instead of answer")

    args = parser.parse_args()

    rag_graph = RAGGraph(
        llm_url=args.llm_url,
        client_url=args.client_url,
        embedding_model_name=args.embedding_model_name,
        file_path=args.file_path,
        sample_query=args.sample_query,
        collection_name=args.collection_name
    )

    final_state = rag_graph.graph.invoke(RAGState(query=args.query, context="", answer="", return_context=args.return_context))
    if args.return_context:
        print(f"Context: {final_state['context']}")
    else:
        print(f"Final answer: {final_state['answer']}")