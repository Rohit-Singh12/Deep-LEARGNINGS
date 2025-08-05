import argparse
from qdrant.loader import LoadDocuments

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Ingest data into FAISS Vector Store")
    parser.add_argument("--file_path", required=True, help="Path to the document to ingest")
    parser.add_argument("--embedding_model", default="BAAI/bge-small-en-v1.5", help="HuggingFace embedding model")
    parser.add_argument("--sample_query", help="Optional: Run a test query after indexing")
    parser.add_argument("--index_path", default="faiss.index", help="Path to save/load FAISS index")
    parser.add_argument("--metadata_path", default="metadata.pkl", help="Path to save/load text metadata")

    args = parser.parse_args()

    loader = LoadDocuments(
        embedding_model_name=args.embedding_model,
        file_path=args.file_path,
        sample_query=args.sample_query,
        index_path=args.index_path,
        metadata_path=args.metadata_path
    )

    loader.initialize_index()
    
