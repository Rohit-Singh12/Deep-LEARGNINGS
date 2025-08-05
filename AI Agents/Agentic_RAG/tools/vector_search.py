from qdrant.loader import LoadDocuments
from langchain.tools import tool
from langchain.pydantic_v1 import BaseModel, Field



class SearchInput(BaseModel):
    query: str = Field(..., description="The search query to find relevant documents.")
    limit: int = Field(10, description="The maximum number of results to return.")
    
@tool
def search_documents(query: str, limit: int = 10):
    """
    Search for documents in the FAISS index based on a query.
    
    Args:
        query (str): The search query.
        limit (int): The maximum number of results to return.
    
    Returns:
        str: A list of documents matching the query.
    """
    # Load FAISS index from disk (already created)
    client = LoadDocuments(
        index_path="faiss.index",
        metadata_path="metadata.pkl"
    )
    print("Searching documents with query:", query)
    result = client.query(
        query_text=query,
        top_k=limit
    )
    print("Search results:", result)
    return result
