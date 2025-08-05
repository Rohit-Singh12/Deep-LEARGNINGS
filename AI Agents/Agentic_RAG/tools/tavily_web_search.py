import os
from langchain_tavily import TavilySearch
from langchain.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv
from langchain.tools import tool

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("TAVILY_API_KEY")

class TavilySearchInput(BaseModel):
    query: str = Field(..., description="The search query to find relevant documents.")

@tool
def search_web(query: str) -> str:
    """
    Search the web using Tavily API and
    return the answer.
    Args:
        query (str): The search query.
    Returns:
        str: The answer from the search.
    Raises:
        ValueError: If the API key is not set.
    """
    if not api_key:
        raise ValueError("TAVILY_API_KEY is not set.")
    
    

    tool = TavilySearch(
        max_results=5,
        topic="general",
        include_answer=True,
        # include_raw_content=True
        # include_images=False,
        # include_image_descriptions=False,
        # search_depth="basic",
        # time_range="day",
        # include_domains=None,
        # exclude_domains=None
    )
    print("Searching the web with query:", query)
    res = tool.invoke({"query": query})
    print("Search results:", res)
    return res['answer']

# search_tool = Tool(
#     func=search_web,
#     name="search_web",
#     description="Search the web using Tavily API and return the answer.",
#     args_schema=TavilySearchInput
# )