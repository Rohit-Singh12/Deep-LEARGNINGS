from typing import Any, Callable, List

from tools.tavily_web_search import search_web
from tools.vector_search import search_documents

TOOLS: List[Callable[..., Any]] = [
    search_web,
    search_documents,
]