from langgraph.graph import StateGraph, START, END
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from LoadDocuments import LoadDocuments
from typing import TypedDict

class RAGState(TypedDict):
    query: str   
    context: str
    answer: str
    return_context: bool = False

class RAGGraph:
    def __init__(self, llm_url, client_url, embedding_model_name, file_path, sample_query=None, collection_name="knowledge_base"):
        self.graph = self.create_rag_graph()
        self.state = RAGState(query="", context="", answer="")
        self.llm_url = llm_url
        self.load_documents = LoadDocuments(
            client_url=client_url,
            embedding_model_name=embedding_model_name,
            file_path=file_path,
            sample_query=sample_query,
            collection_name=collection_name
        )
    def llm_node(self, state: RAGState) -> RAGState:
        print(f"Running LLM node for {state}")
        llm = Ollama(model="llama3.2", base_url=self.llm_url)
        prompt = PromptTemplate.from_template("""
            You are a helpful assistant that answer questions from the context provided.
            Context: {context}
            Question: {query}
            Answer:
            """
        )
        chain = prompt | llm | StrOutputParser()
        print(f"Invoking chain for {state}")
        answer = chain.invoke({"context": state["context"], "query": state["query"]})
        return {**state, "answer": answer}

    def context_node(self, state: RAGState):
        print(f"Querying context for {state}") 
        context = self.load_documents.query(state["query"])
        return {**state, "context": context or "No relevant context found."}
    
    def conditional_edge(self, state: RAGState):
        if state["return_context"]:
            return END
        else:
            return "llm_node"

    def create_rag_graph(self):
        graph = StateGraph(RAGState)
        graph.add_node("context_node", self.context_node)
        graph.add_node("llm_node", self.llm_node)
        
        graph.add_edge(START, "context_node")
        graph.add_conditional_edges("context_node", self.conditional_edge)
        graph.add_edge("llm_node", END)
        rag_graph = graph.compile()
        return rag_graph



