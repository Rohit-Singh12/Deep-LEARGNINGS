from typing import Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.types import Command

from agents.researcher_agent import get_research_agent
from agents.writer_agent import writer_agent
from dotenv import load_dotenv

load_dotenv()

def get_next_node(last_message: BaseMessage, goto: str):
    print("Last message content:", last_message.content)
    if "FINAL ANSWER" in last_message.content:
        return END
    return goto

def build_graph(llm):
    research_agent = get_research_agent(llm)
    write_agent = writer_agent(llm)

    def research_node(state: MessagesState) -> Command[Literal["writer", "__end__"]]:
        print("Starting research node with state:", state, " and llm:", llm)
        result = research_agent.invoke(state)
        print("Research node result:", result)
        # goto = get_next_node(result["messages"][-1], "writer")
        result["messages"][-1] = HumanMessage(content=result["messages"][-1].content, name="researcher")
        return Command(update={"messages": result["messages"]}, goto="writer")

    def writer_node(state: MessagesState) -> Command[Literal["researcher", "__end__"]]:
        print("Starting writer node with state:", state)
        result = write_agent.invoke(state)
        print("Writer node result:", result)
        goto = get_next_node(result["messages"][-1], "researcher")
        result["messages"][-1] = HumanMessage(content=result["messages"][-1].content, name="writer")
        return Command(update={"messages": result["messages"]}, goto=goto)

    workflow = StateGraph(MessagesState)
    workflow.add_node("researcher", research_node)
    workflow.add_node("writer", writer_node)

    workflow.add_edge(START, "researcher")
    graph = workflow.compile()
    return graph


