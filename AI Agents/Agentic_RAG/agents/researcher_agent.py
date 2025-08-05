from langgraph.prebuilt import create_react_agent
# from langchain.agents import AgentExecutor
from tools import TOOLS
from agents.prompts import make_system_prompt

def get_research_agent(llm):
    prompt = make_system_prompt("You can do research using tools like Web Search and VectorDB.\
                     If any question is asked regarding Kubernetes. Try getting result from VectorDB first.")
    agent = create_react_agent(llm, tools=TOOLS, prompt=prompt)
    return agent


