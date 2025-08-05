from langgraph.prebuilt import create_react_agent
# from langchain.agents import AgentExecutor
from agents.prompts import make_system_prompt

def writer_agent(llm):
    prompt = make_system_prompt("You are expert in extracting insight provided by the researcher to answer the user query")
    agent = create_react_agent(llm, tools=[], prompt=prompt)
    # return AgentExecutor(agent=agent, tools=[], verbose=True)
    return agent
