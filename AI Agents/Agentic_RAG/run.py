from agents.graph import build_graph
from utils.load_llm import get_qwen_llm  # your Ollama loader for Qwen
from dotenv import load_dotenv
import argparse
# Load environment variables
load_dotenv()
if __name__ == "__main__":
    llm = get_qwen_llm()
    graph = build_graph(llm)
    parser = argparse.ArgumentParser(description="Get the query")
    parser.add_argument("--query", help="Your query")
    args = parser.parse_args()
    user_query = args.query
    results = graph.invoke(
        {"messages": [("user", user_query)]}
    )
    print(results)
    # for step in events:
    #     print(step)
    #     print("----")
