# from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_ollama import ChatOllama

def get_qwen_llm(model_name: str = 'llama3.1'):
    '''
        Returns a Langchain compatible LLM instance using Ollama's
        llama3:instruct model. 
        Make sure `Ollama` is running and the model is pulled
    '''

    return ChatOllama(model=model_name)
