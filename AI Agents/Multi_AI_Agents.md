# 🤖 Multi-Agent AI System


## 🌟 Introduction & Motivation
Multi-Agent Systems (MAS) are powerful frameworks where multiple autonomous agents interact to achieve common or competing goals. This notebook explores concepts in MAS, including communication, decision-making, and cooperation among AI agents.

### Multi Agents Systems

In this example we will build a AI Agents that researches data from Internet
from the given topic and gives summary.
Instead of using single Agent to do this task we will divide the tasks into
two subtasks -


*   Searching the Internet to collect data about the topic
*   Extract the insights from the data


1.   We will install open-source LLM llama3 and serve it locally
2.   We will use open-source framework **Crew AI**





```python
!pip install crewai
```
```python
!pip install crewai-tools
```

### Running LLM locally


*   Install package colab-xterm to allow the use of terminal in Google Colab
*   Open terminal in notebook using %xterm
*   Install ollama by running following command

```sh
curl -fsSL https://ollama.com/install.sh | sh
ollama server &



```python
!pip install colab-xterm
```

```python
%load_ext colabxterm
```
## Serving LLM locally using
```sh
ollama serve &
```
Ollama starts serving on port 11434

```python
%xterm
```
## Pull model of your choice. Here, llama3.2 is used

```python
!ollama pull llama3.2
```

```python
from crewai import Agent
```
```python
from crewai import LLM
```

```python
llm = LLM(
    model="ollama/llama3.2",
    base_url="http://localhost:11434"
)
```

## Extending capabilities of LLM with tools


*   With LLM models are equipped with tools and other modes through which they
can extract more information, they are able to take more appropriate action
giving them power to act **autonomously**
*   To allow LLM model to search Internet we will use SerperDevTool. You can login to serper and get the API key and set the enivronment variable **SERPER_API_LEY**


```python
!pip install python-dotenv
```

```python
with open(".env","w") as f:
  f.write("SERPER_API_KEY=YOUR_API_KEY\n")
```

```python
from dotenv import load_dotenv
load_dotenv()
```
```python
from crewai_tools import SerperDevTool
serper_dev_tool = SerperDevTool()
```

### Why Multi-Agents?
AI Agent performs much better if the role of the AI Agent is clear and specialized and task assigned is very specific. Making an agent do multi-tasks dilute its response. So, we will create two agents -


*   Research Agent
*   Summarization Agent



## Creating Research Agent

```python
research_agent = Agent(
    role='Internet Researcher',
    goal='''Given the topic, find relevant, authentic and recent development
     happened''',
    backstory=''' You are an Internet researcher, adept that navigating
    internet and gathering high-quality, reliable information
    ''',
    tools=[serper_dev_tool],
    verbose=True,
    llm=llm
)
```
```python
from crewai import Task
```

```python
research_task = Task(
    description='''Use the SerperDevTool to search from most relevant and
    recent data about {topic}''',
    agent=research_agent,
    tools=[serper_dev_tool],
    expected_output='''A detailed research report with key
    insights and source refernces
    '''
)
```

## Creating Summarization Agent

```python
summarization_agent = Agent(
    role='Text Summarizer',
    goal='''Given the text content as input, extract key insights and
    condense it on concise manner''',
    backstory='''You are expert in extracting key insights from the data and
    present it in concise manner''',
    verbose=True,
    llm=llm
)
```

```python
summarization_task = Task(
    agent=summarization_agent,
    description='''Analyze the content carefully and extract key insights from
    the given text''',
    expected_output='''A well-structured summary of the insights obtained from
    the content'''
)
```

## Organize Agents to execute workflow

```python
from crewai import Crew, Process
```

```python
research_crew = Crew(
    agents=[research_agent, summarization_agent],
    tasks=[research_task, summarization_task],
    process=Process.sequential,
    verbose=True
)
```
```python
result = research_crew.kickoff(inputs={'topic':'How Transformers architecure \
 advanced the Large Language Development'})
```

```python
from IPython.display import Markdown
Markdown(result.raw)
```

```python

```

## 🏁 Conclusion
Multi-Agent Systems play a crucial role in modern AI applications. By understanding agent interactions, we can develop smarter, more efficient systems for real-world problems. 🚀

Feel free to experiment with different agent behaviors and extend this project! Contributions are welcome on GitHub. 😊
