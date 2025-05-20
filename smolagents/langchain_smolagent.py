from langchain.agents import load_tools
from smolagents import CodeAgent, Tool
from langchain_ollama.llms import OllamaLLM
import os

#os.environ["SERPAPI_API_KEY"] = "cle_api"

model = OllamaLLM(model = "gemma")
search_tool = Tool.from_langchain(load_tools(["serpapi"])[0])

agent = CodeAgent(tools= [search_tool], model = model )

agent.run("Do a list of the best skatepark in Montréal.")