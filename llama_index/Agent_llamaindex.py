from llama_index.llms.ollama import Ollama
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
import asyncio

def multiply(a:int , b: int) -> int:
    return a *b

llm = Ollama(model= "qwen2.5", is_function_calling_model= True)

agent = AgentWorkflow.from_tools_or_functions([
    FunctionTool.from_defaults(multiply)
], 
llm= llm)



async def main():
    ctx = Context(agent)
    questions = ["what is 2 times 2?", "Hey I am Seb!", "What's my name?"]
    for question in questions:
        print(f"The question was {question}\n\n")
        response = await agent.run(question, ctx = ctx)
        print(f"Agent: {response}\n---------------------")


asyncio.run(main())