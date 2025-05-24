from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent, ReActAgent
from llama_index.llms.ollama import Ollama
import asyncio
import wikipedia

#Tools:
def add(a:int, b:int)-> int:
    """ 
        return the multiplication of a and b
    """
    return a + b

def substract(a:int, b: int) -> int:
    """
        substract b to a, return a -b
    """
    return a - b


def basic_search(query: str) -> str:
    """
        search on the web information about the query
    """
    wikipedia.set_lang("en")
    response = wikipedia.summary(query, sentences=10)
    return response.text

#LLM:
llm = Ollama(model= "qwen2.5", is_function_calling_model= True)

#Agents:
calculator_agent = ReActAgent(
    name = "calculator",
    description="Performs basic arithmetric operations",
    system_prompt= "You are a calculator assistant. Use your tools for any maths operations",
    tools = [add, substract],
    llm= llm
)

search_agent = ReActAgent(
    name="search agent",
    description = "Looks up information about something",
    system_prompt= "Use your tool to search on the web information",
    tools= [basic_search],
    llm = llm
)

agent = AgentWorkflow(
    agents=[calculator_agent, search_agent], root_agent="calculator"
)

async def main():
    questions = ["Hey, what you give me the birthday of Napoléon Bonaparte?",
                 "can you add 5 and 3?",
                 "What should be the age of Napoléon today?"]
    
    for question in questions:
        response = await agent.run(user_msg=question)
        print(f"Question: {question}\n Answer:{response}\n\n")
        

asyncio.run(main())


"""
Testing:


Question: Hey, what you give me the birthday of Napoléon Bonaparte?
 Answer:Napoléon Bonaparte's birthday is May 15, 1769.


Question: can you add 5 and 3?
 Answer:The sum of 5 and 3 is 8.


Question: What should be the age of Napoléon today?
 Answer:If Napoléon were alive today, he would be approximately 2023 - 1769 = 254 years old. However, this is an unrealistic value as people cannot live that long; it simply represents his age based on the current year.


"""