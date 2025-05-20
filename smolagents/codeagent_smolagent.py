from huggingface_hub import login 
from smolagents import LiteLLMModel
from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel
import os
#login()
#os.environ['HF_TOKEN'] = api key 
model = LiteLLMModel(
    model_id="ollama_chat/gemma",
    api_key = "ollama"
)

client = InferenceClientModel(
    api_key="API KEY HERE",
)
agent = CodeAgent(tools = [DuckDuckGoSearchTool()], model = client)#InferenceClientModel(model="Qwen/Qwen2.5-Coder-32B-Instruct"))

agent.run("Search for the best music recommendations for a party at the Wayne's mansion.")



