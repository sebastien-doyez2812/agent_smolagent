from smolagents import load_tool, CodeAgent
from smolagents import LiteLLMModel
import os

# API key need:
#os.environ['HF_TOKEN'] = "API HUGGING FACE HERE"

model = LiteLLMModel(
    model_id="ollama_chat/gemma",
    api_key = "ollama"
)

image_gen_tool = load_tool(
    repo_id= "m_ric/text-to-image",
    trust_remote_code= True
)

agent = CodeAgent(tools= [image_gen_tool],
                  model= model)