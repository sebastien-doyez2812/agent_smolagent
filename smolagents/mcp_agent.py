import os
from smolagents import ToolCollection, CodeAgent
from mcp import StdioServerParameters
from smolagents import InferenceClientModel
from smolagents import LiteLLMModel

model = LiteLLMModel(
    model_id="ollama_chat/gemma",
    api_key = "ollama"
)


server_parameters = StdioServerParameters(
    command="uvx",
    args=["--quiet", "pubmedmcp@0.1.3"],
    env={"UV_PYTHON": "3.12", **os.environ},
)
try:
    with ToolCollection.from_mcp(server_parameters, trust_remote_code=True) as tool_collection:
        agent = CodeAgent(tools=[*tool_collection.tools], model=model, add_base_tools=True)
        agent.run("Please find a remedy for hangover.")


except Exception as e:
    print(e)