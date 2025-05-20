from smolagents import ToolCallingAgent, DuckDuckGoSearchTool, InferenceClientModel
import os
#login()
#os.environ['HF_TOKEN'] = API key 

agent = ToolCallingAgent(tools= [DuckDuckGoSearchTool()], model = InferenceClientModel())

agent.run("Search for the best music recommendations for a party at the Wayne's mansion.")