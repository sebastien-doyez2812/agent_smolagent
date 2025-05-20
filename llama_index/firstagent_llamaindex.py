from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from dotenv import load_dotenv
import os



load_dotenv(".env")
hf_token = os.getenv("HF_TOKEN")

llm = HuggingFaceInferenceAPI(
    model_name = "Qwen/Qwen2.5-Coder-32B-Instruct",
    temperature = 0.7,
    max_tokens = 100,
    token = hf_token
)

response = llm.complete("Hi, how are you?")
print(response)