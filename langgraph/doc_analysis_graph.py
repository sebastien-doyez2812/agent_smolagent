import base64
from typing import List, TypedDict, Annotated, Optional
from langchain_ollama import OllamaLLM
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from IPython.display import Image, display




class AgentState(TypedDict):
    input_file: Optional[str]
    messages: Annotated[list[AnyMessage], add_messages]


vision_llm = Ollama(model = "llava")

def extract_text(img_path:str) -> str:
    all_text = ""

    try:
        with open(img_path, 'rb') as image_file:
            image_bytes = image_file.read()

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        message = [
            HumanMessage(
                content=[
                     {
                        "type": "text",
                        "text": (
                            "Extract all the text from this image. "
                            "Return only the extracted text, no explanations."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        },
                    },
                ]
            )
        ]
        rep = vision_llm.invoke(message)

        all_text += rep +"\n\n"
        return all_text.strip()
    except Exception as e:
        error_msg = f"Error: {e}"
        print(error_msg)
        return ""
    
