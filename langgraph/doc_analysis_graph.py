import base64
from typing import List, TypedDict, Annotated, Optional
from langchain_ollama import OllamaLLM
from langchain_ollama import ChatOllama
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
    

def devide(a: int, b:int) -> float:
    return a/b


tools = [
    divide,
    extract_text
]


llm = ChatOllama(model= gemma)

llm_with_tools = llm.bind_tools(tools= tools)

# Node: 
def assistant (state: AgentState):
    # Description for the tools:
    textual_description_of_tool="""
        extract_text(img_path: str) -> str:
            Extract text from an image file using a multimodal model.

            Args:
                img_path: A local image file path (strings).

            Returns:
                A single string containing the concatenated text extracted from each image.
        divide(a: int, b: int) -> float:
            Divide a and b
        """
    image = state["input_file"]
    sys_msg = SystemMessage(content=f"You are a helpful butler named Assistant. You can analyse documents and run computations with provided tools:\n{textual_description_of_tool} \n You have access to some optional images. Currently the loaded image is: {image}")


    return {
        "messages": [llm_with_tools.invoke([sys_msg] + state["messages"])],
        "input_file": state["input_file"]
    }