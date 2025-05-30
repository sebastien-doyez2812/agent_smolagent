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


vision_llm = OllamaLLM(model = "llava")

def extract_text(img_path:str) -> str:
    """
    Extract text from an image file using a multimodal model


    """
    all_text = ""

    try:
        # with open(img_path, 'rb') as image_file:
        #     image_bytes = image_file.read()

        # image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        # message = [
        #     HumanMessage(
        #         content=[
        #              {
        #                 "type": "text",
        #                 "text": (
        #                     "Extract all the text from this image. "
        #                     "Return only the extracted text, no explanations."
        #                 ),
        #             },
        #             {
        #                 "type": "image_path",
        #                 "image_path": {
        #                     "path": img_path
        #                 },
        #             },
        #         ]
        #     )
        # ]
        rep = vision_llm.invoke(f"Extract the text from the image stored at {img_path}")

        print(f"rep from llava is {rep}")
        all_text += rep +"\n\n"
        return all_text.strip()
    except Exception as e:
        error_msg = f"Error: {e}"
        print(error_msg)
        return ""
    

def divide(a: int, b:int) -> float:
    """
    Divide a by b and return the result
    """
    return a/b


tools = [
    divide,
    extract_text
]


llm = ChatOllama(model= "qwen2.5")

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
    sys_msg = SystemMessage(content= "You are a helpful assistant. Your tools are:\n"
        "- extract_text(img_path): Extract text from an image.\n"
        "- divide(a, b): Perform division.\n\n"
        "Please respond with actions based on the task provided. "
        "If an image is provided, assume it is for text extraction." \
        f"If the user ask you to describe a picture, it's this image: Currently the loaded image is: {image}")


    return {
        "messages": [llm_with_tools.invoke([sys_msg] + state["messages"])],
        "input_file": state["input_file"]
    }


builder = StateGraph(AgentState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))


builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
react_graph = builder.compile()


#Test:
# Calculator:
# msg = [HumanMessage("Divide 5 by 2")]
# messages = react_graph.invoke({"message": msg, "input_file": None})

# # Show the messages
# for m in messages['messages']:
#     m.pretty_print()

# # Describe:
msg = [HumanMessage(content= "Describe the image")]
messages = react_graph.invoke({"message": msg, "input_file": "C:/Users/doyez/OneDrive/Images/Screenshots/Capture.png" })

# Show the messages
print("message = ", messages)
for m in messages['messages']:
    m.pretty_print()