import os
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_ollama import OllamaLLM

# Define the State
class State(TypedDict):
    email: Dict[str, Any]
    email_category : Optional[str]
    spam_reason : Optional[str]
    is_spam: Optional[bool]
    email_draft:Optional[str]
    messages: List[Dict[str, Any]]


# Definethe nodes
model = OllamaLLM(name= "gemma")


# Edge:

# StateGraph:

# Visualisation

# Invokation