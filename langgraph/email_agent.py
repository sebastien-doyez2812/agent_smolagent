import os
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_ollama import OllamaLLM

# Define the State
class EmailState(TypedDict):
    email: Dict[str, Any]
    email_category : Optional[str]
    spam_reason : Optional[str]
    is_spam: Optional[bool]
    email_draft:Optional[str]
    messages: List[Dict[str, Any]]


# Definethe nodes
model = OllamaLLM(name= "gemma")

def read_email(state: EmailState):
    email = state["email"]

    print(f"The Agent is reading an email from {email["sender"]}, subject: {email["subject"]}")
    return {}


def classify_email(state: EmailState):
    email = state["email"]
    prompt = f"""
    As an agent, analyze this email and determine if it is spam or legitimate.
    
    Email:
    From: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}
    
    First, determine if this email is spam. If it is spam, explain why.
    If it is legitimate, categorize it (inquiry, complaint, thank you, etc.).
    """

    messages = [HumanMessage(content=prompt)]
    rep = model.invoke(messages)

    # Is the email a spam?
    response_text = rep.content.lower()
    is_spam = "spam" in response_text and "not spam" not in response_text

    spam_reason = None
    if is_spam and "reason" in response_text:
        spam_reason = response_text.spilt("reason:")[1].strip()

    # Check for the category:
    email_category = None
    if not is_spam:
        categories = ["inquiry", "complaint", "thank you", "request", "information"]
        for category in categories:
            if category in response_text:
                email_category = category
                break

    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.content}
    ]

    return{
        "is_spam": is_spam,
        "spam_reason": spam_reason,
        "email_category": email_category,
        "messages": new_messages
    }


def handle_spam(state : EmailState):


def write_back(state : EmailState):



def send_back_for_validation(state: EmailState):
# Edge:

# StateGraph:

# Visualisation

# Invokation