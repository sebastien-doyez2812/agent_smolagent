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

def handle_spam(state: EmailState):
    print(f"The email is a Spam, Reason: {state['spam_reason']}")
    print("The email has been moved to the spam folder.")
    
    return {}

def draft_response(state: EmailState):
    """Alfred drafts a preliminary response for legitimate emails"""
    email = state["email"]
    category = state["email_category"] or "general"
    
    # Prepare our prompt for the LLM
    prompt = f"""
    As an assistant, draft a polite preliminary response to this email.
    
    Email:
    From: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}
    
    This email has been categorized as: {category}
    
    Draft a brief, professional response that Mr. Doyez can review and personalize before sending.
    """
    
    # Call the LLM
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)
    
    # Update messages for tracking
    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.content}
    ]
    
    # Return state updates
    return {
        "email_draft": response.content,
        "messages": new_messages
    }


def send_back_for_validation(state: EmailState):
    email = state["email"]
    
    print("\n" + "="*50)
    print(f"Sir, you've received an email from {email['sender']}.")
    print(f"Subject: {email['subject']}")
    print(f"Category: {state['email_category']}")
    print("\nI've prepared a draft response for your review:")
    print("-"*50)
    print(state["email_draft"])
    print("="*50 + "\n")
    
    # We're done processing this email
    return {}


# Edge:
def route_email(state: EmailState) ->str:
    if state["is_spam"]:
        return "spam"
    else:
        return "legitimate"

# StateGraph:

# Visualisation

# Invokation