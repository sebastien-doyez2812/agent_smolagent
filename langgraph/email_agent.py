import os
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_ollama import OllamaLLM
from IPython.display import Image, display
from langfuse.callback import CallbackHandler

# API key for langfuse:
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-1fe75db7-9979-4139-8f8a-69add99f16c1" 
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-21a66749-d2b2-4ab2-b23b-2e3f9c70963c"
os.environ["LANGFUSE_HOST"] = "https://us.cloud.langfuse.com"


langfuse_handler = CallbackHandler()
# Define the State
class EmailState(TypedDict):
    email: Dict[str, Any]
    email_category : Optional[str]
    spam_reason : Optional[str]
    is_spam: Optional[bool]
    email_draft:Optional[str]
    messages: List[Dict[str, Any]]


# Definethe nodes
model = OllamaLLM(model="gemma")

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
    response_text = rep.lower()
    print(response_text)
    is_spam = not "legitimate" in response_text

    spam_reason = None
    if is_spam and "reason" in response_text:
        spam_reason = response_text.split("reason:")[1].strip()

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
        {"role": "assistant", "content": rep}
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
        {"role": "assistant", "content": response}
    ]
    
    # Return state updates
    return {
        "email_draft": response,
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
email_graph = StateGraph(EmailState)

# Add the node:
email_graph.add_node("read_email", read_email)
email_graph.add_node("classify_email", classify_email)
email_graph.add_node("handle_spam", handle_spam)
email_graph.add_node("draft_rep", draft_response)
email_graph.add_node("send_to_validate", send_back_for_validation)

# Make the edge (links between the nodes)
email_graph.add_edge(START, "read_email")
email_graph.add_edge("read_email", "classify_email")
email_graph.add_conditional_edges("classify_email",
route_email,
    {
        "spam": "handle_spam",
        "legitimate": "draft_rep"
    }   
)
email_graph.add_edge("handle_spam", END)
email_graph.add_edge("draft_rep", "send_to_validate" )
email_graph.add_edge("send_to_validate", END)

compiled_graph = email_graph.compile()

# Visualisation
display(Image(compiled_graph.get_graph().draw_mermaid_png()))


# Invokation

legitimate_email = {
    "sender": "john.smith@example.com",
    "subject": "Question about your services",
    "body": "Dear Mr. Doyez, I was referred to you by a colleague and I'm interested in learning more about your consulting services. Could we schedule a call next week? Best regards, John Smith"
}

# Tests the SDK connection with the server
langfuse_handler.auth_check()

print("\nProcessing legitimate email...")
legitimate_result = compiled_graph.invoke({
    "email": legitimate_email,
    "is_spam": None,
    "spam_reason": None,
    "email_category": None,
    "email_draft": None,
    "messages": []
},
config={"callbacks": [langfuse_handler]})

