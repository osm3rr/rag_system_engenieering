# agent/nodes/story.py
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.state import GraphState

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def story_node(state: GraphState):
    print(">> Nodo: Historia")
    result = llm.invoke(f"Escribe una historia corta sobre: {state['input']}")
    return {"output": result.content}