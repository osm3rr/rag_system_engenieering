# agent/nodes/joke.py
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.state import GraphState

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def joke_node(state: GraphState):
    print(">> Nodo: Chiste")
    result = llm.invoke(f"Cuenta un chiste sobre: {state['input']}")
    return {"output": result.content}