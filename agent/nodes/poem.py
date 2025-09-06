# agent/nodes/poem.py
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.state import GraphState

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def poem_node(state: GraphState):
    print(">> Nodo: Poema")
    result = llm.invoke(f"Escribe un poema breve sobre: {state['input']}")
    return {"output": result.content}