# main.py
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.graph import StateGraph, END

# Cargar variables de entorno al inicio
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY no encontrada en el archivo .env")

# --- Importamos nuestras piezas modulares de la carpeta 'agent' ---
from agent.state import GraphState
from agent.nodes import (
    router_node,
    conditional_edge,
    story_node,
    joke_node,
    poem_node,
)

# --- 1. Ensamblaje del Grafo ---
workflow = StateGraph(GraphState)
workflow.add_node("router", router_node)
workflow.add_node("story", story_node)
workflow.add_node("joke", joke_node)
workflow.add_node("poem", poem_node)
workflow.set_entry_point("router")
workflow.add_conditional_edges(
    "router", conditional_edge,
    {"story": "story", "joke": "joke", "poem": "poem"}
)
workflow.add_edge("story", END)
workflow.add_edge("joke", END)
workflow.add_edge("poem", END)
app_graph = workflow.compile()

# --- 2. API con FastAPI ---
app = FastAPI(title="Agente Enrutador con Arquitectura Modular")

class RequestBody(BaseModel):
    query: str

@app.post("/invoke")
def invoke_agent(body: RequestBody):
    """Ejecuta el agente enrutador con la consulta del usuario."""
    initial_state = {"input": body.query}
    final_state = app_graph.invoke(initial_state)
    return {"response": final_state.get("output")}

@app.get("/")
def read_root():
    return {"status": "ok", "message": "API del enrutador modular está en línea."}