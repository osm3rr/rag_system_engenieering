# main.py
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Carga variable de entorno
load_dotenv()

# Importamos las piezas de nuestra arquitectura
from agent.state import EstadoGlobal
from agent.supervisor import DecisionSupervisor
from agent.specialists import agente_consultas, agente_inscripcion, agente_pagos

# --- NODO SUPERVISOR ---
def supervisor_node(state: EstadoGlobal) -> dict:
    print(">> Supervisor evaluando...")
    router_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    llm_router = router_llm.with_structured_output(DecisionSupervisor)
    
    # Creamos un prompt para el supervisor
    system_prompt = SystemMessage(
        content="""Eres el supervisor de un equipo de asistentes de IA. 
        Analiza el historial de la conversación y decide cuál de los siguientes agentes debe actuar a continuación:
        - agente_consultas: Útil para responder preguntas sobre productos, servicios y formaciones.
        - agente_inscripcion: Útil para gestionar el proceso de registro, disponibilidad y confirmación.
        - agente_pagos: Útil para todo lo relacionado con pagos, divisas y verificación.
        - finalizar: Útil si la conversación ha terminado o el usuario se ha despedido.
        Responde únicamente con el nombre del agente a activar."""
    )
    
    # El supervisor solo ve el historial de mensajes para tomar su decisión
    decision = llm_router.invoke([system_prompt] + state["messages"])
    print(f"   Decisión del Supervisor: Enviar a '{decision.siguiente_agente}'")
    return {"siguiente_agente": decision.siguiente_agente}

# --- ENSAMBLAJE DEL GRAFO PRINCIPAL ---
builder = StateGraph(EstadoGlobal)

# Añadimos los nodos: el supervisor y cada agente especialista
builder.add_node("supervisor", supervisor_node)
builder.add_node("agente_consultas", agente_consultas)
builder.add_node("agente_inscripcion", agente_inscripcion)
builder.add_node("agente_pagos", agente_pagos)

# El punto de entrada es siempre el supervisor
builder.set_entry_point("supervisor")

# Aristas condicionales: desde el supervisor hacia los especialistas
builder.add_conditional_edges(
    "supervisor",
    lambda state: state.get("siguiente_agente"),
    {
        "agente_consultas": "agente_consultas",
        "agente_inscripcion": "agente_inscripcion",
        "agente_pagos": "agente_pagos",
        "finalizar": END
    }
)

# Aristas de retorno: después de que un especialista termina, vuelve al supervisor
builder.add_edge("agente_consultas", END)
builder.add_edge("agente_inscripcion", END)
builder.add_edge("agente_pagos", END)

# Compilamos el grafo final
workflow_supervisor = builder.compile()

# --- API con FastAPI ---
app = FastAPI(title="Agente Supervisor de Adakademy")

class RequestBody(BaseModel):
    query: str

@app.post("/invoke")
def invoke_agent(body: RequestBody):
    initial_state = {"messages": [("user", body.query)]}
    final_state = workflow_supervisor.invoke(initial_state)
    return {"response": final_state["messages"][-1].content}