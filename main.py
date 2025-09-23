#Importaciones
import agent.config
import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import Dict

#Importaciones de LangChain y LangGraph
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Importamos desde otras carpetas
from agent.state import GraphState
from agent.supervisor import DecisionSupervisor
from agent.specialists import agente_documento_01, agente_documento_02
from agent.config import embeddings_cohere, llm_gemini

# Directorios
VECTOR_STORE_PATH = "vector_store"
PDF_TEMP_PATH = "temp_pdf"
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(PDF_TEMP_PATH, exist_ok=True)

# Nodos

# Nodo Supervisor Router
def supervisor_router_node(state: GraphState) -> Dict:
    print(">> Supervisor (Router): Evaluando a quién delegar...")
    llm_router = llm_gemini.with_structured_output(DecisionSupervisor)
    prompt = SystemMessage(content="""Eres un supervisor de un equipo de agentes. Analiza la consulta del usuario y enrútala al especialista correcto.
        - 'agente_documento_01': Para preguntas sobre peliculas.
        - 'agente_documento_02': Para preguntas sobre series.
        - 'finalizar': Si el usuario se despide.
        """)
    decision = llm_router.invoke([prompt] + state["messages"]) # Pasa todo el historial de mensajes
    print(f"   Decisión: Delegar a '{decision.siguiente_agente}'")
    return {"siguiente_agente": decision.siguiente_agente}

# Nodo Especialista
async def specialist_executor_node(state: GraphState) -> Dict:
    agent_name = state["siguiente_agente"]
    print(f">> Ejecutando Especialista: {agent_name}")
    specialist = {"agente_documento_01": agente_documento_01, "agente_documento_02": agente_documento_02}.get(agent_name)
    if not specialist: raise ValueError(f"Agente desconocido: {agent_name}")
    
    # Los especialistas solo necesitan el último mensaje para hacer la búsqueda RAG
    result = await specialist.ainvoke({"messages": state["messages"]}) # Pasa todo el historial de mensajes
    informe = result['messages'][-1].content
    print(f"   Informe del especialista: {informe[:300]}...")
    return {"informe_especialista": informe}

# Nodo Supervisor Sintetizador
def supervisor_synthesizer_node(state: GraphState) -> Dict:
    print(">> Supervisor (Synthesizer): Formulando respuesta final...")
    informe = state["informe_especialista"]
    pregunta_usuario = state["messages"][-1].content # Ultimo mensaje del usuario
    print(f"   Pregunta del usuario: {pregunta_usuario}") #Importante para debug
    template = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente servicial y experto. Tu tarea es tomar la pregunta de un usuario y la información recuperada por un especialista, y formular una respuesta clara y amable en español."),
        MessagesPlaceholder(variable_name="historial_chat"), # Mantener el historial del chat
        ("user", f"Pregunta del usuario:\n{pregunta_usuario}\n\nInformación recuperada por el especialista:\n{informe}")
    ])

    synthesis_chain = template | llm_gemini #cadena une la plantilla con LLM

    response = synthesis_chain.invoke({
        "historial_chat": state["messages"],
        "informe_tecnico": informe
    })
    print(f"   Respuesta final generada: {response.content[:300]}...")
    return {"messages": [response]}

# Ensamblando el grafo
builder = StateGraph(GraphState)
builder.add_node("supervisor_router", supervisor_router_node)
builder.add_node("specialist_executor", specialist_executor_node)
builder.add_node("supervisor_synthesizer", supervisor_synthesizer_node)

builder.set_entry_point("supervisor_router")
builder.add_conditional_edges(
    "supervisor_router",
    lambda state: state["siguiente_agente"],
    {"agente_documento_01": "specialist_executor", "agente_documento_02": "specialist_executor", "finalizar": END}
)
builder.add_edge("specialist_executor", "supervisor_synthesizer")
builder.add_edge("supervisor_synthesizer", END)

# Compilando el grafo
checkpointer = InMemorySaver()
workflow = builder.compile(checkpointer=checkpointer)
print("Grafo Supervisor Multi-RAG (Delegación/Síntesis) compilado y listo.")

# Aplicación FastAPI
app = FastAPI(title="Servidor Supervisor Multi-RAG")

# Carga de documentos PDF y creación de vector stores
@app.post("/upload")
async def upload_pdf(doc_type: str = Form(...), file: UploadFile = File(...), chunk_size: int = Form(500), chunk_overlap: int = Form(125)):
    if doc_type not in ["documento_01", "documento_02"]:
        raise HTTPException(status_code=400, detail="doc_type debe ser 'documento_01' o 'documento_02'.")
    
    ruta_vector_store = os.path.join(VECTOR_STORE_PATH, doc_type)
    os.makedirs(ruta_vector_store, exist_ok=True)
    temp_file_path = os.path.join(PDF_TEMP_PATH, file.filename)

    try:
        with open(temp_file_path, "wb") as f: f.write(await file.read())
        loader = PyPDFLoader(temp_file_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(loader.load_and_split())
        vector_store = FAISS.from_documents(chunks, embeddings_cohere)
        vector_store.save_local(ruta_vector_store)
        return {"message": f"Documento para '{doc_type}' procesado con {len(chunks)} chunks."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file_path): os.remove(temp_file_path)

class QueryBody(BaseModel):
    thread_id: str
    query: str

# Consultas al agente supervisor Router
@app.post("/query")
async def query_agent(body: QueryBody):
    config = {"configurable": {"thread_id": body.thread_id}}
    try:
        # 1. Preparamos la entrada para el agente con el nuevo mensaje del usuario.
        input_data = {"messages": [HumanMessage(content=body.query)]}
        
        # 2. Ejecutamos el stream, lo consumimos para asegurarnos de que el grafo se ejecute hasta el final.
        async for _ in workflow.astream(input_data, config):
            pass

        # 3. Una vez que el stream ha terminado, obtenemos explícitamente el estado final y completo
        #    de la conversación usando el mismo config (que contiene el thread_id).
        final_state_snapshot = await workflow.aget_state(config)
        
        # 4. Extraemos la lista de mensajes del estado final recuperado.
        final_messages = final_state_snapshot.values["messages"]
        
        # 5. Devolvemos el contenido del último mensaje.
        return {"response": final_messages[-1].content}

    except Exception as e:
        # Es útil imprimir el traceback en la consola para depuración
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ocurrió un error al invocar al agente: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)