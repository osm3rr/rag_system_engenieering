# app.py

# IMPORTACIONES
import agent.config # Carga variables de entorno
import os
import streamlit as st
import uuid
from typing import Dict

# Importaciones de LangChain y LangGraph
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Importaciones de módulos
from agent.state import GraphState
from agent.supervisor import DecisionSupervisor
from agent.specialists import agente_documento_01, agente_documento_02
from agent.config import embeddings_cohere, llm_gemini

# CONFIGURACIÓN Y LÓGICA DEL BACKEND
# Lógica de construcción del grafo.

# Directorios
VECTOR_STORE_BASE_PATH = "vector_stores"
PDF_TEMP_PATH = "temp_pdf"
os.makedirs(PDF_TEMP_PATH, exist_ok=True)
os.makedirs(VECTOR_STORE_BASE_PATH, exist_ok=True)

# Nodos del Grafo (supervisor_router, specialist_executor, supervisor_synthesizer)
def supervisor_router_node(state: GraphState) -> Dict:
    print(">> Supervisor (Router): Evaluando a quién delegar...")
    llm_router = llm_gemini.with_structured_output(DecisionSupervisor)
    prompt = SystemMessage(content="""Eres un supervisor de un equipo de agentes. Analiza la consulta del usuario y enrútala al especialista correcto.

Reglas de Enrutamiento:
- 'agente_documento_01': Para preguntas sobre el directorio de proyectos de ingeniería.
- 'agente_documento_02': Para preguntas sobre el manual de procedimientos.
- Si el usuario te pide que verifiques o corrobores información, DEBES enrutar la tarea al especialista correspondiente.
- 'finalizar': Si el usuario se despide.
        """)
    decision = llm_router.invoke([prompt] + state["messages"])
    print(f"   Decisión: Delegar a '{decision.siguiente_agente}'")
    return {"siguiente_agente": decision.siguiente_agente}

def specialist_executor_node(state: GraphState) -> Dict:
    agent_name = state["siguiente_agente"]
    print(f">> Ejecutando Especialista: {agent_name}")
    specialist = {"agente_documento_01": agente_documento_01, "agente_documento_02": agente_documento_02}.get(agent_name)
    if not specialist: raise ValueError(f"Agente desconocido: {agent_name}")
    
    # Llamada síncrona '.invoke()' con el historial completo
    result = specialist.invoke({"messages": state["messages"]})
    
    informe = result['messages'][-1].content
    print(f"   Informe del especialista: {informe[:300]}...")
    return {"informe_especialista": informe}

def supervisor_synthesizer_node(state: GraphState) -> Dict:
    print(">> Supervisor (Synthesizer): Formulando respuesta final...")
    informe = state["informe_especialista"]
    pregunta_usuario = state["messages"][-1].content
    print(f"   Pregunta del usuario: {pregunta_usuario}")
    template = ChatPromptTemplate.from_messages([
        ("system", """Eres un asistente de atención al cliente. Tu única función es tomar la información técnica recuperada por un especialista y presentarla de forma clara y amable al usuario.

Sigue estas reglas de forma OBLIGATORIA:
1.  Basa tu respuesta EXCLUSIVAMENTE en la "Información recuperada por el especialista". NO uses conocimiento previo ni inventes datos.
2.  Si el informe del especialista indica que no se encontró información, tu única respuesta debe ser informar al usuario de que no se encontró la información en el documento.
3.  Si la pregunta del usuario es una continuación de la conversación (p. ej., "y sobre ella?", "dime más"), usa el "Historial de la conversación" para entender el contexto, pero la información para la respuesta DEBE provenir únicamente del informe actual del especialista.
4.  Mantén siempre un tono servicial y dirígete al usuario por su nombre si lo conoces por el historial.
5.  NO justifiques ni inventes explicaciones si los datos del usuario contradicen los del especialista. Simplemente presenta la información del especialista como la fuente de verdad. Por ejemplo, si el usuario dice "La puntuación es X" y el informe dice "La puntuación es Y", tu respuesta debe ser "Según mis datos, la puntuación es Y."."""),
        MessagesPlaceholder(variable_name="historial_chat"),
        ("user", f"Pregunta del usuario:\n{pregunta_usuario}\n\nInformación recuperada por el especialista:\n{informe}")
    ])
    synthesis_chain = template | llm_gemini
    response = synthesis_chain.invoke({
        "historial_chat": state["messages"],
        "informe_tecnico": informe
    })
    print(f"   Respuesta final generada: {response.content[:300]}...")
    return {"messages": [response]}


# Ensamblado del grafo
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

checkpointer = InMemorySaver()
workflow = builder.compile(checkpointer=checkpointer)
print("Grafo Supervisor Multi-RAG (Delegación/Síntesis) compilado y listo para Streamlit.")

# LÓGICA DE LA INTERFAZ DE USUARIO CON STREAMLIT

# Configuración de la página
st.set_page_config(page_title="Ingelect", page_icon="🤖", layout="wide")
st.title("Ingelect 🤖 - Engennier Assistant")

# Barra lateral para carga de documentos
with st.sidebar:
    st.header("Gestión de Documentos")
    
    doc_type = st.selectbox("Elige el tipo de documento a cargar", ["documento_01", "documento_02"])
    
    uploaded_file = st.file_uploader("Sube un archivo PDF", type="pdf", key=f"uploader_{doc_type}")
    
    # Expander para agrupar la configuración avanzada
    with st.expander("Configuración de Procesamiento (Chunking)"):
        
        # st.number_input para el tamaño del chunk
        chunk_size = st.number_input(
            "Tamaño del Chunk", 
            min_value=100, 
            max_value=6000, 
            value=500, 
            step=50, # El valor que se incrementa/decrementa con los botones
            help="El número máximo de caracteres en cada chunk. Afecta el detalle de la información."
        )

        # st.number_input para el overlap del chunk
        chunk_overlap = st.number_input(
            "Solapamiento de Chunks", 
            min_value=0, 
            max_value=2000, 
            value=125, 
            step=25, # El salto más pequeño aquí
            help="Cuántos caracteres se superponen entre chunks para mantener el contexto."
        )
        # ---------------------

    if st.button(f"Procesar Documento: {doc_type}"):
        if uploaded_file is not None:
            with st.spinner(f"Procesando '{uploaded_file.name}'..."):
                
                # Guardado temporal del archivo
                temp_file_path = os.path.join(PDF_TEMP_PATH, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Lógica de procesamiento
                loader = PyPDFLoader(temp_file_path)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                chunks = text_splitter.split_documents(loader.load_and_split())
                
                if chunks:
                    ruta_vector_store = os.path.join(VECTOR_STORE_BASE_PATH, doc_type)
                    vector_store = FAISS.from_documents(chunks, embeddings_cohere)
                    vector_store.save_local(ruta_vector_store)
                    st.success(f"¡'{doc_type}' procesado! ({len(chunks)} chunks)")
                else:
                    st.error("No se pudo extraer texto del PDF.")

                os.remove(temp_file_path)
        else:
            st.warning("Por favor, sube un archivo PDF.")

# Interfaz de Chat
st.divider()

# Inicialización del thread_id y el historial de mensajes en el estado de la sesión de Streamlit
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes del historial guardado en la sesión de Streamlit
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Entrada de usuario
if prompt := st.chat_input("Pregunta sobre proyectos (doc_01) o procedimientos (doc_02)..."):
    # Añadir el mensaje del usuario a la lista de la UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            
            #    Se construye el historial de entrada para el grafo.
            #    Se convierte el historial de diccionarios de la UI al formato que LangGraph espera (HumanMessage, AIMessage).
            #    Esto asegura que el grafo reciba TODA la conversación anterior.
            input_messages = []
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    input_messages.append(HumanMessage(content=msg["content"]))
                else:
                    input_messages.append(AIMessage(content=msg["content"]))

            #    Se prepara el input y la configuración para la llamada.
            #    El input contiene el historial completo.
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            input_data = {"messages": input_messages}
            
            #    Se invoca el workflow con el historial completo.
            final_state = workflow.invoke(input_data, config)
            
            #    Se obtiene solo la última respuesta (la que acaba de generar el agente).
            response_content = final_state["messages"][-1].content
            
            #    Se muestra la respuesta y se añade al historial de la UI.
            st.write(response_content)
            st.session_state.messages.append({"role": "assistant", "content": response_content})