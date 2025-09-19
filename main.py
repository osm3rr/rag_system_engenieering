# main.py

# --- IMPORTACIÓN ESTRATÉGICA DE CONFIGURACIÓN ---
# Al importar este módulo primero, nos aseguramos de que .env se cargue
# antes de que cualquier otro módulo intente usar las variables de entorno.
import agent.config

# --- IMPORTACIONES ESTÁNDAR Y DE TERCEROS ---
import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

# --- IMPORTACIONES DE LANGCHAIN Y LANGGRAPH ---
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import InMemorySaver

# --- IMPORTACIONES DE NUESTROS MÓDULOS ---
# Importamos las piezas ya construidas de nuestra arquitectura.
from agent.state import GraphState
from agent.specialists import agente_rag
from agent.config import embeddings_cohere  # Importamos la instancia de embeddings

# --- 1. CONFIGURACIÓN DE RUTAS Y DIRECTORIOS ---
VECTOR_STORE_PATH = "vector_store"
PDF_TEMP_PATH = "temp_pdf"
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(PDF_TEMP_PATH, exist_ok=True)

# --- 2. ENSAMBLAJE DEL GRAFO DE LANGGRAPH ---
# La estructura del grafo es simple: solo tiene un nodo, nuestro agente RAG.
builder = StateGraph(GraphState)

builder.add_node("agente_rag", agente_rag)
builder.set_entry_point("agente_rag")
builder.set_finish_point("agente_rag")

# Compilamos el grafo con un checkpointer para habilitar la memoria entre turnos.
checkpointer = InMemorySaver()
workflow = builder.compile(checkpointer=checkpointer)

print("Grafo RAG con un solo agente, compilado y listo.")

# --- 3. INICIALIZACIÓN DE LA APP FASTAPI ---
app = FastAPI(
    title="Servidor RAG con Agente Único",
    description="Sube un PDF a /upload y haz preguntas en /query."
)

# --- 4. ENDPOINTS DE LA API ---

@app.post("/upload")
async def upload_and_vectorize_pdf(file: UploadFile = File(...)):
    """
    Endpoint para subir un archivo PDF. El archivo se procesa, se vectoriza
    y se guarda en una base de datos vectorial FAISS persistente.
    """
    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="El archivo debe ser un PDF.")

    temp_file_path = os.path.join(PDF_TEMP_PATH, file.filename)
    try:
        # 1. Guardar el archivo subido temporalmente en el disco.
        with open(temp_file_path, "wb") as f:
            f.write(await file.read())
        
        print(f"Archivo '{file.filename}' subido. Procesando...")

        # 2. Cargar el documento PDF y dividirlo en chunks.
        loader = PyPDFLoader(temp_file_path)
        pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1774, chunk_overlap=225)
        chunks = text_splitter.split_documents(pages)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No se pudo extraer texto del PDF.")

        print(f"Documento dividido en {len(chunks)} chunks. Vectorizando...")

        # 3. Crear la base de datos vectorial usando los chunks y el modelo de embeddings.
        vector_store = FAISS.from_documents(chunks, embeddings_cohere)

        # 4. Guardar el índice de FAISS localmente para persistencia.
        vector_store.save_local(VECTOR_STORE_PATH)

        print(f"Base de datos vectorial creada y guardada en '{VECTOR_STORE_PATH}'.")
        
        return {"message": f"Archivo '{file.filename}' procesado exitosamente.", "chunks_created": len(chunks)}
    
    except Exception as e:
        # Capturamos cualquier error durante el proceso para dar una respuesta clara.
        raise HTTPException(status_code=500, detail=f"Ocurrió un error al procesar el archivo: {e}")
    finally:
        # 5. Asegurarnos de eliminar el archivo temporal después del proceso.
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

class QueryBody(BaseModel):
    thread_id: str
    query: str

@app.post("/query")
async def query_agent(body: QueryBody):
    """
    Endpoint para hacer preguntas al agente. Usa .invoke() para obtener la respuesta final.
    """
    config = {"configurable": {"thread_id": body.thread_id}}
    
    try:
        # Usamos .invoke() que espera hasta que el grafo termine y devuelve el estado final.
        final_state = await workflow.ainvoke({"messages": [("user", body.query)]}, config)
        
        # La respuesta para el usuario es el contenido del último mensaje en el estado final.
        response_content = final_state["messages"][-1].content
        
        return {"response": response_content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ocurrió un error al invocar al agente: {e}")
# --- BLOQUE PARA EJECUCIÓN DIRECTA ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)