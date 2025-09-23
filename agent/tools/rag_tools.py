# Importciones
import os
from langchain_core.tools import tool, BaseTool
from langchain_community.vectorstores import FAISS

# Importacion de instancia desde config
from agent.config import embeddings_cohere

VECTOR_STORE_PATH = "vector_store"

def crear_herramienta_rag(nombre_documento: str, descripcion: str) -> BaseTool:
    """
    Fábrica que crea una herramienta de búsqueda RAG para un documento específico.
    """
    nombre_herramienta = f"buscar_en_{nombre_documento}"

    @tool
    def pdf_search(query: str):
        # --- ¡DOCSTRING AÑADIDO AQUÍ! ---
        """Busca información en el documento especificado para responder a la consulta del usuario."""
        # ---------------------------------
        ruta_vector_store = os.path.join(VECTOR_STORE_PATH, nombre_documento)
        print(f"--- Herramienta RAG ({nombre_documento}): Buscando '{query}' ---")

        if not os.path.exists(ruta_vector_store) or not os.listdir(ruta_vector_store):
            return f"Error: La base de datos para '{nombre_documento}' no existe."
        try:
            vector_store = FAISS.load_local(ruta_vector_store, embeddings_cohere, allow_dangerous_deserialization=True)
            retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'fetch_k': 20})
            results = retriever.invoke(query)
            context = "\n---\n".join([doc.page_content for doc in results])
            return context if context else "No se encontró información relevante."
        except Exception as e:
            return f"Error al buscar en '{nombre_documento}': {e}"
            
    pdf_search.description = descripcion
    return pdf_search

herramienta_doc_01 = crear_herramienta_rag(
    nombre_documento="documento_01", 
    descripcion="Busca en el documento 01, que es una guia de peliculas"
)

herramienta_doc_02 = crear_herramienta_rag(
    nombre_documento="documento_02",
    descripcion="Busca en el documento 02, que es una guia de series"
)