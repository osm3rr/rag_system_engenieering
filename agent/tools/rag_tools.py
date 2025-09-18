# agent/tools/rag_tools.py
import os
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS

# ¡Importamos la instancia ya configurada!
from agent.config import embeddings_cohere

VECTOR_STORE_PATH = "vector_store"

@tool
def pdf_search(query: str):
    """Busca en el documento PDF."""
    print(f"--- Herramienta RAG: Buscando '{query}' ---")
    if not os.path.exists(VECTOR_STORE_PATH) or not os.listdir(VECTOR_STORE_PATH):
        return "Error: Base de datos vectorial no encontrada. Suba un PDF en /upload."

    try:
        # Usamos la instancia importada
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings_cohere, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever()
        results = retriever.invoke(query)
        context = "\n---\n".join([doc.page_content for doc in results])
        print(f"--- Contexto recuperado: {context[:500]}... ---")
        return context
    except Exception as e:
        return f"Error al buscar en la base de datos vectorial: {e}"