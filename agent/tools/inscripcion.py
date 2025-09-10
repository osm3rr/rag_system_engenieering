# agent/tools/inscripcion.py
from langchain_core.tools import tool

@tool
def verificar_disponibilidad(formacion: str):
    """Verifica si hay cupos disponibles para una formación."""
    print(f"--- Herramienta: Verificando disponibilidad para '{formacion}' ---")
    return "Sí, hay cupos disponibles."

# ... otras herramientas de inscripción ...