# agent/tools/consultas.py
from langchain_core.tools import tool

@tool
def proveer_informacion_formaciones(tema: str):
    """Provee información detallada sobre formaciones disponibles."""
    print(f"--- Herramienta: Buscando información sobre '{tema}' ---")
    return f"Aquí tienes la información sobre las formaciones de '{tema}'."

@tool
def recomendar_formaciones(perfil: dict):
    """Recomienda formaciones basadas en el perfil del participante."""
    print(f"--- Herramienta: Recomendando formaciones para perfil {perfil} ---")
    return "Basado en tu perfil, te recomendamos el curso de 'Introducción a la IA'."