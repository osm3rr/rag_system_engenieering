# agent/supervisor.py
from pydantic import BaseModel, Field
from typing import Literal

# El esquema de Pydantic que fuerza al LLM a tomar una decisión estructurada.
# Esto es la clave de la robustez.
class DecisionSupervisor(BaseModel):
    """Determina el siguiente paso en la conversación."""
    siguiente_agente: Literal[
        "agente_consultas", 
        "agente_inscripcion", 
        "agente_pagos", 
        "finalizar"
    ] = Field(
        description="Elige el agente especialista más adecuado o 'finalizar' si la conversación ha concluido."
    )