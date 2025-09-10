# agent/state.py
# agent/state.py
from typing import TypedDict, Annotated, Optional, Literal
from langgraph.graph.message import add_messages

# El Estado Global que será compartido a través de todo el grafo.
# Cada agente podrá leer y escribir en él.
class EstadoGlobal(TypedDict):
    # El historial de la conversación es fundamental.
    messages: Annotated[list, add_messages]
    
    # La decisión del supervisor sobre a qué agente llamar a continuación.
    siguiente_agente: str
    
    # Datos que se pueden ir recopilando durante el flujo.
    perfil_participante: Optional[dict]
    formacion_seleccionada: Optional[str]
    disponibilidad_confirmada: bool
    monto_a_pagar: Optional[float]
    moneda: Optional[Literal["USD", "BOLIVARES"]]
    pago_verificado: bool