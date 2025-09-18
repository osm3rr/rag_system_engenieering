# agent/state.py
from langgraph.prebuilt.chat_agent_executor import AgentState

# Usamos el AgentState predefinido. Es un TypedDict que ya incluye 
# la clave 'messages' con el reducer add_messages, perfecto para agentes.
class GraphState(AgentState):
    pass