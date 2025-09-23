from langgraph.prebuilt.chat_agent_executor import AgentState

# AgentState predefinido.
class GraphState(AgentState):
    siguiente_agente: str
    informe_especialista: str