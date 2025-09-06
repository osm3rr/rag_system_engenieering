# agent/nodes/router.py
from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.state import GraphState

# Inicializamos el LLM aquí, ya que será usado por el enrutador
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class Route(BaseModel):
    step: Literal["poem", "story", "joke"]

router_llm = llm.with_structured_output(Route)

def router_node(state: GraphState):
    print(">> Nodo: Enrutador")
    decision = router_llm.invoke([
        SystemMessage(content="Decide si el usuario quiere un poema ('poem'), una historia ('story'), o un chiste ('joke')."),
        HumanMessage(content=state["input"])
    ])
    return {"decision": decision.step}

def conditional_edge(state: GraphState):
    return state["decision"]