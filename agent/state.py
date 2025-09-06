# agent/state.py
from typing import TypedDict

class GraphState(TypedDict):
    """El estado que fluye a través de nuestro grafo."""
    input: str
    decision: str
    output: str