# agent/specialists.py
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ¡Importamos el LLM ya configurado!
from agent.config import llm_gemini
from .tools.rag_tools import pdf_search

tools = [pdf_search]

system_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Eres un asistente experto en productos eléctricos de la marca NDU. Tu única fuente de conocimiento es un documento técnico que se te ha proporcionado.

Sigue estas reglas estrictamente:

1.  Para responder CUALQUIER pregunta del usuario, DEBES usar la herramienta `pdf_search`. Es tu única herramienta.
2.  Basa tu respuesta EXCLUSIVAMENTE en la información devuelta por la herramienta `pdf_search`.
3.  Si la herramienta `pdf_search` no encuentra información relevante o la pregunta no está relacionada con el documento, responde amablemente que no tienes la información en el documento.
4.  NO inventes respuestas ni utilices tu conocimiento general. Si la información no está en el documento, no la sabes."""
        ), # Mantén tu prompt aquí
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Creamos el agente usando el LLM importado
agente_rag = create_react_agent(llm_gemini, tools=tools, prompt=system_prompt)