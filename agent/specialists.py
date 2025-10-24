# agent/specialists.py
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from agent.config import llm_gemini
from .tools.rag_tools import herramienta_doc_01, herramienta_doc_02

prompt_doc_01 = ChatPromptTemplate.from_messages([
    ("system", "Eres un experto en el directorio de proyectos de ingeniería (documento 01). Usa tu herramienta para encontrar proyectos específicos."),
    MessagesPlaceholder(variable_name="messages"),
])
agente_documento_01 = create_react_agent(llm_gemini, tools=[herramienta_doc_01], prompt=prompt_doc_01)

prompt_doc_02 = ChatPromptTemplate.from_messages([
    ("system", "Eres un experto en el manual de procedimientos (documento 02). Usa tu herramienta para encontrar información sobre los procedimientos."),
    MessagesPlaceholder(variable_name="messages"),
])
agente_documento_02 = create_react_agent(llm_gemini, tools=[herramienta_doc_02], prompt=prompt_doc_02)