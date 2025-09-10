# agent/specialists.py
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from .tools.consultas import proveer_informacion_formaciones, recomendar_formaciones
from .tools.inscripcion import verificar_disponibilidad
from .tools.pagos import calcular_monto_bcv, verificar_pago

# Carga variable de entorno
load_dotenv()

# agent/specialists.py
import os # Asegúrate de que 'os' esté importado
from langchain_google_genai import ChatGoogleGenerativeAI

# ...

# Le pasamos explícitamente la API key que cargamos desde .env
# LLM que usarán nuestros agentes (con la corrección de la API key)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# --- Agente 1: Consultas ---
herramientas_consultas = [proveer_informacion_formaciones, recomendar_formaciones]
prompt_consultas = ChatPromptTemplate.from_messages(
    [
        ("system", "Eres un amable asistente de Adakademy. Tu objetivo es proveer información sobre formaciones y servicios."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
agente_consultas = create_react_agent(llm, tools=herramientas_consultas, prompt=prompt_consultas)

# --- Agente 2: Inscripción ---
herramientas_inscripcion = [verificar_disponibilidad]
prompt_inscripcion = ChatPromptTemplate.from_messages(
    [
        ("system", "Eres un asistente de inscripciones. Tu objetivo es guiar al usuario en el proceso de registro."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
agente_inscripcion = create_react_agent(llm, tools=herramientas_inscripcion, prompt=prompt_inscripcion)

# --- Agente 3: Pagos ---
herramientas_pagos = [calcular_monto_bcv, verificar_pago]
prompt_pagos = ChatPromptTemplate.from_messages(
    [
        ("system", "Eres un asistente de pagos. Tu objetivo es procesar y verificar los pagos de las inscripciones."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
agente_pagos = create_react_agent(llm, tools=herramientas_pagos, prompt=prompt_pagos)