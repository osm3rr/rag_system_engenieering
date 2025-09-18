# agent/config.py
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
# ¡NUEVAS IMPORTACIONES PARA COHERE!
from langchain_cohere import CohereEmbeddings

# --- CARGA CENTRALIZADA DE VARIABLES DE ENTORNO ---
load_dotenv()

# --- VERIFICACIÓN Y CREACIÓN DE INSTANCIAS ---

# Verificamos la clave de Google para el LLM
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("La GOOGLE_API_KEY no ha sido configurada en el archivo .env")

# Verificamos la clave de Cohere para los Embeddings
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("La COHERE_API_KEY no ha sido configurada en el archivo .env")

# --- INSTANCIAS DE SERVICIOS COMPARTIDOS ---

# El LLM sigue siendo Gemini de Google
llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=GOOGLE_API_KEY
)

# ¡CAMBIO AQUÍ! Ahora usamos Cohere para los embeddings.
# Pasamos la clave directamente para mayor robustez.
embeddings_cohere = CohereEmbeddings(
    model="embed-english-light-v3.0", 
    cohere_api_key=COHERE_API_KEY
)