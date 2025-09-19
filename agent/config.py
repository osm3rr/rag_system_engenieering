import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import CohereEmbeddings

#VARIABLES DE ENTORNO
load_dotenv()

#Google para el LLM
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("La GOOGLE_API_KEY no ha sido configurada en el archivo .env")

#Cohere para los Embeddings
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("La COHERE_API_KEY no ha sido configurada en el archivo .env")

# El LLM
llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=GOOGLE_API_KEY
)

# Embeddings.
embeddings_cohere = CohereEmbeddings(
    model="embed-english-light-v3.0", 
    cohere_api_key=COHERE_API_KEY
)