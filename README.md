# RAG System Engineering

Este proyecto implementa un sistema de Recuperación Aumentada por Generación (RAG) orientado a la ingeniería, diseñado para asistir en la gestión, consulta y procesamiento de documentos técnicos mediante el uso de agentes inteligentes y almacenamiento vectorial.

## Características principales
- **Agentes Especializados**: Módulos de agentes para tareas específicas (consultas, inscripción, pagos, etc.).
- **Supervisor**: Controla y coordina la interacción entre agentes y herramientas.
- **Almacenamiento Vectorial**: Utiliza FAISS para indexar y buscar información relevante en documentos PDF.
- **Integración con APIs**: Soporte para Google API y Cohere API (ver archivo `.env_example`).
- **Procesamiento de Documentos**: Carpeta `temp_pdf/` para almacenamiento temporal de PDFs y `vector_stores/` para índices vectoriales por documento.

## Estructura del proyecto
```
rag_system_engenieering/
├── app.py                  # Punto de entrada principal
├── requirements.txt        # Dependencias del proyecto
├── agent/
│   ├── config.py           # Configuración de agentes
│   ├── specialists.py      # Definición de agentes especialistas
│   ├── state.py            # Estado global del sistema
│   ├── supervisor.py       # Lógica del supervisor
│   └── tools/
│       └── rag_tools.py    # Herramientas de RAG
├── temp_pdf/               # PDFs temporales
├── vector_stores/          # Índices vectoriales FAISS
└── .env_example            # Ejemplo de variables de entorno
```

## Instalación
1. Clona el repositorio:
   ```bash
   git clone <url-del-repositorio>
   cd rag_system_engenieering
   ```
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Configura las variables de entorno:
   - Copia `.env_example` a `.env` y coloca tus claves de API.

## Uso
Ejecuta la aplicación principal:
```bash
streamlit run app.py
```

## Dependencias principales
- Python 3.12+
- FAISS
- Google API Client
- Cohere
- Otros (ver `requirements.txt`)

## Contribución
Las contribuciones son bienvenidas. Por favor, abre un issue o pull request para sugerencias o mejoras.

## Licencia
Este proyecto está licenciado bajo los términos de la licencia MIT. Consulta el archivo `LICENSE` para más detalles.
