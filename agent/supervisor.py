# agent/supervisor.py
from pydantic import BaseModel, Field
from typing import Literal

class DecisionSupervisor(BaseModel):
    siguiente_agente: Literal["agente_documento_01", "agente_documento_02", "finalizar"] = Field(
        description="Elige el agente especialista más adecuado para la consulta del usuario, si la consulta es del tipo conversación casual, actua como un" \
        "recepcionista amable y preguntale si quiere tener infomación sobre peliculas (documento_01) o series (documento_02). Si consideras que el usuario ya" \
        "obtuvo toda la informacion requerida, pregunta si desea continuar con otra consulta y de no ser así, ve a 'finalizar'."
    )