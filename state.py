"""
state.py
--------
Esquema de estado compartido del orquestador y los "artefactos"
(Pydantic) que cada agente especialista produce.

Por qué esto resuelve dos de los puntos de la consigna a la vez:

1. "Estado Compartido Estructurado ... que permita rastrear qué agente
   ha contribuido con qué información": en vez de que todo viva mezclado
   dentro de `messages` (texto libre), cada especialista escribe su
   aporte en SU PROPIO campo tipado (`investigacion`, `analisis`). El
   supervisor puede entonces saber exactamente qué se hizo y qué falta
   mirando esos campos, sin tener que releer ni interpretar la
   conversación completa.

2. "Gestión de Estado y Validación": cada artefacto es un modelo
   Pydantic con sus propias reglas (scores entre 0 y 1, campos
   obligatorios) -- un dato que no cumple la forma esperada no puede
   llegar a formar parte del estado.
"""

from typing import List, Literal, Optional

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Artefacto del Investigador
# ---------------------------------------------------------------------------
class NoticiaIA(BaseModel):
    """Una noticia simulada encontrada por el Investigador."""

    titular: str
    fuente: str
    fecha: str
    resumen: str = Field(min_length=10, description="Resumen breve de la noticia.")


class ResultadoInvestigacion(BaseModel):
    """Lo que el Investigador aporta al estado compartido."""

    tema_buscado: str
    noticias: List[NoticiaIA] = Field(default_factory=list)

    @property
    def encontro_resultados(self) -> bool:
        return len(self.noticias) > 0


# ---------------------------------------------------------------------------
# Artefacto del Analista
# ---------------------------------------------------------------------------
class AnalisisPorNoticia(BaseModel):
    """El sentimiento detectado para UNA noticia puntual."""

    titular: str
    sentimiento: Literal["positivo", "neutral", "negativo"]
    score: float = Field(ge=0.0, le=1.0, description="0 = muy negativo, 1 = muy positivo.")


class ResultadoAnalisis(BaseModel):
    """Lo que el Analista aporta al estado compartido."""

    sentimiento_general: Literal["positivo", "neutral", "negativo", "sin_datos"]
    score_promedio: float = Field(ge=0.0, le=1.0)
    detalle: List[AnalisisPorNoticia] = Field(default_factory=list)
    # False cuando no había noticias para analizar -- es la señal que usa
    # el Supervisor para decidir si hace falta volver a investigar en vez
    # de cerrar la conversación con una respuesta vacía.
    suficiente: bool


# ---------------------------------------------------------------------------
# Estado global del grafo
# ---------------------------------------------------------------------------
# Hereda de MessagesState (así conserva `messages` con su reducer
# add_messages, por si se quiere loguear la conversación de alto nivel),
# y le agrega los campos propios de este ejercicio -- ninguno de estos
# campos nuevos necesita un reducer: cada uno se REEMPLAZA entero cuando
# el nodo correspondiente corre, igual que en el ejercicio del pipeline
# Investigador/Revisor.
class AgentState(MessagesState):
    topic: str

    # Control del Supervisor
    next_agent: str
    task_completed: bool
    pasos: int  # contador anti "Supervisor Infinito"

    # Contribuciones de cada especialista, por separado y tipadas
    investigacion: Optional[ResultadoInvestigacion]
    analisis: Optional[ResultadoAnalisis]


# Techo de seguridad: si el Supervisor da más vueltas que esto sin llegar
# a una conclusión, se corta el ciclo y se responde con lo que se tenga
# hasta ese momento, en vez de seguir dando vueltas indefinidamente.
MAX_PASOS = 6
