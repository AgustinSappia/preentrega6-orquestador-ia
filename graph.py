"""
graph.py
--------
El grafo principal del orquestador: un nodo Supervisor que decide
dinámicamente si manda la tarea al Investigador, al Analista, o si ya
puede finalizar -- la topología jerárquica completa de la Pre-entrega 6.
"""

import os
from typing import Awaitable, Callable, Literal, Optional

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from agents.analyst_agent import analizar
from agents.research_agent import investigar
from state import MAX_PASOS, AgentState

load_dotenv(override=True)

RECURSION_LIMIT = 50  # red de seguridad adicional, por detrás del contador `pasos`


# ---------------------------------------------------------------------------
# Decisión del Supervisor
# ---------------------------------------------------------------------------
class DecisionSupervisor(BaseModel):
    """Salida estructurada del Supervisor -- el `Literal` es justamente
    lo que la consigna pide para las aristas condicionales: el LLM no
    puede devolver cualquier string, solo uno de estos tres valores."""

    next: Literal["Investigador", "Analista", "FINALIZAR"] = Field(
        description="El siguiente agente que debe actuar, o FINALIZAR si ya se puede responder."
    )


PROMPT_SUPERVISOR = (
    "Eres el supervisor de un sistema de investigación y análisis de "
    "noticias de Inteligencia Artificial. Coordinás a dos especialistas: "
    "el Investigador (busca noticias sobre un tema puntual) y el Analista "
    "(evalúa el sentimiento de las noticias que ya se encontraron). Mirá "
    "el estado actual del proceso y decidí quién debe actuar a "
    "continuación, o si ya se puede FINALIZAR.\n\n"
    "Reglas para decidir:\n"
    "- Si todavía no hay ninguna investigación hecha, mandá al "
    "Investigador.\n"
    "- Si ya hay una investigación pero todavía no hay un análisis sobre "
    "ella, mandá al Analista.\n"
    "- Si el análisis dice que fue insuficiente (no había noticias para "
    "analizar), mandá de nuevo al Investigador para que intente con un "
    "término de búsqueda más amplio -- esto es un refinamiento, no un "
    "error.\n"
    "- Si ya hay un análisis marcado como suficiente, FINALIZAR: la tarea "
    "está completa.\n"
)


def _resumen_estado(state: AgentState) -> str:
    """Arma un resumen COMPACTO del estado para pasarle al Supervisor --
    no el historial completo de mensajes. El Supervisor necesita saber
    en qué etapa está el proceso, no releer toda la conversación entre
    los especialistas: esto es la aplicación concreta de evitar la
    "Contaminación de Contexto" que señala la consigna como error
    común."""
    lineas = [f"Tema solicitado: {state['topic']}", f"Paso número: {state['pasos']}"]

    investigacion = state.get("investigacion")
    if investigacion is None:
        lineas.append("Investigación: todavía no se hizo.")
    else:
        lineas.append(
            f"Investigación: se buscó '{investigacion.tema_buscado}', "
            f"se encontraron {len(investigacion.noticias)} noticia(s)."
        )

    analisis = state.get("analisis")
    if analisis is None:
        lineas.append("Análisis: todavía no se hizo.")
    else:
        lineas.append(
            f"Análisis: sentimiento_general='{analisis.sentimiento_general}', "
            f"score_promedio={analisis.score_promedio}, suficiente={analisis.suficiente}."
        )

    return "\n".join(lineas)


DecisorType = Callable[[AgentState], Awaitable[str]]


def _construir_decisor_default(modelo) -> DecisorType:
    """Decisor de producción: usa salida estructurada del LLM."""
    llm_ruteo = modelo.with_structured_output(DecisionSupervisor)

    async def decidir(state: AgentState) -> str:
        mensajes = [
            SystemMessage(content=PROMPT_SUPERVISOR),
            HumanMessage(content=_resumen_estado(state)),
        ]
        decision = await llm_ruteo.ainvoke(mensajes)
        return decision.next

    return decidir


def _construir_modelo():
    provider = os.getenv("LLM_PROVIDER", "openai")

    if provider == "openai":
        modelo = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=modelo, temperature=0)

    elif provider == "anthropic":
        modelo = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        return ChatAnthropic(model=modelo, temperature=0)

    raise ValueError(f"Proveedor no soportado: {provider}")


# ---------------------------------------------------------------------------
# Construcción del grafo
# ---------------------------------------------------------------------------
def construir_grafo(
    modelo=None,
    decidir_siguiente_paso: Optional[DecisorType] = None,
    max_pasos: int = MAX_PASOS,
) -> CompiledStateGraph:
    """Arma y compila el StateGraph del orquestador.

    `modelo` (el LLM que usan los especialistas) y `decidir_siguiente_paso`
    (la función de decisión del Supervisor) se inyectan por separado --
    mismo patrón de inyección de dependencias que las pre-entregas
    anteriores, llevado un paso más allá: en los tests no hace falta
    simular un LLM real que sepa devolver salida estructurada, alcanza
    con inyectar una función de decisión ya guionada.
    """
    llm = modelo or _construir_modelo()
    decidir = decidir_siguiente_paso or _construir_decisor_default(llm)

    async def supervisor_node(state: AgentState) -> dict:
        pasos = state.get("pasos", 0) + 1

        # Techo de seguridad contra el "Supervisor Infinito": si ya
        # dimos demasiadas vueltas, cerramos la conversación con lo que
        # tengamos en vez de seguir pidiéndole al LLM que decida.
        if pasos > max_pasos:
            return {"next_agent": "FINALIZAR", "task_completed": True, "pasos": pasos}

        siguiente = await decidir(state)
        return {
            "next_agent": siguiente,
            "task_completed": siguiente == "FINALIZAR",
            "pasos": pasos,
        }

    async def investigador_node(state: AgentState) -> dict:
        analisis_previo = state.get("analisis")

        # Si ya hubo un intento de análisis y resultó insuficiente (no
        # había noticias), esta es una ronda de REFINAMIENTO: en vez de
        # repetir el mismo término exacto (que ya sabemos que no
        # encuentra nada), ampliamos la búsqueda a un término más
        # genérico. Esto es lo que demuestra, en código, la decisión de
        # "refinar antes de responder" que pide la consigna.
        if analisis_previo is not None and not analisis_previo.suficiente:
            termino = "ia"
        else:
            termino = state["topic"]

        resultado = await investigar(llm, termino)

        # Importante: cada vez que el Investigador produce una
        # investigación NUEVA, cualquier análisis anterior queda
        # obsoleto -- pertenecía a la investigación vieja, no a esta.
        # Si no lo limpiáramos acá, el Supervisor vería el análisis
        # insuficiente de la ronda anterior y mandaría de nuevo al
        # Investigador en bucle, en vez de pasar al Analista con los
        # resultados nuevos que acabamos de conseguir.
        return {"investigacion": resultado, "analisis": None}

    async def analista_node(state: AgentState) -> dict:
        resultado = await analizar(llm, state["investigacion"])
        return {"analisis": resultado}

    grafo = StateGraph(AgentState)
    grafo.add_node("Supervisor", supervisor_node)
    grafo.add_node("Investigador", investigador_node)
    grafo.add_node("Analista", analista_node)

    grafo.set_entry_point("Supervisor")

    # La única arista condicional: sale del Supervisor y rutea según
    # `next_agent`, que viene acotado por el Literal de DecisionSupervisor
    # (más el corte forzado del contador de pasos).
    grafo.add_conditional_edges(
        "Supervisor",
        lambda state: state["next_agent"],
        {"Investigador": "Investigador", "Analista": "Analista", "FINALIZAR": END},
    )

    # Los especialistas SIEMPRE vuelven al Supervisor -- nunca se
    # comunican entre sí ni decen por su cuenta que la tarea terminó.
    grafo.add_edge("Investigador", "Supervisor")
    grafo.add_edge("Analista", "Supervisor")

    return grafo.compile()


async def ejecutar_orquestador(grafo: CompiledStateGraph, topic: str) -> dict:
    """Helper de invocación: arma el estado inicial completo (todos los
    campos presentes desde el arranque, ninguno implícito) y fija un
    `recursion_limit` explícito como red de seguridad adicional."""
    estado_inicial: AgentState = {
        "messages": [],
        "topic": topic,
        "next_agent": "",
        "task_completed": False,
        "pasos": 0,
        "investigacion": None,
        "analisis": None,
    }
    return await grafo.ainvoke(estado_inicial, config={"recursion_limit": RECURSION_LIMIT})
