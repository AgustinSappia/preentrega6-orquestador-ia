"""
tests/test_graph.py
---------------------
Tests de integración del grafo completo: construyen el `StateGraph` de
punta a punta (Supervisor + Investigador + Analista) con un LLM falso y
una función de decisión inyectada, sin necesidad de ninguna API real.

Cubre los tres escenarios pedidos por la consigna:
  1. Camino feliz (Investigador -> Analista -> FINALIZAR).
  2. Ciclo de refinamiento (la primera búsqueda no encuentra nada, el
     Investigador amplía el término, recién ahí el Analista encuentra
     datos suficientes).
  3. Corte por contador de pasos ("Supervisor Infinito" simulado con un
     decisor roto que siempre pide más investigación).
"""
from langchain_core.messages import AIMessage

from agents.research_agent import _NOTICIAS_IA
from graph import construir_grafo, ejecutar_orquestador
from tests.fakes import ModeloFalso


# ---------------------------------------------------------------------------
# Funciones de decisión inyectadas (reemplazan al LLM del Supervisor).
# ---------------------------------------------------------------------------
async def decisor_de_referencia(state):
    """Implementación de referencia, en Python puro, de la lógica que
    el Supervisor le pediría a un LLM que siguiera."""
    investigacion = state.get("investigacion")
    analisis = state.get("analisis")
    if investigacion is None:
        return "Investigador"
    if analisis is None:
        return "Analista"
    if not analisis.suficiente:
        return "Investigador"
    return "FINALIZAR"


async def decisor_terco(state):
    """Simula un Supervisor 'roto' que siempre pide más investigación,
    sin importar el estado -- para probar el corte por contador de
    pasos (la protección contra el 'Supervisor Infinito')."""
    return "Investigador"


def _noticias_que_matchean(termino: str):
    termino = termino.lower()
    out = []
    for n in _NOTICIAS_IA:
        texto = " ".join([n["titular"], n["resumen"], *n["etiquetas"]]).lower()
        if termino in texto:
            out.append(
                {
                    "titular": n["titular"],
                    "fuente": n["fuente"],
                    "fecha": n["fecha"],
                    "resumen": n["resumen"],
                }
            )
    return out


# ---------------------------------------------------------------------------
# Escenario 1: camino feliz.
# ---------------------------------------------------------------------------
async def test_camino_feliz():
    noticias = _noticias_que_matchean("modelos")
    modelo = ModeloFalso(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "buscar_noticias_ia", "args": {"tema": "modelos", "maximo": 5}, "id": "c1"}],
            ),
            AIMessage(content="Encontré noticias sobre modelos."),
            AIMessage(
                content="",
                tool_calls=[{"name": "analizar_sentimiento_noticias", "args": {"noticias": noticias}, "id": "c2"}],
            ),
            AIMessage(content="Listo, analicé el sentimiento."),
        ]
    )
    grafo = construir_grafo(modelo=modelo, decidir_siguiente_paso=decisor_de_referencia)
    resultado = await ejecutar_orquestador(grafo, "modelos")

    assert resultado["task_completed"] is True
    assert resultado["pasos"] == 3
    assert resultado["investigacion"].tema_buscado == "modelos"
    assert len(resultado["investigacion"].noticias) == len(noticias)
    assert resultado["analisis"].suficiente is True


# ---------------------------------------------------------------------------
# Escenario 2: ciclo de refinamiento.
# ---------------------------------------------------------------------------
async def test_ciclo_de_refinamiento():
    noticias_amplias = _noticias_que_matchean("ia")
    modelo = ModeloFalso(
        responses=[
            # Primer intento: un tema muy específico que no matchea nada.
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "buscar_noticias_ia", "args": {"tema": "blockchain cuántico", "maximo": 5}, "id": "c1"}
                ],
            ),
            AIMessage(content="No encontré noticias sobre ese tema."),
            # El Analista, al recibir 0 noticias, hace short-circuit y
            # NI SIQUIERA llama al modelo (ver analyst_agent.analizar) --
            # por eso no hay que scriptear respuestas para ese paso.
            # Segundo intento (refinamiento, término ampliado a "ia"):
            AIMessage(
                content="", tool_calls=[{"name": "buscar_noticias_ia", "args": {"tema": "ia", "maximo": 5}, "id": "c3"}]
            ),
            AIMessage(content="Esta vez encontré noticias."),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "analizar_sentimiento_noticias", "args": {"noticias": noticias_amplias}, "id": "c4"}
                ],
            ),
            AIMessage(content="Listo, analicé el sentimiento."),
        ]
    )
    grafo = construir_grafo(modelo=modelo, decidir_siguiente_paso=decisor_de_referencia)
    resultado = await ejecutar_orquestador(grafo, "blockchain cuántico")

    assert resultado["task_completed"] is True
    assert resultado["analisis"].suficiente is True
    # La investigación final tiene que ser la de la búsqueda ampliada,
    # no la específica original (que no encontró nada).
    assert resultado["investigacion"].tema_buscado == "ia"
    assert len(resultado["investigacion"].noticias) == len(noticias_amplias)


# ---------------------------------------------------------------------------
# Escenario 3: corte por contador de pasos.
# ---------------------------------------------------------------------------
async def test_corte_por_contador_de_pasos():
    modelo = ModeloFalso(
        responses=[
            AIMessage(
                content="", tool_calls=[{"name": "buscar_noticias_ia", "args": {"tema": "x", "maximo": 5}, "id": "c1"}]
            ),
            AIMessage(content="No encontré nada."),
        ]
    )
    grafo = construir_grafo(modelo=modelo, decidir_siguiente_paso=decisor_terco, max_pasos=3)
    resultado = await ejecutar_orquestador(grafo, "tema inexistente")

    # Con un "Supervisor roto" que siempre pide más investigación, el
    # grafo tiene que cortar solo por el contador de pasos en vez de
    # colgarse en un ciclo infinito.
    assert resultado["pasos"] == 4  # max_pasos=3 -> se corta al pisar el paso 4
    assert resultado["task_completed"] is True
