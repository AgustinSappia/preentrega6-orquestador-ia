"""
tests/fakes.py
---------------
Utilidades compartidas por toda la suite de tests: un modelo de chat
"falso" que no llama a ninguna API real, pero que SÍ es una instancia
genuina de `BaseChatModel` (requisito de `create_react_agent`, que
rechaza objetos duck-typed que no sean un `Runnable` real -- ver nota
más abajo).
"""
from typing import Any

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel


class ModeloFalso(FakeMessagesListChatModel):
    """Subclase de un `BaseChatModel` real que devuelve, en orden, los
    mensajes de la lista `responses` que se le pasó al construirlo.

    Hace falta subclasearlo para sobreescribir `bind_tools`: la
    implementación por defecto de `BaseChatModel.bind_tools` levanta
    `NotImplementedError`, y `create_react_agent` siempre llama a
    `bind_tools` sobre el modelo que recibe. Como en estos tests nunca
    nos importa qué herramientas quedaron "bindeadas" (el modelo falso
    ya tiene sus respuestas guionadas de antemano), alcanza con
    devolver `self` sin hacer nada más.
    """

    def bind_tools(self, tools, **kwargs: Any):
        return self
