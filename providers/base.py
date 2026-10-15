from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator
from client.response import StreamEvent


class BaseProvider(ABC):
    name: str

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        stream: bool = True,
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream standard StreamEvent objects across all providers."""
        pass

    @abstractmethod
    async def close(self) -> None:
        pass