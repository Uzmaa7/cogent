from typing import Any, AsyncGenerator
from providers.factory import create_provider
from providers.base import BaseProvider
from client.response import StreamEvent


class LLMClient:
    def __init__(self, provider_name: str, api_key: str, model_id: str | None = None) -> None:
        self.provider: BaseProvider = create_provider(
            provider_name=provider_name,
            api_key=api_key,
            model_id=model_id,
        )

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        stream: bool = True,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[StreamEvent, None]:
        async for event in self.provider.complete(
            messages=messages,
            tools=tools,
            system=system,
            stream=stream,
            max_tokens=max_tokens,
        ):
            yield event

    async def close(self) -> None:
        await self.provider.close()