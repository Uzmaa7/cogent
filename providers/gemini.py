import asyncio
import json
import uuid
from typing import Any, AsyncGenerator
from google import genai
from google.genai import types

from providers.base import BaseProvider
from client.response import (
    StreamEventType,
    StreamEvent,
    TextDelta,
    TokenUsage,
    ToolCall,
    ToolCallDelta,
)


def _to_gemini_contents(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gemini_contents = []
    for m in messages:
        if m.get("role") == "system":
            continue

        role = "model" if m.get("role") == "assistant" else "user"
        content = m.get("content")

        if isinstance(content, str):
            gemini_contents.append({"role": role, "parts": [{"text": content}]})
        elif isinstance(content, list):
            parts = []
            for b in content:
                if b.get("type") == "text":
                    parts.append({"text": b.get("text", "")})
                elif b.get("type") == "tool_use":
                    parts.append({"function_call": {"name": b["name"], "args": b.get("input", {})}})
                elif b.get("type") == "tool_result":
                    # Fix: Gemini functionResponse expects function 'name', not tool_use_id
                    func_name = b.get("name") or b.get("tool_name") or "unknown_tool"
                    parts.append({
                        "function_response": {
                            "name": func_name,
                            "response": {"output": b["content"], "is_error": b.get("is_error", False)},
                        }
                    })
            gemini_contents.append({"role": role, "parts": parts})
    return gemini_contents


class GeminiProvider(BaseProvider):
    name = "gemini"

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash") -> None:
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.max_retries = 3

    async def close(self) -> None:
        pass

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        stream: bool = True,
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        contents = _to_gemini_contents(messages)

        tool_declarations = None
        if tools:
            func_decls = [
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema") or t.get("parameters"),
                }
                for t in tools
            ]
            tool_declarations = [{"function_declarations": func_decls}]

        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=tool_declarations,
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        for attempt in range(self.max_retries + 1):
            try:
                if stream:
                    async for event in self._generate_stream(contents, config):
                        yield event
                else:
                    async for event in self._generate_non_stream(contents, config):
                        yield event
                return
            except Exception as e:
                if attempt < self.max_retries:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                else:
                    yield StreamEvent(type=StreamEventType.ERROR, error=f"Gemini API error: {e}")
                    return

    async def _generate_stream(
        self, contents: list[dict[str, Any]], config: types.GenerateContentConfig
    ) -> AsyncGenerator[StreamEvent, None]:
        input_tokens, output_tokens, cached_tokens = 0, 0, 0

        async for chunk in await self.client.aio.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config,
        ):
            if chunk.usage_metadata:
                input_tokens = chunk.usage_metadata.prompt_token_count or input_tokens
                output_tokens = chunk.usage_metadata.candidates_token_count or output_tokens
                cached_tokens = getattr(chunk.usage_metadata, "cached_content_token_count", 0) or 0

            if chunk.candidates and chunk.candidates[0].content:
                for part in chunk.candidates[0].content.parts:
                    if part.text:
                        yield StreamEvent(type=StreamEventType.TEXT_DELTA, text_delta=TextDelta(part.text))

                    if part.function_call:
                        call_id = f"call_{uuid.uuid4().hex[:8]}"
                        fn_name = part.function_call.name
                        fn_args = part.function_call.args or {}
                        args_str = json.dumps(fn_args)

                        yield StreamEvent(
                            type=StreamEventType.TOOL_CALL_START,
                            tool_call_delta=ToolCallDelta(call_id=call_id, name=fn_name),
                        )
                        yield StreamEvent(
                            type=StreamEventType.TOOL_CALL_DELTA,
                            tool_call_delta=ToolCallDelta(
                                call_id=call_id,
                                name=fn_name,
                                arguments_delta=args_str,
                            ),
                        )
                        yield StreamEvent(
                            type=StreamEventType.TOOL_CALL_COMPLETE,
                            tool_call=ToolCall(call_id=call_id, name=fn_name, arguments=fn_args),
                        )

        yield StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            finish_reason="end_turn",
            usage=TokenUsage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cached_tokens=cached_tokens,
            ),
        )

    async def _generate_non_stream(
        self, contents: list[dict[str, Any]], config: types.GenerateContentConfig
    ) -> AsyncGenerator[StreamEvent, None]:
        full_text = ""
        last_event: StreamEvent | None = None
        async for event in self._generate_stream(contents, config):
            if event.type == StreamEventType.TEXT_DELTA and event.text_delta:
                full_text += event.text_delta.content
            elif event.type == StreamEventType.MESSAGE_COMPLETE:
                last_event = event

        yield StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            text_delta=TextDelta(full_text) if full_text else None,
            finish_reason=last_event.finish_reason if last_event else "end_turn",
            usage=last_event.usage if last_event else None,
        )