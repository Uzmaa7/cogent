import asyncio
import json
import re
from typing import Any, AsyncGenerator
from groq import AsyncGroq, APIConnectionError, APIError, RateLimitError
from config.config import Config
from providers.base import BaseProvider
from client.response import (
    StreamEventType,
    StreamEvent,
    TextDelta,
    TokenUsage,
    ToolCall,
    ToolCallDelta,
)



def _to_groq_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groq_msgs = []
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            continue

        # Handled standard OpenAI format:
        if role == "assistant":
            msg_obj: dict[str, Any] = {
                "role": "assistant",
                "content": msg.get("content") or None,
            }
            if msg.get("tool_calls"):
                msg_obj["tool_calls"] = msg.get("tool_calls")
            groq_msgs.append(msg_obj)

        elif role == "tool":
            groq_msgs.append({
                "role": "tool",
                "tool_call_id": msg.get("tool_call_id"),
                "content": msg.get("content", ""),
            })

        else:
            groq_msgs.append({
                "role": role,
                "content": msg.get("content", "")
            })

    return groq_msgs


class GroqProvider(BaseProvider):
    name = "groq"

    def __init__(self, api_key: str, config: Config, model: str = "openai/gpt-oss-120b",) -> None:

        # print(f"DEBUG [GroqProvider.__init__]: Received api_key arg -> repr: {repr(api_key)}")
        clean_key = api_key.strip().strip('"').strip("'") if api_key else ""
        # print(f"DEBUG [GroqProvider.__init__]: Instantiating AsyncGroq with key length -> {len(clean_key)}, prefix -> {clean_key[:5]}...")

        

        
        self.config = config
        self.client = AsyncGroq(api_key=clean_key)
        self.model = model
        self.max_retries = 3
        

    async def close(self) -> None:
        await self.client.close()

    def _build_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema") or t.get("parameters", {"type": "object", "properties": {}}),
                },
            }
            for t in tools
        ]

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        stream: bool = True,
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:


        # print(f"DEBUG [GroqProvider.complete]: Executing API request using model -> {self.model}")
        # print(f"DEBUG [GroqProvider.complete]: Client API key set -> {self.client.api_key[:5]}...")
        
        formatted_messages = []
        if system:
            formatted_messages.append({"role": "system", "content": system})

        formatted_messages.extend(_to_groq_messages(messages))

        formatted_tools = self._build_tools(tools) if tools else None

        kwargs: dict[str, Any] = {
            # "model": self.model,
            "model": self.config.model_name,
            "messages": formatted_messages,
            "stream": True,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if formatted_tools:
            kwargs["tools"] = formatted_tools

        for attempt in range(self.max_retries + 1):
            try:
                if stream:
                    async for event in self._generate_stream(kwargs):
                        yield event
                else:
                    async for event in self._generate_non_stream(kwargs):
                        yield event
                return
            except RateLimitError as e:
                if attempt < self.max_retries:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                else:
                    yield StreamEvent(type=StreamEventType.ERROR, error=f"Max retries exceeded: {e}")
                    return
            except APIConnectionError as e:
                    print(f"DEBUG [GroqProvider APIError caught]: {e}")
                    if attempt < self.max_retries:
                        wait_time = 2**attempt
                        await asyncio.sleep(wait_time)
                    else:
                        yield StreamEvent(type=StreamEventType.ERROR, error=f"API Connection Error: {e}")
                        return
            except APIError as e:
                yield StreamEvent(type=StreamEventType.ERROR, error=f"API error: {e}")
                return

    async def _generate_stream(self, kwargs: dict[str, Any]) -> AsyncGenerator[StreamEvent, None]:
        tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = "end_turn"
        prompt_tokens, completion_tokens = 0, 0

        try:
            # print("DEBUG [_generate_stream]: About to await client.chat.completions.create()...")
            res_stream = await self.client.chat.completions.create(**kwargs)
            # print("DEBUG [_generate_stream]: Successfully created res_stream! Starting streaming loop...")

            async for chunk in res_stream:
                # print(f"DEBUG [_generate_stream]: Received chunk -> {chunk}")
                
                if hasattr(chunk, "x_groq") and chunk.x_groq and hasattr(chunk.x_groq, "usage"):
                    prompt_tokens = getattr(chunk.x_groq.usage, "prompt_tokens", 0) or prompt_tokens
                    completion_tokens = getattr(chunk.x_groq.usage, "completion_tokens", 0) or completion_tokens

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                delta = choice.delta
                if not delta:
                    continue

                if delta.content:
                    yield StreamEvent(type=StreamEventType.TEXT_DELTA, text_delta=TextDelta(delta.content))

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls:
                            call_id = tc.id or f"tc_{idx}"
                            tool_calls[idx] = {"id": call_id, "name": "", "arguments": ""}
                            if tc.function and tc.function.name:
                                tool_calls[idx]["name"] = tc.function.name
                                yield StreamEvent(
                                    type=StreamEventType.TOOL_CALL_START,
                                    tool_call_delta=ToolCallDelta(call_id=call_id, name=tc.function.name),
                                )

                        if tc.function and tc.function.arguments:
                            tool_calls[idx]["arguments"] += tc.function.arguments
                            yield StreamEvent(
                                type=StreamEventType.TOOL_CALL_DELTA,
                                tool_call_delta=ToolCallDelta(
                                    call_id=tool_calls[idx]["id"],
                                    name=tool_calls[idx]["name"],
                                    arguments_delta=tc.function.arguments,
                                ),
                            )

        except APIError as err:
            print(f"DEBUG [_generate_stream EXCEPTION]: APIError caught -> {err}")
            failed_gen = getattr(err, "error", {}).get("failed_generation") if hasattr(err, "error") else None
            if getattr(err, "status_code", None) == 400 and failed_gen:
                regex = r"<function=(\w+)\s+(.+?)</function>"
                matches = re.findall(regex, failed_gen, re.DOTALL)
                for name, args_str in matches:
                    idx = len(tool_calls)
                    call_id = f"tc_{name}_{idx}"
                    tool_calls[idx] = {"id": call_id, "name": name, "arguments": args_str}
                    yield StreamEvent(
                        type=StreamEventType.TOOL_CALL_START,
                        tool_call_delta=ToolCallDelta(call_id=call_id, name=name),
                    )
                    yield StreamEvent(
                        type=StreamEventType.TOOL_CALL_DELTA,
                        tool_call_delta=ToolCallDelta(call_id=call_id, name=name, arguments_delta=args_str),
                    )
                finish_reason = "tool_calls"
            else:
                raise err

        for tc in tool_calls.values():
            try:
                parsed_args = json.loads(tc["arguments"])
            except Exception:
                parsed_args = {"raw_arguments": tc["arguments"]}

            yield StreamEvent(
                type=StreamEventType.TOOL_CALL_COMPLETE,
                tool_call=ToolCall(call_id=tc["id"], name=tc["name"], arguments=parsed_args),
            )

        yield StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            finish_reason="tool_use" if finish_reason == "tool_calls" else "end_turn",
            usage=TokenUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        )

    async def _generate_non_stream(self, kwargs: dict[str, Any]) -> AsyncGenerator[StreamEvent, None]:
        full_text = ""
        last_event: StreamEvent | None = None
        async for event in self._generate_stream(kwargs):
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