from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()
from typing import AsyncGenerator
from agent.events import AgentEvent, AgentEventType

from client.response import StreamEventType, TokenUsage, ToolCall, ToolResultMessage

import os


from pathlib import Path
import json
from config.config import Config
from agent.session import Session


class Agent:
    def __init__(self, config: Config):
        self.config = config
        self.session: Session | None = Session(self.config)



    async def run(self, message: str):
        yield AgentEvent.agent_start(message)
        #add user message to context
        self.session.context_manager.add_user_message(message)

        final_response: str | None = None

        async for event in self._agentic_loop():
            yield event

            if event.type == AgentEventType.TEXT_COMPLETE:
                final_response = event.data.get("content")

        yield AgentEvent.agent_end(final_response)

    async def _agentic_loop(self) -> AsyncGenerator[AgentEvent, None]:
        # print("DEBUG STEP 1: Entering _agentic_loop")
        max_turns = self.config.max_turns

        for turn_num in range(max_turns):
            await self.session.increment_turn()
        # prompt = "Hey whats going on"

            response_text = ""

            tool_schemas = self.session.tool_registry.get_schemas()

            # print(f"DEBUG STEP 2: Got tool schemas -> Count: {len(tool_schemas) if tool_schemas else 0}")

            tool_calls: list[ToolCall] = []

            # print("DEBUG STEP 4: About to call client.chat_completion...")


            async for event in self.session.client.chat_completion(self.session.context_manager.get_messages(), tools=tool_schemas if tool_schemas else None, stream=True):
                # print("DEBUG STEP 5: Generator object created, starting 'async for event in completion_gen'...")

                if event.type == StreamEventType.TEXT_DELTA:
                    if event.text_delta:
                        content = event.text_delta.content
                        response_text += content
                        yield AgentEvent.text_delta(content)

                elif event.type == StreamEventType.TOOL_CALL_COMPLETE:
                    if event.tool_call:
                        tool_calls.append(event.tool_call)


                elif event.type == StreamEventType.ERROR:
                    yield AgentEvent.agent_error(
                        event.error or "Unknown error occurred.",
                    ) 

            self.session.context_manager.add_assistant_message(
                response_text or None,
                (
                    [
                        {
                            "id": tc.call_id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else str(tc.arguments),
                            },
                        }
                        for tc in tool_calls
                    ]
                    if tool_calls
                    else None
                ),
            )
            
            if response_text:
                yield AgentEvent.text_complete(response_text)


            if not tool_calls:
                return

            tool_call_results: list[ToolResultMessage] = []

            for tool_call in tool_calls:
                yield AgentEvent.tool_call_start(
                    tool_call.call_id,
                    tool_call.name,
                    tool_call.arguments,
                )

                result = await self.session.tool_registry.invoke(
                    tool_call.name,
                    tool_call.arguments,
                    self.config.cwd,
                )


                yield AgentEvent.tool_call_complete(
                    tool_call.call_id,
                    tool_call.name,
                    result,
                )

                tool_call_results.append(
                    ToolResultMessage(
                        tool_call_id=tool_call.call_id,
                        content=result.to_model_output(),
                        is_error=not result.success,
                    )
                )


            for tool_result in tool_call_results:
                self.session.context_manager.add_tool_result(
                    tool_result.tool_call_id,
                    tool_result.content,
     
                )





    async def __aenter__(self) -> Agent:
        return self

    async def __aexit__(self, exc_type, exc_value, tb)-> None:
        if self.session and self.session.client:
            await self.session.client.close()
            self.session = None
                
        