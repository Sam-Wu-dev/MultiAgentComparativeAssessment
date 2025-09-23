import uuid
import asyncio
from typing import Any, Tuple
from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

import asyncio
import uuid
from google.adk.sessions import InMemorySessionService
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.agents.callback_context import CallbackContext
from google.genai import types


def run_agent_with_input_state_sync(
    agent: LlmAgent,
    input_state: dict[str, Any],
    app_name: str = "app",
    user_id: str = "user",
    output_override: str = None,
) -> Tuple[str, dict]:
    session_id = f"session_{uuid.uuid4().hex}"
    print(f"[INFO] Creating session with ID: {session_id}")

    session_service = InMemorySessionService()

    async def setup_and_run():
        agent_to_use = agent
        if output_override:
            agent_to_use = agent.copy(update={"output_key": output_override})

        session = await session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            state=input_state,
        )

        user_content = types.Content(role="user", parts=[types.Part(text="")])

        runner = Runner(
            agent=agent_to_use,
            app_name=app_name,
            session_service=session_service,
        )

        final = "No response"
        async for event in runner.run_async(
            user_id=user_id, session_id=session_id, new_message=user_content
        ):
            if event.is_final_response() and event.content and event.content.parts:
                final = event.content.parts[0].text

        session = await session_service.get_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )
        return final, dict(session.state)

    return asyncio.new_event_loop().run_until_complete(setup_and_run())
