"""Callback functions for debate framework agents."""

import logging
from google.adk.agents.callback_context import CallbackContext
from typing import Callable, List, Optional
from google.genai import types
from google.adk.agents import LoopAgent
from typing import Callable, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def set_max_iterations_callback(
    max_iteration_key: str = "",
) -> Callable[[CallbackContext], Optional[types.Content]]:
    def callback(callback_context: CallbackContext) -> Optional[types.Content]:
        agent = callback_context._invocation_context.agent
        if max_iteration_key:
            value = callback_context.state[max_iteration_key]
            if value and str(value).isdigit():
                if hasattr(agent, "max_iterations"):
                    agent.max_iterations = int(value)
                    print(
                        f"[Callback] Set {agent.name}.max_iterations = {value} from state['{max_iteration_key}']"
                    )
                else:
                    print("[Callback] Agent does not support max_iterations.")
            else:
                print(
                    f"[Callback] Cannot set max_iterations: '{max_iteration_key}' not found or not numeric."
                )
        return None

    return callback


def chain_callbacks(
    callbacks: List[Callable[[CallbackContext], Optional[types.Content]]],
) -> Callable[[CallbackContext], Optional[types.Content]]:
    def chained(callback_context: CallbackContext) -> Optional[types.Content]:
        for cb in callbacks:
            cb(callback_context)
        return None

    return chained


def append_element_to_list_callback(
    list_key: str = "list",
    element_key: str = "element",
) -> Callable[[CallbackContext], Optional[types.Content]]:
    def callback(callback_context: CallbackContext) -> Optional[types.Content]:
        try:
            # Get the target list from the state, or initialize it if not present
            target_list = callback_context.state.get(list_key, [])
            # Get the source item to append
            source_item = callback_context.state.get(element_key)

            # Append the source item if it exists
            if source_item is not None:
                target_list.append(source_item)
                callback_context.state[list_key] = target_list
                print(f"[Callback] Appended to '{list_key}'-> {source_item}")
        except Exception as e:
            print(f"[Callback] Failed to append to '{list_key}': {e}")
        return None

    return callback
