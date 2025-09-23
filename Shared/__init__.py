try:
    from google.adk.models.lite_llm import LiteLlm
except ImportError as e:
    raise ImportError(
        "LiteLlm requires the 'litellm' package. "
        "Please install it with 'pip install litellm'."
    ) from e
from google.adk.planners import BuiltInPlanner
from google.genai import types

DynamicPlanner = BuiltInPlanner(
    thinking_config=types.ThinkingConfig(thinking_budget=-1)
)
NoSelfReasoningPlanner = BuiltInPlanner(
    thinking_config=types.ThinkingConfig(thinking_budget=0)
)
# https://ai.google.dev/gemini-api/docs/thinking?hl=zh-tw
# MODEL = LiteLlm(model="openai/gpt-4o")
MODEL20 = "gemini-2.0-flash"
# MODEL = "gemini-2.5-pro"
MODEL25 = "gemini-2.5-flash"
# MODEL = "gemini-2.0-flash-lite"
