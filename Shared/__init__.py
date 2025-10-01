try:
    from google.adk.models.lite_llm import LiteLlm
except ImportError as e:
    raise ImportError(
        "LiteLlm requires the 'litellm' package. "
        "Please install it with 'pip install litellm'."
    ) from e

# MODEL = (
#     LiteLlm(
#         model="openai/google/flan-t5-xl",  # "openai/" tells LiteLLM to use OpenAI-format
#         api_base="http://localhost:8080/v1",  # TGI OpenAI-compatible Messages API
#         api_key="none",  # TGI can ignore; keep a placeholder
#     ),
# )
MODEL20 = "gemini-2.0-flash"
