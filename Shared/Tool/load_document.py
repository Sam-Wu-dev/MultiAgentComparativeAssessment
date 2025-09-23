# tools/load_document.py

# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
from google.adk.tools import ToolContext

logger = logging.getLogger(__name__)


def load_document_tool(
    state_name: str, filepath: str, tool_context: ToolContext
) -> dict[str, str]:
    """Loads a text or markdown file and stores its content in ToolContext state.

    Args:
      state_name: The key for store the document.
      filepath: The filepath of document.
      tool_context: ToolContext object for accessing and modifying state.

    Returns:
      A dict with "status" and (optional) "error_message" keys.
    """
    if not filepath or not isinstance(filepath, str):
        return {"status": "error", "error_message": "A valid file path is required."}

    try:
        # Load the document content
        with open(filepath, "r", encoding="utf-8") as f:
            document_content = f.read().strip()
        logger.info(f"[Tool] Loaded document from {filepath}")

        # Update state with document content
        tool_context.state.update({state_name: document_content})

        return {"status": "document loaded"}

    except Exception as e:
        logger.warning(f"[Tool] Failed to load document: {e}")
        return {"status": "error", "error_message": str(e)}
