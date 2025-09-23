# tools/load_candidate.py

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
from typing import List, Dict, Any
from google.adk.tools import ToolContext

logger = logging.getLogger(__name__)

# Directory containing one combined markdown per candidate
# Corrected path includes the double 'hr_compare_agent' folder
CANDIDATES_DIR = "/home/shaosen/LLM/Multi-agent/Candidates"


def load_two_candidates_tool(
    names: List[str], tool_context: ToolContext
) -> Dict[str, Any]:
    """Loads candidate markdown files and stores results in ToolContext state.

    Args:
      names: A list of names.
      tool_context: ToolContext object for accessing and modifying state.

    Returns:
      A dict with "status" and (optional) "error_message" keys.
    """
    if len(names) != 2:
        return {
            "status": "error",
            "error_message": "Provide exactly two candidate names in 'names'.",
        }

    try:
        # Load each candidate's single markdown file
        for idx, state_key in enumerate(["candidatesA", "candidatesB"]):
            name = names[idx]
            # Expect a file at CANDIDATES_DIR/<name>/<name>.md
            path = os.path.join(CANDIDATES_DIR, name, f"{name}.md")
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            tool_context.state[state_key] = content
            logger.info(f"[Tool] Loaded markdown for {name}: {path}")

        # Save the list of candidate names
        tool_context.state["candidate_names"] = names
        return {"status": "loaded", "candidate_names": names}

    except FileNotFoundError as fnf:
        logger.warning(f"[Tool] File not found: {fnf}")
        return {"status": "error", "error_message": f"File not found: {fnf.filename}"}
    except Exception as e:
        logger.warning(f"[Tool] Unexpected error: {e}")
        return {"status": "error", "error_message": str(e)}


def load_single_candidate_tool(name: str, tool_context: ToolContext) -> Dict[str, Any]:
    """Loads a single candidate markdown file and stores it in ToolContext.

    Args:
      name: Candidate name (should match folder and file name).
      tool_context: ToolContext object for accessing and modifying state.

    Returns:
      A dict with "status" and (optional) "error_message" keys.
    """
    try:
        path = os.path.join(CANDIDATES_DIR, name, f"{name}.md")
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        tool_context.state["review_document"] = content
        tool_context.state["candidate_name"] = name
        logger.info(f"[Tool] Loaded markdown for {name}: {path}")
        return {"status": "loaded", "candidate_name": name}

    except FileNotFoundError as fnf:
        logger.warning(f"[Tool] File not found: {fnf}")
        return {"status": "error", "error_message": f"File not found: {fnf.filename}"}
    except Exception as e:
        logger.warning(f"[Tool] Unexpected error: {e}")
        return {"status": "error", "error_message": str(e)}
