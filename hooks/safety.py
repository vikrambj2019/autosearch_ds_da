"""
Safety hooks — PreToolUse guardrails for Bash and code execution.
"""

from __future__ import annotations

import re
from typing import Any

from claude_agent_sdk.types import (
    HookContext,
    HookJSONOutput,
    HookMatcher,
    PreToolUseHookInput,
)

# Patterns that should never appear in Bash commands
_DANGEROUS_BASH_PATTERNS = [
    r"\brm\s+-rf\b",
    r"\brm\s+-r\b",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r"\b:(){ :|:& };:",  # fork bomb
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bkill\s+-9\b",
    r"\bchmod\s+777\b",
    r"\bcurl\b.*\|\s*bash",
    r"\bwget\b.*\|\s*bash",
    r"\bsudo\b",
]

_DANGEROUS_BASH_RE = re.compile("|".join(_DANGEROUS_BASH_PATTERNS), re.IGNORECASE)


async def bash_safety_hook(
    input_data: PreToolUseHookInput,
    tool_use_id: str | None,
    context: HookContext,
) -> HookJSONOutput:
    """Block dangerous Bash commands."""
    tool_input = input_data.get("tool_input", {})
    command = tool_input.get("command", "") if isinstance(tool_input, dict) else ""

    if _DANGEROUS_BASH_RE.search(command):
        return {
            "continue_": False,
            "stopReason": f"Blocked dangerous command: {command[:100]}",
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": "Command matches a dangerous pattern",
            },
        }

    return {}


def create_safety_hooks() -> dict[str, list[HookMatcher]]:
    """Return hook configuration for safety guardrails.

    Usage:
        options = ClaudeAgentOptions(hooks=create_safety_hooks())
    """
    return {
        "PreToolUse": [
            HookMatcher(
                matcher="Bash",
                hooks=[bash_safety_hook],
            ),
        ],
    }
