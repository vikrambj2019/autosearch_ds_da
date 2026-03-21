"""
Registry loader — discovers skill packages and routes questions to skills.
"""

from __future__ import annotations

import json
from pathlib import Path


REGISTRY_DIR = Path(__file__).resolve().parent.parent / "registry"
COMPANY_CONTEXT_DIR = Path(__file__).resolve().parent.parent / "company_context"


def _load_text(path: Path) -> str | None:
    """Read a text file, return None if missing."""
    if path.exists():
        return path.read_text()
    return None


def _load_json(path: Path) -> dict:
    """Read a JSON file, return empty dict if missing."""
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _load_company_context() -> str:
    """Concatenate all company context files into a single string."""
    if not COMPANY_CONTEXT_DIR.exists():
        return ""
    parts: list[str] = []
    for f in sorted(COMPANY_CONTEXT_DIR.iterdir()):
        if f.suffix in (".md", ".txt"):
            parts.append(f"--- {f.name} ---\n{f.read_text()}")
    return "\n\n".join(parts)


def load_skill_manifest(skill_dir: Path) -> dict:
    """Load and return the manifest.json for a skill directory."""
    return _load_json(skill_dir / "manifest.json")


def build_agent_prompt(skill_dir: Path, grain_context: str = "") -> str:
    """Assemble the full agent prompt from skill files.

    Combines:
      1. prompt.md (required — the agent's system prompt)
      2. patterns.md (optional — code patterns appended as reference)
      3. Company context (optional — domain knowledge from company_context/)
      4. {GRAIN_CONTEXT} placeholder replacement
    """
    prompt = _load_text(skill_dir / "prompt.md") or ""
    patterns = _load_text(skill_dir / "patterns.md")
    template = _load_text(skill_dir / "template.py")
    company_ctx = _load_company_context()

    # Inject grain context
    prompt = prompt.replace("{GRAIN_CONTEXT}", grain_context)

    # Append patterns
    if patterns:
        prompt += f"\n\n## Reference Patterns\n\n{patterns}"

    # Append template code
    if template:
        prompt += f"\n\n## Pipeline Template Code\n\n```python\n{template}\n```"

    # Append company context
    if company_ctx:
        prompt += f"\n\n## Company Context\n\n{company_ctx}"

    return prompt


META_SKILLS = {"critic", "reporter", "evaluator"}


def discover_skills() -> dict[str, dict]:
    """Scan registry/ and return {skill_name: manifest} for all valid skills."""
    skills: dict[str, dict] = {}
    if not REGISTRY_DIR.exists():
        return skills
    for skill_dir in sorted(REGISTRY_DIR.iterdir()):
        if not skill_dir.is_dir():
            continue
        manifest = load_skill_manifest(skill_dir)
        if manifest and "name" in manifest:
            skills[manifest["name"]] = manifest
    return skills


def discover_analytics_skills() -> dict[str, dict]:
    """Return only user-facing analytics skills (exclude meta-agents)."""
    return {k: v for k, v in discover_skills().items() if k not in META_SKILLS}


def route_question(question: str, skills: dict[str, dict] | None = None) -> str | None:
    """Match a user question to the best skill based on trigger phrases.

    Uses weighted scoring: multi-word triggers score higher (more specific),
    and ties are broken by specificity (fewer total triggers = more focused skill).
    Returns the skill name, or None if no match.
    """
    if skills is None:
        skills = discover_skills()

    q_lower = question.lower()
    best_skill: str | None = None
    best_score = 0.0

    for name, manifest in skills.items():
        triggers = manifest.get("routing", {}).get("trigger_phrases", [])
        if not triggers:
            continue
        # Multi-word triggers score higher (they're more specific)
        score = sum(
            len(t.split()) for t in triggers if t.lower() in q_lower
        )
        # Normalize by total trigger count to favor focused skills
        # (diagnostic has fewer false-positive triggers than descriptive)
        if score > 0:
            specificity_bonus = score / len(triggers)
            score += specificity_bonus

        if score > best_score:
            best_score = score
            best_skill = name

    return best_skill
