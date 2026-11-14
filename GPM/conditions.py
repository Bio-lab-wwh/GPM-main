"""Shared helpers for control labels and perturbation technology names."""

from __future__ import annotations

CONTROL_CONDITION_LABEL: str = "ctrl"

CONTROL_CONDITION_TOKENS: frozenset[str] = frozenset(
    {
        "ctrl",
        "control",
        "non-targeting",
        "non_targeting",
        "nt",
        "unperturbed",
        "wildtype",
        "wt",
    }
)


def is_control_condition(condition: str) -> bool:
    """Return whether the full condition label is a control. Empty labels count as controls."""
    token = str(condition).strip().lower()
    if not token:
        return True
    return token in CONTROL_CONDITION_TOKENS


def is_control_token(token: str) -> bool:
    """Return whether a token in a composite condition label denotes a control."""
    return str(token).strip().lower() in CONTROL_CONDITION_TOKENS


def normalize_tech_name(tech: str) -> str:
    """Map known technology aliases to KD, KO, or OE and uppercase other labels."""
    key = str(tech).strip().upper()
    aliases = {
        "CRISPRKO": "KO",
        "CRISPR_KO": "KO",
        "KNOCKOUT": "KO",
        "CRISPRI": "KD",
        "CRISPR_I": "KD",
        "CRISPR_KD": "KD",
        "KNOCKDOWN": "KD",
        "CRISPRA": "OE",
        "CRISPR_A": "OE",
        "OVEREXPRESSION": "OE",
    }
    return aliases.get(key, key)
