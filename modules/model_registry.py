"""
modules/model_registry.py — Model & Strategy Version Registry (V9 Phase 1)

Every AlphaObservation is tagged with the model/strategy version active
at the moment it was created. If scoring logic changes later, bump the
version here — old observations keep their original tag and are never
retroactively reassigned, so historical evidence is never corrupted by
a later change in how the score itself is computed.
"""

# Bump these whenever scoring/strategy logic actually changes.
# Never edit history to match a new version — old observations keep
# whatever version was active when they were created.
MODEL_VERSION = "APEX-9.0"

STRATEGY_VERSIONS = {
    "swing":     "SWING-2.0",
    "position":  "POSITION-1.0",
    "long_term": "LONGTERM-1.1",  # bumped for the smooth-scoring fix
    "dividend":  "DIVIDEND-1.0",
    "value":     "VALUE-1.0",
}

def get_model_version() -> str:
    return MODEL_VERSION

def get_strategy_version(strategy_name: str) -> str:
    return STRATEGY_VERSIONS.get((strategy_name or "swing").lower(), "UNKNOWN-1.0")
