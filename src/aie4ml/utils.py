"""Helpers shared across the backend."""


def sanitize_identifier(name: str, prefix: str = 'id') -> str:
    """Return a C/C++ friendly identifier derived from ``name``."""

    if not name:
        return prefix

    filtered = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in str(name))
    filtered = filtered.lstrip('_') or '_'
    if filtered[0].isdigit():
        filtered = f'{prefix}_{filtered}'
    return filtered
