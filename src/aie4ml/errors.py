"""Errors the compiler raises for a design it cannot realise, as opposed to malformed input or a bug."""


class ConfigRefused(Exception):
    """A configuration the compiler cannot realise: a split, contract or layout no kernel implements, one that
    does not fit a tile's memory, or a design no placement or transport can build. Raised only where the refusal
    depends on the configuration chosen, so a search may try another; malformed IR and broken invariants raise
    ordinary errors."""
