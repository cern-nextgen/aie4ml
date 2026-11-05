from __future__ import annotations


def align_up(value: int, multiple: int) -> int:
    """Return the smallest multiple of ``multiple`` that is >= ``value``."""
    if multiple <= 0:
        return max(0, int(value))
    return ((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple)


def ceildiv(numer: int, denom: int) -> int:
    """Integer ceil division with explicit non-zero denominator validation."""
    denom = int(denom)
    if denom <= 0:
        raise ValueError(f'ceildiv requires denom > 0, got {denom}.')
    numer = int(numer)
    return (numer + denom - 1) // denom


def divisors(n: int) -> list[int]:
    """The divisors of ``n``, ascending."""
    return [d for d in range(1, int(n) + 1) if int(n) % d == 0]


def require_power_of_two(label: str, value: int) -> None:
    if value <= 0 or (value & (value - 1)) != 0:
        raise ValueError(f'{label}: must be a power of two, got {value}.')
