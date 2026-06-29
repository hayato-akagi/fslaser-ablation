from enum import IntEnum

class PhaseState(IntEnum):
    SOLID = 0
    MELTING = 1
    LIQUID = 2
    VAPORIZING = 3
    VAPOR = 4


__all__ = ["PhaseState"]