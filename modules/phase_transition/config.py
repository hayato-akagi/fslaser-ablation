"""modules/phase_transition/config — 相転移パラメータ。

相転移に必要な熱力学パラメータを定義する。
modules/material_properties/constants.py からデフォルト値を参照する。
"""

from pydantic import BaseModel
from modules.material_properties.constants import SILICON


class PhaseTransitionConfig(BaseModel):
    """相転移用のパラメータ。
    
    Attributes:
        T_m: 融点 [K]
        T_b: 沸点 [K]
        L_m: 融解潜熱 [J/cm³]
        L_v: 蒸発潜熱 [J/cm³]
        T_room: 室温 [K]
    """
    
    T_m: float = SILICON.T_m
    T_b: float = SILICON.T_b
    L_m: float = SILICON.L_m
    L_v: float = SILICON.L_v
    T_room: float = 300.0
