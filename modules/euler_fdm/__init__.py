"""modules/euler_fdm — オイラー法・有限差分オーケストレーター

フェムト秒レーザーアブレーションシミュレーションのメインオーケストレーター。
4つのドメインモジュール（carrier, optics, ttm, ablation）を時間方向に連成する。

使用方法:
    from modules.euler_fdm.config import EulerFDMConfig
    from modules.euler_fdm.public import run_simulation
    
    config = EulerFDMConfig(fluence=1.5)
    result = run_simulation(config)
    
    print(f"Ablation depth: {result.ablation_depth_nm:.2f} nm")
"""

from modules.euler_fdm.config import (
    EulerFDMConfig,
    GridConfig,
    InitialCondition,
    TimeConfig,
)
from modules.euler_fdm.public import SimulationResult, run_simulation

__all__ = [
    "run_simulation",
    "SimulationResult",
    "EulerFDMConfig",
    "GridConfig",
    "TimeConfig",
    "InitialCondition",
]
