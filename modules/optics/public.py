"""modules/optics/public.py — 外部API窓口と型定義

このモジュールは optics ドメインの唯一の外部インターフェース。
他のモジュールは必ずこのファイルを経由してアクセスする。
"""

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict


class OpticsResult(BaseModel):
    """光学計算の結果を格納する型安全なコンテナ。

    Attributes:
        intensity: I(z) レーザー強度分布 [W/cm²], shape (n_z,)
        source_term: S(z) 熱源項分布 [W/cm³], shape (n_z,)
        reflectivity: R(0,t) 表面反射率, 無次元
        alpha_fca: α_FCA(z) 自由キャリア吸収係数分布 [cm⁻¹], shape (n_z,)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    intensity: NDArray[np.float64]
    source_term: NDArray[np.float64]
    reflectivity: float
    alpha_fca: NDArray[np.float64]


def compute_laser_field(
    ne: NDArray[np.float64],
    Tl: NDArray[np.float64],
    Te: NDArray[np.float64],
    phase_state: NDArray[np.int32],
    t: float,
    config: "OpticsConfig",
) -> OpticsResult:
    """レーザー場の空間分布を計算する（外部API）。

    呼び出しタイミング: euler_fdm の Step 1（各タイムステップの最初）

    Args:
        ne: キャリア密度分布 [cm⁻³], shape (n_z,)
        Tl: 格子温度分布 [K], shape (n_z,)
        Te: 電子温度分布 [K], shape (n_z,)
        phase_state: 相状態分布 (PhaseState), shape (n_z,)
        t: 現在時刻 [s] (t=0 がパルス中心)
        config: 光学計算パラメータ

    Returns:
        OpticsResult: 強度分布、熱源項、反射率、FCA係数
    """
    from modules.optics.solver import compute_laser_field_sequence

    return compute_laser_field_sequence(ne, Tl, Te, phase_state, t, config)
