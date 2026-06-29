"""ttm モジュール公開API。

二温度モデル (Two-Temperature Model) による電子・格子温度の時間発展計算。
相転移（固→液→気）における潜熱管理と熱物性パラメータ切替を含む。
"""

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict


class TTMResult(BaseModel):
    """TTM計算結果。
    
    Attributes:
        Te: 更新後電子温度 [K], shape: (n_z,)
        Tl: 更新後格子温度 [K], shape: (n_z,)
        phase_state: 更新後相状態, shape: (n_z,)
        latent_heat_accumulated: 更新後潜熱蓄積量 [J/cm³], shape: (n_z,)
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    Te: NDArray[np.float64]
    Tl: NDArray[np.float64]
    phase_state: NDArray[np.int32]
    latent_heat_accumulated: NDArray[np.float64]


def advance_temperatures(
    Te: NDArray[np.float64],
    Tl: NDArray[np.float64],
    ne: NDArray[np.float64],
    dne_dt: NDArray[np.float64],
    source_term: NDArray[np.float64],
    phase_state: NDArray[np.int32],
    latent_heat_accumulated: NDArray[np.float64],
    dt: float,
    config: "TTMConfig",  # type: ignore
) -> TTMResult:
    """1タイムステップ分の温度更新。
    
    Args:
        Te: 電子温度 [K], shape: (n_z,)
        Tl: 格子温度 [K], shape: (n_z,)
        ne: キャリア密度 [cm⁻³], shape: (n_z,)
        dne_dt: キャリア時間変化率 [cm⁻³/s], shape: (n_z,)
        source_term: 熱源項 S(z) [W/cm³], shape: (n_z,)
        phase_state: 現在の相状態, shape: (n_z,)
        latent_heat_accumulated: 現在の潜熱蓄積量 [J/cm³], shape: (n_z,)
        dt: 時間刻み [s]
        config: TTM設定
    
    Returns:
        TTMResult: 更新後の温度と相状態
    
    Notes:
        - 入力配列は変更せず、新しい配列を返す
        - euler_fdm の Step 3（carrier の後、ablation の前）で呼ばれる
    """
    # layer1/2/3 を廃止し、solver.py を使用
    from modules.ttm.solver import advance_temperatures_impl
    
    return advance_temperatures_impl(
        Te=Te,
        Tl=Tl,
        ne=ne,
        dne_dt=dne_dt,
        source_term=source_term,
        phase_state=phase_state,
        latent_heat_accumulated=latent_heat_accumulated,
        dt=dt,
        config=config,
    )
