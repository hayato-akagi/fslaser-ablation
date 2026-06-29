"""modules/carrier — Public API and result types.

外部モジュールはこのファイルの関数のみを呼び出す。
"""

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict


class CarrierResult(BaseModel):
    """キャリア密度1ステップ更新の結果。
    
    Attributes:
        ne: 更新後のキャリア密度 [cm⁻³], shape: (n_z,)
        dne_dt: 時間変化率 [cm⁻³/s], shape: (n_z,)
                TTMの電子温度方程式で使用される。
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    ne: NDArray[np.float64]
    dne_dt: NDArray[np.float64]


def advance_carrier_density(
    ne: NDArray[np.float64],
    intensity: NDArray[np.float64],
    Te: NDArray[np.float64],
    Tl: NDArray[np.float64],
    phase_state: NDArray[np.int32],
    dt: float,
    config: "CarrierConfig",  # type: ignore
) -> CarrierResult:
    """1タイムステップ分のキャリア密度を更新する。
    
    Args:
        ne: 現在のキャリア密度 [cm⁻³], shape: (n_z,)
        intensity: レーザー強度 I(z) [W/cm²], shape: (n_z,)
        Te: 電子温度 [K], shape: (n_z,)
        Tl: 格子温度 [K], shape: (n_z,)
        phase_state: 相状態 (PhaseState), shape: (n_z,)
        dt: 時間刻み [s]
        config: CarrierConfig インスタンス
    
    Returns:
        CarrierResult: 更新後のキャリア密度と時間変化率
    
    Notes:
        呼び出しタイミング: euler_fdm の Step 2（optics の後、ttm の前）
        
        支配方程式:
        ∂n_e/∂t = (α_SPA * I)/(hω) + (β * I²)/(2hω) 
                  - γ * n_e³ + θ * n_e - ∇(D_0 * ∇n_e)
    """
    # layer1/2/3 を廃止し、solver.py を使用
    from modules.carrier.solver import advance_carrier_density_impl
    
    return advance_carrier_density_impl(
        ne=ne,
        intensity=intensity,
        Te=Te,
        Tl=Tl,
        phase_state=phase_state,
        dt=dt,
        config=config,
    )
