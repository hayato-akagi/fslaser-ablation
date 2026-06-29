"""modules/carrier — Carrier density evolution model.

フェムト秒レーザー照射時のシリコン伝導帯における
自由キャリア密度 n_e(z,t) の時間発展を計算する。

Public API:
    - advance_carrier_density: 1タイムステップ分のキャリア密度更新
    - CarrierResult: 更新結果の型
    - CarrierConfig: パラメータ設定
"""

from modules.carrier.config import CarrierConfig
from modules.carrier.public import CarrierResult, advance_carrier_density

__all__ = [
    "advance_carrier_density",
    "CarrierResult",
    "CarrierConfig",
]
