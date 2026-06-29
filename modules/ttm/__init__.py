"""ttm モジュール: 二温度モデル (Two-Temperature Model)。

電子温度・格子温度の時間発展計算、相転移、潜熱管理を担当。

Public API:
    advance_temperatures: 1タイムステップ分の温度更新
    TTMResult: 計算結果を格納する型
    TTMConfig: モジュール設定
"""

from modules.ttm.public import advance_temperatures, TTMResult
from modules.ttm.config import TTMConfig

__all__ = [
    "advance_temperatures",
    "TTMResult",
    "TTMConfig",
]
