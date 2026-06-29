"""modules/ablation/public.py — 外部公開API"""

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from modules.ablation.config import AblationConfig


class AblationResult(BaseModel):
    """アブレーション判定の結果。
    
    Attributes:
        ablation_depth: アブレーション深さ [cm]（内部単位系）
        ablated_mask: shape (n_z,), True = アブレーション済みグリッド
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    ablation_depth: float
    ablated_mask: NDArray[np.bool_]


def evaluate_ablation(
    Tl: NDArray[np.float64],
    dz: float,
    config: AblationConfig,
) -> AblationResult:
    """Phase Explosion に基づくアブレーション深さを算出。
    
    呼び出しタイミング: euler_fdm の Step 4（ttm の後）
    
    Args:
        Tl: 格子温度 [K], shape (n_z,). z[0]=表面, z[N-1]=底面
        dz: グリッド間隔 [cm]
        config: アブレーション判定パラメータ
    
    Returns:
        AblationResult: アブレーション深さとマスク
    
    Notes:
        - ステートレス: 前ステップの結果に依存しない
        - 連続性ルール: 表面 z[0] から連続して閾値を超える部分のみ
        - 入力配列 Tl は変更しない
    """
    # layer1/2/3 を廃止し、solver.py を使用
    from modules.ablation.solver import evaluate_ablation_sequence
    
    return evaluate_ablation_sequence(Tl, dz, config)
