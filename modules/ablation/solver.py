"""modules/ablation/solver — アブレーション深さの計算。

Phase Explosion に基づくアブレーション深さを算出する。
layer1/2/3 を統合し、シンプルな実装を提供する。
"""

import numpy as np
from numpy.typing import NDArray

from modules.ablation.config import AblationConfig
from modules.ablation.public import AblationResult


def evaluate_ablation_sequence(
    Tl: NDArray[np.float64],
    dz: float,
    config: AblationConfig,
) -> AblationResult:
    """アブレーション評価。
    
    シーケンス:
    1. 閾値温度を取得
    2. 閾値以上のグリッドをマスク
    3. 表面から連続してマスクが True のカウント
    4. アブレーション深さを計算
    5. 結果を構築
    
    Args:
        Tl: 格子温度 [K], shape (n_z,)
        dz: グリッド間隔 [cm]
        config: アブレーション判定パラメータ
    
    Returns:
        AblationResult: アブレーション深さとマスク
    
    Notes:
        - ステートレス: 前ステップの結果に依存しない
        - 連続性ルール: 表面 z[0] から連続して閾値を超える部分のみ
    """
    # === 1. 閾値温度を取得 ===
    threshold = config.threshold_temperature
    
    # === 2. 閾値以上のグリッドをマスク ===
    threshold_mask = _compute_threshold_mask(Tl, threshold)
    
    # === 3. 表面から連続してマスクが True のカウント ===
    continuous_count = _count_continuous_from_surface(threshold_mask)
    
    # === 4. アブレーション深さを計算 ===
    ablation_depth = continuous_count * dz
    
    # === 5. アブレーションマスクを生成 ===
    ablated_mask = _create_ablated_mask(len(Tl), continuous_count)
    
    # === 6. 結果を構築 ===
    return AblationResult(
        ablation_depth=ablation_depth,
        ablated_mask=ablated_mask,
    )


def _compute_threshold_mask(
    Tl: NDArray[np.float64],
    threshold: float,
) -> NDArray[np.bool_]:
    """閾値以上のグリッドを True とするマスクを生成する。
    
    Args:
        Tl: 格子温度 [K], shape (n_z,)
        threshold: 閾値温度 [K]
    
    Returns:
        マスク配列, shape (n_z,). Tl[i] >= threshold なら True
    """
    return Tl >= threshold


def _count_continuous_from_surface(
    mask: NDArray[np.bool_],
) -> int:
    """表面から連続して True であるグリッド数をカウントする。
    
    Args:
        mask: ブールマスク, shape (n_z,). mask[0]=表面
    
    Returns:
        連続カウント数（0 以上の整数）
    
    Notes:
        表面 mask[0] が False なら 0 を返す。
    """
    n_z = len(mask)
    count = 0
    for i in range(n_z):
        if mask[i]:
            count += 1
        else:
            break
    return count


def _create_ablated_mask(n_z: int, count: int) -> NDArray[np.bool_]:
    """アブレーションマスクを生成する。
    
    表面から count 個のグリッドを True とする。
    
    Args:
        n_z: グリッド総数
        count: 表面からのカウント数
    
    Returns:
        マスク配列, shape (n_z,)
    """
    mask = np.zeros(n_z, dtype=bool)
    if count > 0:
        mask[:count] = True
    return mask
