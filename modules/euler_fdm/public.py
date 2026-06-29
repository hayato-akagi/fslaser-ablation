"""modules/euler_fdm/public.py — 外部API窓口と型定義

euler_fdm ドメインの唯一の外部インターフェース。
他のモジュールやビューレイヤーは必ずこのファイル経由でアクセスする。
"""

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict


class SimulationResult(BaseModel):
    """シミュレーション全体の結果を格納する型安全なコンテナ。
    
    Attributes:
        # 最終状態
        Te_final: 最終電子温度 [K], shape (n_z,)
        Tl_final: 最終格子温度 [K], shape (n_z,)
        ne_final: 最終キャリア密度 [cm⁻³], shape (n_z,)
        
        # アブレーション結果
        ablation_depth_cm: 最終アブレーション深さ [cm]（内部単位系）
        ablation_depth_nm: 最終アブレーション深さ [nm]（SI変換済み）
        ablated_mask: 累積アブレーションマスク, shape (n_z,)
        
        # 時間履歴（スナップショット）
        time_points: 記録時刻 [s], shape (n_snapshots,)
        Te_surface_history: z=0 の Te 履歴 [K], shape (n_snapshots,)
        Tl_surface_history: z=0 の Tl 履歴 [K], shape (n_snapshots,)
        ne_surface_history: z=0 の ne 履歴 [cm⁻³], shape (n_snapshots,)
        reflectivity_history: 表面反射率履歴, shape (n_snapshots,)
        alpha_fca_surface_history: z=0 の α_FCA 履歴 [cm⁻¹], shape (n_snapshots,)
        auger_term_surface_history: z=0 の γne³ 履歴 [cm⁻³/s], shape (n_snapshots,)
        ablation_depth_history: 各ステップの深さ [nm], shape (n_snapshots,)
        
        # メタデータ
        total_steps: 総ステップ数
        fluence: 入力フルエンス [J/cm²]
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # 最終状態
    Te_final: NDArray[np.float64]
    Tl_final: NDArray[np.float64]
    ne_final: NDArray[np.float64]
    
    # アブレーション結果
    ablation_depth_cm: float
    ablation_depth_nm: float
    ablated_mask: NDArray[np.bool_]
    
    # 時間履歴
    time_points: NDArray[np.float64]
    Te_surface_history: NDArray[np.float64]
    Tl_surface_history: NDArray[np.float64]
    ne_surface_history: NDArray[np.float64]
    reflectivity_history: NDArray[np.float64]
    alpha_fca_surface_history: NDArray[np.float64]
    auger_term_surface_history: NDArray[np.float64]
    ablation_depth_history: NDArray[np.float64]
    
    # メタデータ
    total_steps: int
    fluence: float


def run_simulation(config: "EulerFDMConfig") -> SimulationResult:  # type: ignore
    """シミュレーション全体を実行する（外部API）。
    
    このモジュールの唯一の公開関数。4つのドメインモジュール（carrier, optics, ttm, ablation）
    を連成し、オイラー法・有限差分法によるフェムト秒レーザーアブレーションシミュレーションを実行する。
    
    Args:
        config: EulerFDMConfig インスタンス（グリッド、時間、初期条件等を含む）
    
    Returns:
        SimulationResult: 最終状態、アブレーション結果、時間履歴、メタデータ
    
    Notes:
        - このモジュールは物理計算を行わない（各ドメインモジュールに移譲）
        - 状態ベクトルの一元管理と連成ロジックのみを担当
        - CFL安定性条件に基づく適応的時間刻みを使用
    """
    from modules.euler_fdm.solver import run_simulation_impl
    
    return run_simulation_impl(config)
