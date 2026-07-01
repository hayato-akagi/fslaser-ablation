"""ttm モジュールの設定パラメータ。

物理定数および相転移パラメータを Pydantic モデルで型安全に管理。
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict
from modules.material_properties.constants import PHYSICAL, SILICON


class TTMConfig(BaseModel):
    """TTM計算のパラメータ。
    
    Attributes:
        T_m: 融点 [K]
        T_b: 沸点 [K]
        T_room: 室温 [K]
        L_m: 融解潜熱 [J/cm³]
        L_v: 気化潜熱 [J/cm³]
        rho: 密度 [g/cm³]
        k_B: ボルツマン定数 [J/K]
        k_B_eV: ボルツマン定数 [eV/K]
        tau_e_base: 固相基底衝突時間 [s]
        tau_e_ne_ref: 衝突時間の ne 参照密度 [cm⁻³]
        tau_e_liquid: 液相での衝突時間 [s]
        dz: グリッド間隔 [cm] (euler_fdm から注入)
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # 相転移温度（SILICON から）
    T_m: float = SILICON.T_m
    T_b: float = SILICON.T_b
    T_room: float = SILICON.T_room
    
    # 潜熱（SILICON から）
    L_m: float = SILICON.L_m
    L_v: float = SILICON.L_v
    
    # 密度（SILICON から）
    rho: float = SILICON.rho
    
    # ボルツマン定数（PHYSICAL から）
    k_B: float = PHYSICAL.k_B
    k_B_eV: float = PHYSICAL.k_B_eV
    
    # 電子衝突時間（SILICON から、optics と同一）
    tau_e_base: float = SILICON.tau_e_base
    tau_e_ne_ref: float = SILICON.tau_e_ne_ref
    tau_e_liquid: float = SILICON.tau_e_liquid
    
    # グリッド間隔（euler_fdm から注入）
    dz: float

    # Te スキーム選択
    # "cn"    : Predictor-Corrector Crank-Nicolson（半陰的、安定性高）
    # "euler" : 前進オイラー（陽的、論文と同一スキーム。CFL条件を dt_max で管理すること）
    te_scheme: Literal["cn", "euler"] = "cn"

    # Predictor-Corrector Crank-Nicolson パラメータ（te_scheme="cn" 時のみ使用）
    cn_max_iter: int = 50
    cn_tol: float = 1e-6

    # キャリアエネルギー項の制御フラグ
    # True : -(Eg + 3kB*Te + ne*∂Eg/∂ne)*dne_dt - ne*∂Eg/∂Tl*dTl_dt を Te 方程式に含める
    # False: キャリアエネルギー項をゼロとする
    include_carrier_energy_term: bool = True
