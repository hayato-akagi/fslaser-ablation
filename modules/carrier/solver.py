"""modules/carrier/solver — キャリア密度の時間発展計算。

layer1/2/3 を統合し、キャリア密度方程式を解く。
物性計算は material_properties モジュールに委譲する。
"""

import numpy as np
from numpy.typing import NDArray

from modules.carrier.config import CarrierConfig
from modules.carrier.public import CarrierResult
from modules import material_properties


def advance_carrier_density_impl(
    ne: NDArray[np.float64],
    intensity: NDArray[np.float64],
    Te: NDArray[np.float64],
    Tl: NDArray[np.float64],
    phase_state: NDArray[np.int32],
    dt: float,
    config: CarrierConfig,
) -> CarrierResult:
    """1タイムステップ分のキャリア密度を更新する。
    
    シーケンス:
    1. 物性パラメータを取得（material_properties経由）
    2. RHS（5項の合計）を計算
    3. 前進オイラー法で n_e を更新
    4. CarrierResult を構築して返す
    
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
    """
    # === 1. 物性パラメータを取得 ===
    Eg = material_properties.compute_bandgap(Tl, ne, phase_state)
    alpha_spa = material_properties.compute_alpha_spa(Tl, phase_state)
    theta = material_properties.compute_impact_ionization_rate(Te, Eg, config.k_B_eV)
    D0 = material_properties.compute_diffusion_coefficient(Tl, config.T_room)
    
    # === 2. RHS（5項の合計）を計算 ===
    rhs = _compute_rhs(ne, intensity, Te, Tl, phase_state, alpha_spa, theta, D0, config)
    
    # === 3. 前進オイラー法で n_e を更新 ===
    ne_new = _update_carrier_density(ne, rhs, dt)
    
    # === 4. CarrierResult を構築 ===
    result = _build_result(ne_new, rhs)
    
    return result


def _compute_rhs(
    ne: NDArray[np.float64],
    intensity: NDArray[np.float64],
    Te: NDArray[np.float64],
    Tl: NDArray[np.float64],
    phase_state: NDArray[np.int32],
    alpha_spa: NDArray[np.float64],
    theta: NDArray[np.float64],
    D0: NDArray[np.float64],
    config: CarrierConfig,
) -> NDArray[np.float64]:
    """RHS（5項の合計）を計算する。
    
    ∂n_e/∂t = SPA + TPA + Auger + Impact + Diffusion
    
    Args:
        ne: キャリア密度 [cm⁻³], shape: (n_z,)
        intensity: レーザー強度 [W/cm²], shape: (n_z,)
        Te: 電子温度 [K], shape: (n_z,)
        Tl: 格子温度 [K], shape: (n_z,)
        phase_state: 相状態, shape: (n_z,)
        alpha_spa: SPA吸収係数 [cm⁻¹], shape: (n_z,)
        theta: 衝突電離係数 [s⁻¹], shape: (n_z,)
        D0: 拡散係数 [cm²/s], shape: (n_z,)
        config: CarrierConfig
    
    Returns:
        RHS [cm⁻³/s], shape: (n_z,)
    """
    spa = _compute_spa_term(intensity, alpha_spa, config)
    tpa = _compute_tpa_term(intensity, config)
    auger = _compute_auger_term(ne, config)
    impact = _compute_impact_term(ne, theta)
    diffusion = _compute_diffusion_term(ne, D0, config.dz)
    
    rhs = spa + tpa + auger + impact + diffusion
    return rhs


def _compute_spa_term(
    intensity: NDArray[np.float64],
    alpha_spa: NDArray[np.float64],
    config: CarrierConfig,
) -> NDArray[np.float64]:
    """SPA項を計算する。
    
    SPA = α_SPA × I / hω
    
    Args:
        intensity: レーザー強度 [W/cm²], shape: (n_z,)
        alpha_spa: SPA吸収係数 [cm⁻¹], shape: (n_z,)
        config: CarrierConfig
    
    Returns:
        SPA項 [cm⁻³/s], shape: (n_z,)
    """
    hw_J = config.photon_energy_J
    spa_term = alpha_spa * intensity / hw_J
    return spa_term


def _compute_tpa_term(
    intensity: NDArray[np.float64],
    config: CarrierConfig,
) -> NDArray[np.float64]:
    """TPA項を計算する。
    
    TPA = β × I² / (2hω)
    
    Args:
        intensity: レーザー強度 [W/cm²], shape: (n_z,)
        config: CarrierConfig
    
    Returns:
        TPA項 [cm⁻³/s], shape: (n_z,)
    """
    beta_cgs = config.beta_cgs
    hw_J = config.photon_energy_J
    tpa_term = beta_cgs * intensity**2 / (2.0 * hw_J)
    return tpa_term


def _compute_auger_term(
    ne: NDArray[np.float64],
    config: CarrierConfig,
) -> NDArray[np.float64]:
    """Auger項を計算する。
    
    Auger = -γ × n_e³
    
    Args:
        ne: キャリア密度 [cm⁻³], shape: (n_z,)
        config: CarrierConfig
    
    Returns:
        Auger項 [cm⁻³/s], shape: (n_z,)
    """
    auger_term = -config.gamma * ne**3
    return auger_term


def _compute_impact_term(
    ne: NDArray[np.float64],
    theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Impact項を計算する。
    
    Impact = θ × n_e
    
    Args:
        ne: キャリア密度 [cm⁻³], shape: (n_z,)
        theta: 衝突電離係数 [s⁻¹], shape: (n_z,)
    
    Returns:
        Impact項 [cm⁻³/s], shape: (n_z,)
    """
    impact_term = theta * ne
    return impact_term


def _compute_diffusion_term(
    ne: NDArray[np.float64],
    D0: NDArray[np.float64],
    dz: float,
) -> NDArray[np.float64]:
    """拡散項を計算する（FDM中心差分）。
    
    ∇(D_0 × ∇n_e)|_i = [D_{i+1/2}(n_{i+1} - n_i) - D_{i-1/2}(n_i - n_{i-1})] / dz²
    
    境界条件: 断熱 Neumann（dn_e/dz = 0）
    
    Args:
        ne: キャリア密度 [cm⁻³], shape: (n_z,)
        D0: 拡散係数 [cm²/s], shape: (n_z,)
        dz: グリッド間隔 [cm]
    
    Returns:
        拡散項 [cm⁻³/s], shape: (n_z,)
    """
    n_z = len(ne)
    diffusion_term = np.zeros_like(ne)
    
    for i in range(n_z):
        diffusion_term[i] = _compute_diffusion_at_point(ne, D0, i, dz)
    
    return diffusion_term


def _compute_diffusion_at_point(
    ne: NDArray[np.float64],
    D0: NDArray[np.float64],
    i: int,
    dz: float,
) -> float:
    """i 番目のグリッド点で拡散項を計算する。
    
    Args:
        ne: キャリア密度 [cm⁻³], shape: (n_z,)
        D0: 拡散係数 [cm²/s], shape: (n_z,)
        i: グリッド点インデックス
        dz: グリッド間隔 [cm]
    
    Returns:
        拡散項 [cm⁻³/s]
    """
    n_z = len(ne)
    
    # ゴースト点処理（断熱境界条件）
    ne_i = ne[i]
    ne_im1 = ne[i - 1] if i > 0 else ne[0]
    ne_ip1 = ne[i + 1] if i < n_z - 1 else ne[n_z - 1]
    
    D0_i = D0[i]
    D0_im1 = D0[i - 1] if i > 0 else D0[0]
    D0_ip1 = D0[i + 1] if i < n_z - 1 else D0[n_z - 1]
    
    # 界面での拡散係数（線形補間）
    D0_i_plus_half = (D0_i + D0_ip1) / 2.0
    D0_i_minus_half = (D0_im1 + D0_i) / 2.0
    
    # フラックス計算
    flux_plus = D0_i_plus_half * (ne_ip1 - ne_i)
    flux_minus = D0_i_minus_half * (ne_i - ne_im1)
    
    # 発散
    divergence = (flux_plus - flux_minus) / (dz * dz)
    
    return divergence


def _update_carrier_density(
    ne: NDArray[np.float64],
    rhs: NDArray[np.float64],
    dt: float,
) -> NDArray[np.float64]:
    """前進オイラー法で n_e を更新する。
    
    n_e^{n+1} = n_e^n + dt × RHS
    
    Args:
        ne: 現在のキャリア密度 [cm⁻³], shape: (n_z,)
        rhs: RHS [cm⁻³/s], shape: (n_z,)
        dt: 時間刻み [s]
    
    Returns:
        更新後のキャリア密度 [cm⁻³], shape: (n_z,)
    """
    ne_new = ne + dt * rhs
    
    # 非負制約
    ne_new = np.maximum(ne_new, 0.0)
    
    return ne_new


def _build_result(
    ne_new: NDArray[np.float64],
    rhs: NDArray[np.float64],
) -> CarrierResult:
    """CarrierResult を構築する。
    
    Args:
        ne_new: 更新後のキャリア密度 [cm⁻³], shape: (n_z,)
        rhs: RHS（時間変化率） [cm⁻³/s], shape: (n_z,)
    
    Returns:
        CarrierResult
    """
    result = CarrierResult(ne=ne_new, dne_dt=rhs)
    return result
