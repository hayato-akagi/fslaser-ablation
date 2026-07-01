"""modules/material_properties/silicon — シリコンの物性パラメータ計算。

全ての物性計算を純粋関数として実装する。
固相・液相で異なる物性は、phase_state を引数に取り、内部で分岐する。
"""

import numpy as np
from numpy.typing import NDArray

from modules import PhaseState
from modules.material_properties.constants import SILICON
from modules.material_properties.drude_plasma import (
    compute_nu_phonon as _compute_nu_phonon,
    compute_nu_ei_spitzer as _compute_nu_ei_spitzer,
    compute_nu_ee as _compute_nu_ee,
)


def compute_bandgap(
    Tl: NDArray[np.float64],
    ne: NDArray[np.float64],
    phase_state: NDArray[np.int32],
) -> NDArray[np.float64]:
    """バンドギャップエネルギー E_g を計算する（論文 Table 1）。

    固相 (SOLID, MELTING):
        E_g = 1.16 - 7.02e-4 × T_l² / (T_l + 1108) - 1.5e-8 × ne^(1/3)

    液相 (LIQUID, VAPORIZING, VAPOR):
        E_g = 0

    Args:
        Tl: 格子温度 [K], shape: (n_z,)
        ne: キャリア密度 [cm⁻³], shape: (n_z,)
        phase_state: 相状態, shape: (n_z,)

    Returns:
        E_g [eV], shape: (n_z,)
    """
    Eg = np.zeros_like(Tl)

    solid_mask = _is_solid_phase(phase_state)

    Eg[solid_mask] = _compute_bandgap_solid(Tl[solid_mask], ne[solid_mask])
    Eg[~solid_mask] = 0.0

    return Eg


def compute_alpha_spa(
    Tl: NDArray[np.float64],
    phase_state: NDArray[np.int32],
) -> NDArray[np.float64]:
    """SPA吸収係数 α_SPA(T_l) を計算する。

    固相 (SOLID, MELTING):
        多項式フィット（固相実測値）
        α_SPA = -58.95 + 0.6226T - 2.3e-3T² + 3.186e-6T³ + 9.967e-10T⁴ - 1.409e-13T⁵

    液相 (LIQUID, VAPORIZING, VAPOR):
        α_SPA = 0。液相Siは金属的でバンドギャップが消失するため
        固相の多項式フィットは適用不可。光吸収は Drude-FCA が担う。

    Args:
        Tl: 格子温度 [K], shape: (n_z,)
        phase_state: 相状態, shape: (n_z,)

    Returns:
        α_SPA [cm⁻¹], shape: (n_z,)
    """
    alpha = np.zeros_like(Tl)

    solid_mask = _is_solid_phase(phase_state)

    T = Tl[solid_mask]
    alpha[solid_mask] = (
        -58.95
        + 0.6226 * T
        - 2.3e-3 * T**2
        + 3.186e-6 * T**3
        + 9.967e-10 * T**4
        - 1.409e-13 * T**5
    )

    return alpha


def compute_impact_ionization_rate(
    Te: NDArray[np.float64],
    Eg: NDArray[np.float64],
    k_B_eV: float = 8.617e-5,
) -> NDArray[np.float64]:
    """衝突電離係数 θ を計算する。

    θ = 3.6e10 × exp(-1.5 × E_g / (k_B × T_e))

    Args:
        Te: 電子温度 [K], shape: (n_z,)
        Eg: バンドギャップエネルギー [eV], shape: (n_z,)
        k_B_eV: ボルツマン定数 [eV/K]

    Returns:
        θ [s⁻¹], shape: (n_z,)

    Notes:
        ゼロ除算回避: T_e < 1e-10 の場合は θ = 0
    """
    safe_Te = np.maximum(Te, 1e-10)
    exponent = -1.5 * Eg / (k_B_eV * safe_Te)
    exponent_clipped = np.clip(exponent, -100.0, 100.0)

    theta = 3.6e10 * np.exp(exponent_clipped)
    theta[Te < 1e-10] = 0.0

    return theta


def compute_diffusion_coefficient(
    Tl: NDArray[np.float64],
    T_room: float = 300.0,
) -> NDArray[np.float64]:
    """両極性拡散係数 D_0 を計算する。
    
    D_0 = 18 × T_room / T_l
    
    Args:
        Tl: 格子温度 [K], shape: (n_z,)
        T_room: 室温 [K]
    
    Returns:
        D_0 [cm²/s], shape: (n_z,)
    
    """
    d0 = 18.0 * T_room / Tl
    return d0


def compute_tau_e(
    ne: NDArray[np.float64],
    Te: NDArray[np.float64],
    Tl: NDArray[np.float64],
    phase_state: NDArray[np.int32],
) -> NDArray[np.float64]:
    """電子衝突時間 τ_e を計算する。

    論文 Fig.6 再現のため、論文 Table 1 の経験式を使用している。
    論文再現性を優先し、高度なDrude-Plasmaモデルは使用していない。

    固相 (SOLID, MELTING):
        τ_e = 240 × (1 + n_e / 6×10²⁰) [fs]

    液相・気相 (LIQUID, VAPORIZING, VAPOR):
        τ_e = 1 ps

    Args:
        ne: キャリア密度 [cm⁻³], shape: (n_z,)
        Te: 電子温度 [K], shape: (n_z,)  （シグネチャ互換のため保持）
        Tl: 格子温度 [K], shape: (n_z,)  （シグネチャ互換のため保持）
        phase_state: 相状態, shape: (n_z,)

    Returns:
        τ_e [s], shape: (n_z,)
    """
    tau_e = np.zeros_like(ne)

    solid_mask = _is_solid_phase(phase_state)
    liquid_mask = ~solid_mask

    tau_e[solid_mask] = 240e-15 * (1.0 + ne[solid_mask] / 6.0e20)
    tau_e[liquid_mask] = 1.0e-12

    return tau_e


def compute_thermal_capacity_electron(
    Te: NDArray[np.float64],
    ne: NDArray[np.float64],
    phase_state: NDArray[np.int32],
    k_B: float = 1.381e-23,
) -> NDArray[np.float64]:
    """電子熱容量 Ce を計算する。
    
    固相: Ce = 3 * n_e * k_B
    液相・気相: Ce = 1e-4 * T_e （論文に基づく）
    
    Args:
        Te: 電子温度 [K], shape: (n_z,)
        ne: キャリア密度 [cm⁻³], shape: (n_z,)
        phase_state: 相状態, shape: (n_z,)
        k_B: ボルツマン定数 [J/K]
    
    Returns:
        Ce [J/(cm³·K)], shape: (n_z,)
    """
    Ce = np.zeros_like(Te)
    
    solid_mask = _is_solid_phase(phase_state)
    liquid_mask = ~solid_mask
    
    Ce[solid_mask] = 3.0 * ne[solid_mask] * k_B
    Ce[liquid_mask] = 1e-4 * Te[liquid_mask]
    
    # ゼロ除算回避（計算上の最小値のみ）
    Ce = np.maximum(Ce, 1e-30)
    return Ce


def compute_thermal_capacity_lattice(
    Tl: NDArray[np.float64],
    phase_state: NDArray[np.int32],
    rho: float = 2.54,
) -> NDArray[np.float64]:
    """格子熱容量 Cl を計算する。
    
    固相: Cl = 1.978 + 3.54e-4 * Tl - 3.68 / Tl² [J/(cm³·K)]
    液相: Cl = 1.06 * ρ [J/(cm³·K)]
    
    Args:
        Tl: 格子温度 [K], shape: (n_z,)
        phase_state: 相状態, shape: (n_z,)
        rho: 密度 [g/cm³]
    
    Returns:
        Cl [J/(cm³·K)], shape: (n_z,)
    """
    Cl = np.zeros_like(Tl)
    
    solid_mask = _is_solid_phase(phase_state)
    liquid_mask = ~solid_mask
    
    Cl[solid_mask] = 1.978 + 3.54e-4 * Tl[solid_mask] - 3.68 / (Tl[solid_mask]**2)
    Cl[liquid_mask] = 1.06 * rho
    
    return Cl


def compute_thermal_conductivity_electron(
    Te: NDArray[np.float64],
    phase_state: NDArray[np.int32],
) -> NDArray[np.float64]:
    """電子熱伝導率 Ke を計算する。
    
    固相: Ke = 1.6e-11 * (-3.47e8 + 4.45e6 * Te)
    液相: Ke = 0.67
    
    Args:
        Te: 電子温度 [K], shape: (n_z,)
        phase_state: 相状態, shape: (n_z,)
    
    Returns:
        Ke [W/(cm·K)], shape: (n_z,)
    """
    Ke = np.zeros_like(Te)
    
    solid_mask = _is_solid_phase(phase_state)
    liquid_mask = ~solid_mask
    
    Ke[solid_mask] = 1.6e-11 * (-3.47e8 + 4.45e6 * Te[solid_mask])
    Ke[liquid_mask] = 0.67
    
    # 非負保証と上限設定（数値安定性）
    Ke = np.clip(Ke, 0.0, 100.0)
    return Ke


def compute_thermal_conductivity_lattice(
    Tl: NDArray[np.float64],
    phase_state: NDArray[np.int32],
    T_m: float,
) -> NDArray[np.float64]:
    """格子熱伝導率 Kl を計算する。
    
    固相: Kl = 1585 * Tl^(-1.23)
    液相: Kl = 0.5 + 2.9e-4 * (Tl - T_m)
    
    Args:
        Tl: 格子温度 [K], shape: (n_z,)
        phase_state: 相状態, shape: (n_z,)
        T_m: 融点 [K]
    
    Returns:
        Kl [W/(cm·K)], shape: (n_z,)
    """
    Kl = np.zeros_like(Tl)
    
    solid_mask = _is_solid_phase(phase_state)
    liquid_mask = ~solid_mask
    
    # 固相: Tl が極小値にならないようクランプ
    Tl_safe = np.maximum(Tl, 10.0)
    Kl[solid_mask] = 1585.0 * (Tl_safe[solid_mask] ** (-1.23))
    Kl[liquid_mask] = 0.5 + 2.9e-4 * (Tl[liquid_mask] - T_m)
    
    # 非負保証と上限設定（数値安定性）
    Kl = np.clip(Kl, 0.0, 1e10)
    return Kl


def compute_tau_el(
    ne: NDArray[np.float64],
    phase_state: NDArray[np.int32],
) -> NDArray[np.float64]:
    """電子-格子エネルギー緩和時間 τ_EL を計算する（Yoffa モデル）。

    固相: τ_EL = τ_0 × [1 + ne / nc]
        高キャリア密度でフォノン結合がスクリーニングされ緩和が遅くなる。
        τ_0 = 240 fs, nc = 6×10²⁰ cm⁻³

    液相: τ_EL = τ_e_liquid = 1 ps（Drude と同値）

    Args:
        ne: キャリア密度 [cm⁻³], shape: (n_z,)
        phase_state: 相状態, shape: (n_z,)

    Returns:
        τ_EL [s], shape: (n_z,)
    """
    TAU_0 = 240e-15
    NC = 6.0e20
    TAU_EL_LIQUID = 1e-12

    tau_el = np.zeros_like(ne)

    solid_mask = _is_solid_phase(phase_state)
    liquid_mask = ~solid_mask

    tau_el[solid_mask] = TAU_0 * (1.0 + ne[solid_mask] / NC)
    tau_el[liquid_mask] = TAU_EL_LIQUID

    return tau_el


def compute_electron_lattice_coupling(
    Ce: NDArray[np.float64],
    ne: NDArray[np.float64],
    phase_state: NDArray[np.int32],
) -> NDArray[np.float64]:
    """電子-格子結合因子 G を計算する。

    G = Ce / τ_EL  （Yoffa モデルによるエネルギー緩和時間を使用）

    Args:
        Ce: 電子熱容量 [J/(cm³·K)], shape: (n_z,)
        ne: キャリア密度 [cm⁻³], shape: (n_z,)
        phase_state: 相状態, shape: (n_z,)

    Returns:
        G [W/(cm³·K)], shape: (n_z,)
    """
    tau_el = compute_tau_el(ne, phase_state)
    G = Ce / tau_el
    return G


def compute_bandgap_derivative(
    ne: NDArray[np.float64],
    phase_state: NDArray[np.int32],
) -> NDArray[np.float64]:
    """バンドギャップのキャリア密度偏微分 ∂Eg/∂ne を計算する。

    固相: ∂Eg/∂ne = -(1.5e-8 / 3) × ne^(-2/3)  [eV·cm³]
    液相: 0

    Args:
        ne: キャリア密度 [cm⁻³], shape: (n_z,)
        phase_state: 相状態, shape: (n_z,)

    Returns:
        ∂Eg/∂ne [eV·cm³], shape: (n_z,)
    """
    dEg_dne = np.zeros_like(ne)
    solid_mask = _is_solid_phase(phase_state)
    ne_safe = np.maximum(ne[solid_mask], 1.0)
    dEg_dne[solid_mask] = -(1.5e-8 / 3.0) * ne_safe ** (-2.0 / 3.0)
    return dEg_dne


def compute_bandgap_derivative_tl(
    Tl: NDArray[np.float64],
    phase_state: NDArray[np.int32],
) -> NDArray[np.float64]:
    """バンドギャップの格子温度偏微分 ∂Eg/∂Tl を計算する。

    固相: ∂Eg/∂Tl = -7.02e-4 × Tl × (Tl + 2216) / (Tl + 1108)²  [eV/K]
    液相: 0

    Args:
        Tl: 格子温度 [K], shape: (n_z,)
        phase_state: 相状態, shape: (n_z,)

    Returns:
        ∂Eg/∂Tl [eV/K], shape: (n_z,)
    """
    dEg_dTl = np.zeros_like(Tl)

    solid_mask = _is_solid_phase(phase_state)

    Tl_safe = np.maximum(Tl[solid_mask], 1.0)
    dEg_dTl[solid_mask] = (
        -7.02e-4 * Tl_safe * (Tl_safe + 2216.0) / (Tl_safe + 1108.0) ** 2
    )

    return dEg_dTl


# ========================================
# プライベート補助関数
# ========================================

def _is_solid_phase(phase_state: NDArray[np.int32]) -> NDArray[np.bool_]:
    """固相判定のマスクを作成する。
    
    固相: SOLID または MELTING
    
    Args:
        phase_state: 相状態, shape: (n_z,)
    
    Returns:
        固相マスク, shape: (n_z,)
    """
    return (phase_state == PhaseState.SOLID) | (phase_state == PhaseState.MELTING)


def _compute_bandgap_solid(
    Tl: NDArray[np.float64],
    ne: NDArray[np.float64],
) -> NDArray[np.float64]:
    """固相でのバンドギャップエネルギーを計算する（Varshni式 + BGR）。

    E_g = 1.16 - 7.02e-4 × T_l² / (T_l + 1108) - 1.5e-8 × ne^(1/3)

    Args:
        Tl: 格子温度 [K], shape: (n_solid,)
        ne: キャリア密度 [cm⁻³], shape: (n_solid,)

    Returns:
        E_g [eV], shape: (n_solid,)
    """
    varshni = 1.16 - 7.02e-4 * Tl**2 / (Tl + 1108.0)
    bgr = 1.5e-8 * ne ** (1.0 / 3.0)
    return varshni - bgr
