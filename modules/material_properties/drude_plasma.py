"""modules/material_properties/drude_plasma — Drude-Plasma モデルの散乱頻度計算。

τ_e の動的計算に用いる電子-格子・電子-イオン散乱頻度の純粋関数群。
silicon.py の compute_tau_e から内部呼び出しされる。
"""

import numpy as np
from numpy.typing import NDArray

from modules.material_properties.constants import PHYSICAL, SILICON
from modules.material_properties.unit_conversions import convert_cm3_to_m3


def compute_nu_phonon(Tl: NDArray[np.float64]) -> NDArray[np.float64]:
    """電子-格子（フォノン）散乱頻度 [s⁻¹]。

    ν_phonon = (1/τ_0) × (Tl / T_room)

    Args:
        Tl: 格子温度 [K], shape: (n,)

    Returns:
        ν_phonon [s⁻¹], shape: (n,)
    """
    return (1.0 / SILICON.tau_e_phonon_base) * (Tl / SILICON.T_room)


def compute_debye_length(
    ne_m3: NDArray[np.float64],
    Te: NDArray[np.float64],
) -> NDArray[np.float64]:
    """デバイ遮蔽長 λ_D [m]。

    λ_D = sqrt(ε_s k_B T_e / (n_e e²))

    Args:
        ne_m3: キャリア密度 [m⁻³], shape: (n,)
        Te: 電子温度 [K], shape: (n,)

    Returns:
        λ_D [m], shape: (n,)
    """
    safe_ne = np.maximum(ne_m3, 1.0)
    return np.sqrt(
        SILICON.epsilon_s * PHYSICAL.k_B * Te
        / (safe_ne * PHYSICAL.e_charge ** 2)
    )


def compute_b_min(Te: NDArray[np.float64]) -> NDArray[np.float64]:
    """最短接近距離 b_min = max(b_classical, b_quantum) [m]。

    b_cl = e² / (12π ε_s k_B T_e)   （古典的最接近距離）
    b_qu = ℏ / sqrt(2 m* k_B T_e)   （量子的ド・ブロイ半径）

    Args:
        Te: 電子温度 [K], shape: (n,)

    Returns:
        b_min [m], shape: (n,)
    """
    h_bar_J = PHYSICAL.h_plank / (2.0 * np.pi)
    kBTe = PHYSICAL.k_B * Te

    b_cl = PHYSICAL.e_charge ** 2 / (12.0 * np.pi * SILICON.epsilon_s * kBTe)
    b_qu = h_bar_J / np.sqrt(2.0 * SILICON.m_eff_optical * kBTe)

    return np.maximum(b_cl, b_qu)


def compute_coulomb_log(
    ne_m3: NDArray[np.float64],
    Te: NDArray[np.float64],
) -> NDArray[np.float64]:
    """クーロン対数 ln(Λ) = max(ln(λ_D / b_min), 1.0)。

    高密度極限での負値・1未満をクランプして数値安定性を保つ。

    Args:
        ne_m3: キャリア密度 [m⁻³], shape: (n,)
        Te: 電子温度 [K], shape: (n,)

    Returns:
        ln(Λ) [無次元], shape: (n,)
    """
    lambda_D = compute_debye_length(ne_m3, Te)
    b_min = compute_b_min(Te)
    return np.maximum(np.log(lambda_D / b_min), 1.0)


def compute_spitzer_coefficient() -> float:
    """Spitzer係数 C_ei を基本物理定数から導出する [m³·K^(3/2)·s⁻¹]。

    C_ei = e⁴ / (12√2 π^(3/2) ε_s² m*^(1/2) k_B^(3/2))

    Returns:
        C_ei [m³·K^(3/2)·s⁻¹]
    """
    numerator = PHYSICAL.e_charge ** 4
    denominator = (
        12.0 * np.sqrt(2.0) * np.pi ** 1.5
        * SILICON.epsilon_s ** 2
        * np.sqrt(SILICON.m_eff_optical)
        * PHYSICAL.k_B ** 1.5
    )
    return numerator / denominator


def compute_nu_ee(
    ne: NDArray[np.float64],
    Te: NDArray[np.float64],
) -> NDArray[np.float64]:
    """電子-電子散乱頻度 ν_ee [s⁻¹]（Yoffa/Chen モデル）。

    高励起密度プラズマで支配的になる散乱機構。Te および ne に比例して増大し、
    Spitzer 式とは逆に高温・高密度で τ_e を短縮させる。

    ν_ee = c_ee_base × (Te / T_room) × (ne_m3 / n0_valence_m3)

    Args:
        ne: キャリア密度 [cm⁻³], shape: (n,)
        Te: 電子温度 [K], shape: (n,)

    Returns:
        ν_ee [s⁻¹], shape: (n,)
    """
    ne_m3 = convert_cm3_to_m3(ne)
    return (
        SILICON.c_ee_base
        * (Te / SILICON.T_room)
        * (ne_m3 / SILICON.n0_valence_m3)
    )


def compute_nu_ei_spitzer(
    ne: NDArray[np.float64],
    Te: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Spitzer型電子-イオン散乱頻度 ν_ei [s⁻¹]。

    ν_ei = C_ei × n_e × ln(Λ) / T_e^(3/2)

    Args:
        ne: キャリア密度 [cm⁻³], shape: (n,)
        Te: 電子温度 [K], shape: (n,)

    Returns:
        ν_ei [s⁻¹], shape: (n,)
    """
    ne_m3 = convert_cm3_to_m3(ne)
    safe_Te = np.maximum(Te, 1.0)
    ln_lambda = compute_coulomb_log(ne_m3, safe_Te)
    C_ei = compute_spitzer_coefficient()
    return C_ei * ne_m3 * ln_lambda / (safe_Te ** 1.5)
