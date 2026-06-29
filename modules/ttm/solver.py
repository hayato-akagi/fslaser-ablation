"""modules/ttm/solver — 二温度モデル（TTM）の時間発展計算。

Predictor-Corrector Crank-Nicolson 法による1次元有限差分法で解く。

更新順序:
  1. 電子温度 Te: CN + Predictor-Corrector
     - 拡散項 ∂/∂z(Ke ∂Te/∂z) は半陰的（Crank-Nicolson）
     - 結合項・熱源・キャリア項は陽的
  2. 格子温度 Tl: 陽的オイラー + 相転移判定

CN 三重対角行列の係数（内点 1 ≤ i ≤ N-2）:
  Ke_{i±1/2} = 0.5 * (Ke[i] + Ke[i±1])
  a_i =  -0.5 * Ke_{i-1/2} / dz²
  b_i = Ce_i/dt + 0.5 * (Ke_{i+1/2} + Ke_{i-1/2}) / dz²
  c_i =  -0.5 * Ke_{i+1/2} / dz²
  rhs_i = Ce_i/dt * Te_i^n
        + 0.5 * [∇(Ke_old ∇Te)]_i   （陽的半）
        + rhs_nondiff_i              （結合・熱源・キャリア項）

境界条件: 断熱 Neumann (dTe/dz = 0), ghost cell 法
  i=0  : a_0 = 0,      Ke_{-1/2} = 0
  i=N-1: c_{N-1} = 0,  Ke_{N-1/2} = 0
"""

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_banded

from modules.ttm.config import TTMConfig
from modules.ttm.public import TTMResult
from modules import material_properties, phase_transition
from modules.phase_transition.config import PhaseTransitionConfig


def advance_temperatures_impl(
    Te: NDArray[np.float64],
    Tl: NDArray[np.float64],
    ne: NDArray[np.float64],
    dne_dt: NDArray[np.float64],
    source_term: NDArray[np.float64],
    phase_state: NDArray[np.int32],
    latent_heat_accumulated: NDArray[np.float64],
    dt: float,
    config: TTMConfig,
) -> TTMResult:
    """1タイムステップ分の温度更新（Predictor-Corrector Crank-Nicolson）。

    シーケンス:
    1. 物性パラメータを取得（旧値 t=n）
    2. 格子温度 RHS を計算 → dTl_dt（陽的、キャリア項で使用）
    3. 電子温度の非拡散項 RHS を計算（陽的）
    4. Predictor-Corrector CN ループで Te^{n+1} を解く
    5. 格子温度を相転移判定込みで更新（陽的）
    6. TTMResult を構築して返す
    """
    # === 1. 物性パラメータ（旧値） ===
    Ce_old  = material_properties.compute_thermal_capacity_electron(Te, ne, phase_state)
    Ke_old  = material_properties.compute_thermal_conductivity_electron(Te, phase_state)
    Cl      = material_properties.compute_thermal_capacity_lattice(Tl, phase_state)
    Kl      = material_properties.compute_thermal_conductivity_lattice(Tl, phase_state, config.T_m)
    G       = material_properties.compute_electron_lattice_coupling(Ce_old, ne, phase_state)
    Eg      = material_properties.compute_bandgap(Tl, ne, phase_state)
    dEg_dne = material_properties.compute_bandgap_derivative(ne, phase_state)
    dEg_dTl = material_properties.compute_bandgap_derivative_tl(Tl, phase_state)

    # === 2. 格子温度 RHS と dTl_dt（陽的） ===
    rhs_l   = _compute_lattice_rhs(Te, Tl, Kl, G, config)
    Cl_safe = np.maximum(Cl, 1e-30)
    dTl_dt  = np.nan_to_num(rhs_l / Cl_safe, nan=0.0, posinf=1e15, neginf=-1e15)

    # === 3. 電子温度の非拡散項 RHS（陽的） ===
    rhs_nondiff = _compute_electron_nondiff_rhs(
        Te, Tl, G, source_term,
        ne, dne_dt, dTl_dt,
        Eg, dEg_dne, dEg_dTl,
        config,
    )

    # === 4. Predictor-Corrector CN で Te^{n+1} を解く ===
    Te_new = _predictor_corrector_step(
        Te, Ce_old, Ke_old, rhs_nondiff, ne, phase_state, dt, config
    )

    # === 5. 格子温度: 相転移判定を適用（陽的） ===
    phase_config = PhaseTransitionConfig(
        T_m=config.T_m,
        T_b=config.T_b,
        L_m=config.L_m,
        L_v=config.L_v,
        T_room=config.T_room,
    )
    Tl_new, phase_state_new, latent_new = phase_transition.apply_phase_transitions(
        Tl=Tl,
        rhs_l=rhs_l,
        Cl=Cl,
        phase_state=phase_state,
        latent_heat_accumulated=latent_heat_accumulated,
        dt=dt,
        config=phase_config,
    )

    # === 6. TTMResult を構築 ===
    return TTMResult(
        Te=Te_new,
        Tl=Tl_new,
        phase_state=phase_state_new,
        latent_heat_accumulated=latent_new,
    )


# ============================================================================
# Predictor-Corrector Crank-Nicolson（電子温度）
# ============================================================================


def _predictor_corrector_step(
    Te: NDArray[np.float64],
    Ce_old: NDArray[np.float64],
    Ke_old: NDArray[np.float64],
    rhs_nondiff: NDArray[np.float64],
    ne: NDArray[np.float64],
    phase_state: NDArray[np.int32],
    dt: float,
    config: TTMConfig,
) -> NDArray[np.float64]:
    """Predictor-Corrector ループで Te^{n+1} を収束まで解く。

    各反復 k:
      1. Ce^(k), Ke^(k) を Te^(k) から計算
      2. CN 三重対角系を構築・求解 → Te^(k+1)
      3. 収束判定: max|Te^(k+1) - Te^(k)| < cn_tol
    """
    Te_new = Te.copy()

    for _ in range(config.cn_max_iter):
        Te_prev = Te_new

        Ce_iter = material_properties.compute_thermal_capacity_electron(Te_new, ne, phase_state)
        Ke_iter = material_properties.compute_thermal_conductivity_electron(Te_new, phase_state)

        ab, rhs = build_cn_matrix(Te, Ce_iter, Ke_iter, Ke_old, rhs_nondiff, dt, config.dz)
        Te_new  = _solve_banded_system(ab, rhs)
        Te_new  = np.clip(Te_new, config.T_room, 1e7)

        if np.max(np.abs(Te_new - Te_prev)) < config.cn_tol:
            break

    return Te_new


def build_cn_matrix(
    Te: NDArray[np.float64],
    Ce: NDArray[np.float64],
    Ke_impl: NDArray[np.float64],
    Ke_expl: NDArray[np.float64],
    rhs_nondiff: NDArray[np.float64],
    dt: float,
    dz: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Crank-Nicolson 三重対角行列を構築する。

    Args:
        Te:          電子温度（旧値 t=n） [K]
        Ce:          電子熱容量（反復値） [J/(cm³·K)]
        Ke_impl:     電子熱伝導率（陰的半用、反復値） [W/(cm·K)]
        Ke_expl:     電子熱伝導率（陽的半用、旧値） [W/(cm·K)]
        rhs_nondiff: 非拡散項 RHS（結合・熱源・キャリア） [W/cm³]
        dt:          時間刻み [s]
        dz:          グリッド間隔 [cm]

    Returns:
        ab:  scipy solve_banded 用バンド行列 shape (3, n_z)
             ab[0, 1:] = 上対角 c[:-1]
             ab[1, :]  = 主対角 b
             ab[2, :-1] = 下対角 a[1:]
        rhs: 右辺ベクトル shape (n_z,)
    """
    n   = len(Te)
    dz2 = dz * dz

    if n == 1:
        ab = np.zeros((3, 1))
        ab[1, 0] = np.maximum(Ce[0], 1e-30) / dt
        rhs = np.maximum(Ce, 1e-30) / dt * Te + rhs_nondiff
        return ab, rhs

    Ke_impl_half = 0.5 * (Ke_impl[:-1] + Ke_impl[1:])  # K_{i+1/2}（陰的半用）

    Ce_safe = np.maximum(Ce, 1e-30)

    # --- 三重対角係数 ---
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)

    # 内点 i = 1, ..., N-2
    Ke_r = Ke_impl_half[1:]   # K_{i+1/2} for i=1,...,N-2
    Ke_l = Ke_impl_half[:-1]  # K_{i-1/2} for i=1,...,N-2
    a[1:-1] = -0.5 * Ke_l / dz2
    b[1:-1] = Ce_safe[1:-1] / dt + 0.5 * (Ke_r + Ke_l) / dz2
    c[1:-1] = -0.5 * Ke_r / dz2

    # 左端 i=0（Neumann: Ke_{-1/2} = 0）
    b[0] = Ce_safe[0] / dt + 0.5 * Ke_impl_half[0] / dz2
    c[0] = -0.5 * Ke_impl_half[0] / dz2

    # 右端 i=N-1（Neumann: Ke_{N-1/2} = 0）
    a[-1] = -0.5 * Ke_impl_half[-1] / dz2
    b[-1] = Ce_safe[-1] / dt + 0.5 * Ke_impl_half[-1] / dz2

    # --- 右辺ベクトル ---
    expl_diff = _compute_thermal_diffusion(Te, Ke_expl, dz)
    rhs = Ce_safe / dt * Te + 0.5 * expl_diff + rhs_nondiff

    # --- scipy banded format ---
    ab = np.zeros((3, n))
    ab[0, 1:]  = c[:-1]   # 上対角（ab[0,0] は未使用）
    ab[1, :]   = b
    ab[2, :-1] = a[1:]    # 下対角（ab[2,-1] は未使用）

    return ab, rhs


def _solve_banded_system(
    ab: NDArray[np.float64],
    rhs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """scipy solve_banded で三重対角系を解く（(1,1)-banded）。"""
    return solve_banded((1, 1), ab, rhs)


# ============================================================================
# 電子温度: 非拡散項（陽的）
# ============================================================================


def _compute_electron_nondiff_rhs(
    Te: NDArray[np.float64],
    Tl: NDArray[np.float64],
    G: NDArray[np.float64],
    source_term: NDArray[np.float64],
    ne: NDArray[np.float64],
    dne_dt: NDArray[np.float64],
    dTl_dt: NDArray[np.float64],
    Eg: NDArray[np.float64],
    dEg_dne: NDArray[np.float64],
    dEg_dTl: NDArray[np.float64],
    config: TTMConfig,
) -> NDArray[np.float64]:
    """電子温度方程式の非拡散項 RHS を計算する（陽的）。

    rhs = -G*(Te-Tl) + S  [+ carrier_energy_term if config.include_carrier_energy_term]
    """
    coupling = -G * (Te - Tl)
    if config.include_carrier_energy_term:
        carrier = _compute_carrier_energy_term(
            Te, ne, dne_dt, dTl_dt, Eg, dEg_dne, dEg_dTl, config
        )
    else:
        carrier = np.zeros_like(Te)
    rhs = coupling + source_term + carrier
    return np.nan_to_num(rhs, nan=0.0, posinf=1e15, neginf=-1e15)


def _compute_carrier_energy_term(
    Te: NDArray[np.float64],
    ne: NDArray[np.float64],
    dne_dt: NDArray[np.float64],
    dTl_dt: NDArray[np.float64],
    Eg: NDArray[np.float64],
    dEg_dne: NDArray[np.float64],
    dEg_dTl: NDArray[np.float64],
    config: TTMConfig,
) -> NDArray[np.float64]:
    """キャリア生成に伴うエネルギー変化項を計算する。

    - (Eg + 3kB*Te + ne*∂Eg/∂ne)*dne_dt - ne*∂Eg/∂Tl*dTl_dt  [W/cm³]
    """
    eV_to_J = 1.602e-19
    coeff_ne = (Eg + 3.0 * config.k_B_eV * Te + ne * dEg_dne) * eV_to_J
    term_ne  = -coeff_ne * dne_dt
    term_tl  = -ne * dEg_dTl * eV_to_J * dTl_dt
    result   = term_ne + term_tl
    return np.nan_to_num(result, nan=0.0, posinf=1e15, neginf=-1e15)


# ============================================================================
# 格子温度: 陽的スキーム
# ============================================================================


def _compute_lattice_rhs(
    Te: NDArray[np.float64],
    Tl: NDArray[np.float64],
    Kl: NDArray[np.float64],
    G: NDArray[np.float64],
    config: TTMConfig,
) -> NDArray[np.float64]:
    """格子温度方程式の RHS を計算する（陽的スキーム）。

    rhs_l = ∇(Kl∇Tl) + G*(Te - Tl)  [W/cm³]
    """
    diffusion = _compute_thermal_diffusion(Tl, Kl, config.dz)
    coupling  = G * (Te - Tl)
    rhs = diffusion + coupling
    return np.nan_to_num(rhs, nan=0.0, posinf=1e15, neginf=-1e15)


def _compute_thermal_diffusion(
    T: NDArray[np.float64],
    K: NDArray[np.float64],
    dz: float,
) -> NDArray[np.float64]:
    """熱拡散項 ∇(K∇T) を中心差分（ベクトル化）で計算する。

    ∇(K∇T)|_i = [K_{i+1/2}*(T_{i+1}-T_i) - K_{i-1/2}*(T_i-T_{i-1})] / dz²
    境界条件: 断熱 Neumann（dT/dz = 0）
    """
    if len(T) == 1:
        return np.zeros_like(T)

    K_half = 0.5 * (K[:-1] + K[1:])
    flux   = K_half * np.diff(T) / dz

    diff = np.zeros_like(T)
    diff[0]    = flux[0] / dz
    diff[1:-1] = (flux[1:] - flux[:-1]) / dz
    diff[-1]   = -flux[-1] / dz

    return diff
