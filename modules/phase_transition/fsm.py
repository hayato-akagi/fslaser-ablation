"""modules/phase_transition/fsm — 相転移の有限状態機械（FSM）。

シリコンの固相・液相・気相間の遷移を管理する。
潜熱の蓄積・消費を追跡し、温度を適切に制限する。
"""

import numpy as np
from numpy.typing import NDArray

from modules import PhaseState
from modules.phase_transition.config import PhaseTransitionConfig


def apply_phase_transitions(
    Tl: NDArray[np.float64],
    rhs_l: NDArray[np.float64],
    Cl: NDArray[np.float64],
    phase_state: NDArray[np.int32],
    latent_heat_accumulated: NDArray[np.float64],
    dt: float,
    config: PhaseTransitionConfig,
) -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.float64]]:
    """相転移判定と潜熱処理を適用する。
    
    各グリッド点で格子温度の更新と相転移判定を行う。
    相転移時は潜熱を蓄積・消費し、温度を相境界値に固定する。
    
    Args:
        Tl: 格子温度 [K], shape: (n_z,)
        rhs_l: 格子温度方程式の RHS [W/cm³], shape: (n_z,)
        Cl: 格子熱容量 [J/(cm³·K)], shape: (n_z,)
        phase_state: 現在の相状態, shape: (n_z,)
        latent_heat_accumulated: 現在の潜熱蓄積量 [J/cm³], shape: (n_z,)
        dt: 時間刻み [s]
        config: 相転移設定
    
    Returns:
        更新後の (Tl, phase_state, latent_heat_accumulated)
    """
    # コピーを作成（不変性保証）
    Tl_new = Tl.copy()
    phase_state_new = phase_state.copy()
    latent_new = latent_heat_accumulated.copy()
    
    n_z = len(Tl)
    for i in range(n_z):
        Tl_new[i], phase_state_new[i], latent_new[i] = _process_single_grid_point(
            Tl=Tl_new[i],
            rhs_l=rhs_l[i],
            Cl=Cl[i],
            phase_state=phase_state_new[i],
            latent_heat=latent_new[i],
            dt=dt,
            config=config,
        )
    
    return Tl_new, phase_state_new, latent_new


def _process_single_grid_point(
    Tl: float,
    rhs_l: float,
    Cl: float,
    phase_state: int,
    latent_heat: float,
    dt: float,
    config: PhaseTransitionConfig,
) -> tuple[float, int, float]:
    """単一グリッド点の相転移処理。
    
    現在の相状態に応じて適切な処理関数にディスパッチする。
    
    Args:
        Tl: 格子温度 [K]
        rhs_l: 格子温度方程式の RHS [W/cm³]
        Cl: 格子熱容量 [J/(cm³·K)]
        phase_state: 現在の相状態
        latent_heat: 現在の潜熱蓄積量 [J/cm³]
        dt: 時間刻み [s]
        config: 相転移設定
    
    Returns:
        更新後の (Tl, phase_state, latent_heat)
    """
    if phase_state == PhaseState.SOLID:
        return _process_solid_phase(Tl, rhs_l, Cl, latent_heat, dt, config)
    elif phase_state == PhaseState.MELTING:
        return _process_melting_phase(Tl, rhs_l, Cl, latent_heat, dt, config)
    elif phase_state == PhaseState.LIQUID:
        return _process_liquid_phase(Tl, rhs_l, Cl, latent_heat, dt, config)
    elif phase_state == PhaseState.VAPORIZING:
        return _process_vaporizing_phase(Tl, rhs_l, latent_heat, dt, config)
    else:  # VAPOR
        return _process_vapor_phase(Tl, rhs_l, Cl, dt, config)


def _process_solid_phase(
    Tl: float,
    rhs_l: float,
    Cl: float,
    latent_heat: float,
    dt: float,
    config: PhaseTransitionConfig,
) -> tuple[float, int, float]:
    """SOLID 相の処理。
    
    通常の温度更新を行い、融点に達した場合は MELTING へ遷移する。
    
    Args:
        Tl: 格子温度 [K]
        rhs_l: 格子温度方程式の RHS [W/cm³]
        Cl: 格子熱容量 [J/(cm³·K)]
        latent_heat: 現在の潜熱蓄積量 [J/cm³]
        dt: 時間刻み [s]
        config: 相転移設定
    
    Returns:
        更新後の (Tl, phase_state, latent_heat)
    """
    Tl_next = Tl + dt * rhs_l / Cl
    
    # 安全チェック: NaN/Inf 除去
    if not np.isfinite(Tl_next):
        Tl_next = Tl
    
    # 温度範囲制限
    Tl_next = np.clip(Tl_next, config.T_room, 1e6)
    
    if Tl_next >= config.T_m:
        # 融点到達 → MELTING へ遷移
        excess_energy = Cl * (Tl_next - config.T_m)
        return config.T_m, PhaseState.MELTING, latent_heat + excess_energy
    else:
        # 通常の温度上昇
        return Tl_next, PhaseState.SOLID, latent_heat


def _process_melting_phase(
    Tl: float,
    rhs_l: float,
    Cl: float,
    latent_heat: float,
    dt: float,
    config: PhaseTransitionConfig,
) -> tuple[float, int, float]:
    """MELTING 相の処理。
    
    温度を融点に固定し、入力エネルギーを潜熱に蓄積する。
    潜熱が融解潜熱に達したら LIQUID へ遷移する。
    
    Args:
        Tl: 格子温度 [K]（融点に固定されている）
        rhs_l: 格子温度方程式の RHS [W/cm³]
        Cl: 格子熱容量 [J/(cm³·K)]
        latent_heat: 現在の潜熱蓄積量 [J/cm³]
        dt: 時間刻み [s]
        config: 相転移設定
    
    Returns:
        更新後の (Tl, phase_state, latent_heat)
    """
    # エネルギーを潜熱に蓄積
    energy_input = dt * rhs_l  # [W/cm³] * [s] = [J/cm³]
    latent_heat_new = latent_heat + energy_input
    
    if latent_heat_new >= config.L_m:
        # 融解完了 → LIQUID へ遷移
        remaining_energy = latent_heat_new - config.L_m
        Tl_new = config.T_m + remaining_energy / Cl
        return Tl_new, PhaseState.LIQUID, 0.0
    else:
        # まだ融解中
        return config.T_m, PhaseState.MELTING, latent_heat_new


def _process_liquid_phase(
    Tl: float,
    rhs_l: float,
    Cl: float,
    latent_heat: float,
    dt: float,
    config: PhaseTransitionConfig,
) -> tuple[float, int, float]:
    """LIQUID 相の処理。
    
    通常の温度更新を行い、沸点に達した場合は VAPORIZING へ遷移する。
    
    Args:
        Tl: 格子温度 [K]
        rhs_l: 格子温度方程式の RHS [W/cm³]
        Cl: 格子熱容量 [J/(cm³·K)]
        latent_heat: 現在の潜熱蓄積量 [J/cm³]
        dt: 時間刻み [s]
        config: 相転移設定
    
    Returns:
        更新後の (Tl, phase_state, latent_heat)
    """
    Tl_next = Tl + dt * rhs_l / Cl
    
    # 安全チェック: NaN/Inf 除去
    if not np.isfinite(Tl_next):
        Tl_next = Tl
    
    # 温度範囲制限
    Tl_next = np.clip(Tl_next, config.T_room, 1e6)
    
    if Tl_next >= config.T_b:
        # 沸点到達 → VAPORIZING へ遷移
        excess_energy = Cl * (Tl_next - config.T_b)
        # 安全チェック
        if not np.isfinite(excess_energy):
            excess_energy = 0.0
        return config.T_b, PhaseState.VAPORIZING, latent_heat + excess_energy
    else:
        # 通常の温度上昇
        return Tl_next, PhaseState.LIQUID, latent_heat


def _process_vaporizing_phase(
    Tl: float,
    rhs_l: float,
    latent_heat: float,
    dt: float,
    config: PhaseTransitionConfig,
) -> tuple[float, int, float]:
    """VAPORIZING 相の処理。
    
    温度を沸点に固定し、入力エネルギーを潜熱に蓄積する。
    潜熱が蒸発潜熱に達したら VAPOR へ遷移する。
    
    Args:
        Tl: 格子温度 [K]（沸点に固定されている）
        rhs_l: 格子温度方程式の RHS [W/cm³]
        latent_heat: 現在の潜熱蓄積量 [J/cm³]
        dt: 時間刻み [s]
        config: 相転移設定
    
    Returns:
        更新後の (Tl, phase_state, latent_heat)
    """
    # エネルギーを潜熱に蓄積
    energy_input = dt * rhs_l  # [W/cm³] * [s] = [J/cm³]
    latent_heat_new = latent_heat + energy_input
    
    if latent_heat_new >= config.L_v:
        # 気化完了 → VAPOR へ遷移
        # Cl は固相式をそのまま使用（沸点での値）
        Cl_at_boiling = 1.978 + 3.54e-4 * config.T_b - 3.68 / (config.T_b ** 2)
        remaining_energy = latent_heat_new - config.L_v
        Tl_new = config.T_b + remaining_energy / Cl_at_boiling
        return Tl_new, PhaseState.VAPOR, 0.0
    else:
        # まだ気化中
        return config.T_b, PhaseState.VAPORIZING, latent_heat_new


def _process_vapor_phase(
    Tl: float,
    rhs_l: float,
    Cl: float,
    dt: float,
    config: PhaseTransitionConfig,
) -> tuple[float, int, float]:
    """VAPOR 相の処理。
    
    通常の温度更新を行う。相転移は発生しない。
    
    Args:
        Tl: 格子温度 [K]
        rhs_l: 格子温度方程式の RHS [W/cm³]
        Cl: 格子熱容量 [J/(cm³·K)]
        dt: 時間刻み [s]
        config: 相転移設定
    
    Returns:
        更新後の (Tl, phase_state, latent_heat)
    """
    Tl_next = Tl + dt * rhs_l / Cl
    
    # 安全チェック: NaN/Inf 除去
    if not np.isfinite(Tl_next):
        Tl_next = Tl
    
    # 温度範囲制限
    Tl_next = np.clip(Tl_next, config.T_room, 1e6)
    
    return Tl_next, PhaseState.VAPOR, 0.0
