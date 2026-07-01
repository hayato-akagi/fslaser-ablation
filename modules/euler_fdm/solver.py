"""modules/euler_fdm/solver.py — 統合ソルバー（メインシミュレーションループ）

パターンB（時間発展モジュール）として、euler_fdm の全ロジックを統合。
シミュレーション全体のロードマップと時間発展計算を一元管理する。

構成：
1. 公開関数: run_simulation_impl()
2. 初期化: _initialize_simulation(), initialize_state_vectors()
3. メインループ: _execute_time_loop(), _execute_one_iteration()
4. 1ステップ実行: _execute_one_step(), _update_state_vectors()
5. スナップショット: _collect_snapshot_data(), _record_snapshot_if_needed()
6. 結果収集: _collect_results()
7. 補助ロジック: compute_safe_dt(), create_domain_configs(), should_record_snapshot()
"""

import numpy as np
from numpy.typing import NDArray

from modules.ablation.config import AblationConfig
from modules.ablation.public import evaluate_ablation
from modules.carrier.config import CarrierConfig
from modules.carrier.public import advance_carrier_density
from modules.euler_fdm.config import EulerFDMConfig, GridConfig, InitialCondition
from modules.euler_fdm.public import SimulationResult
from modules.material_properties.constants import SILICON
from modules.material_properties.unit_conversions import convert_cm_to_nm
from modules.optics.config import OpticsConfig
from modules.optics.public import compute_laser_field
from modules.ttm.config import TTMConfig
from modules.ttm.public import advance_temperatures


# ============================================================================
# 公開関数（外部API）
# ============================================================================


def run_simulation_impl(config: EulerFDMConfig) -> SimulationResult:
    """シミュレーション全体を実行する実装関数。
    
    Args:
        config: euler_fdm の全設定
    
    Returns:
        SimulationResult: 最終状態とスナップショット履歴
    """
    # 1. 初期化フェーズ
    state = _initialize_simulation(config)
    
    # 2. メインタイムループ
    history = _execute_time_loop(state, config)
    
    # 3. 結果の収集
    result = _collect_results(state, history, config)
    
    return result


# ============================================================================
# 初期化
# ============================================================================


def _initialize_simulation(config: EulerFDMConfig) -> dict:
    """シミュレーションの初期化を行う。
    
    Args:
        config: euler_fdm 設定
    
    Returns:
        初期状態を含む state 辞書
    """
    # 各ドメイン Config の生成
    optics_config, carrier_config, ttm_config, ablation_config = create_domain_configs(config)
    
    # 状態ベクトルの初期化
    (
        ne,
        Te,
        Tl,
        phase_state,
        latent_heat_acc,
        cumulative_ablated_mask,
        dTl_dt_prev,
    ) = initialize_state_vectors(config.grid, config.initial)
    
    # 状態辞書を構築
    state = {
        "ne": ne,
        "Te": Te,
        "Tl": Tl,
        "phase_state": phase_state,
        "latent_heat_acc": latent_heat_acc,
        "max_ablation_depth": 0.0,
        "cumulative_ablated_mask": cumulative_ablated_mask,
        "dTl_dt_prev": dTl_dt_prev,
        "t": config.time.t_start,
        "step_index": 0,
        "optics_config": optics_config,
        "carrier_config": carrier_config,
        "ttm_config": ttm_config,
        "ablation_config": ablation_config,
    }
    
    return state


def initialize_state_vectors(
    grid: GridConfig,
    initial: InitialCondition,
) -> tuple[
    NDArray[np.float64],  # ne
    NDArray[np.float64],  # Te
    NDArray[np.float64],  # Tl
    NDArray[np.int32],    # phase_state
    NDArray[np.float64],  # latent_heat_acc
    NDArray[np.bool_],    # cumulative_ablated_mask
    NDArray[np.float64],  # dTl_dt_prev
]:
    """状態ベクトルを初期化する。
    
    全てのグリッド点で uniform な初期値を設定する。
    
    Args:
        grid: グリッド設定
        initial: 初期条件
    
    Returns:
        初期化された全状態ベクトルのタプル
    """
    n_z = grid.n_z
    
    ne = np.full(n_z, initial.ne_init, dtype=np.float64)
    Te = np.full(n_z, initial.Te_init, dtype=np.float64)
    Tl = np.full(n_z, initial.Tl_init, dtype=np.float64)
    phase_state = np.zeros(n_z, dtype=np.int32)  # PhaseState.SOLID = 0
    latent_heat_acc = np.zeros(n_z, dtype=np.float64)
    cumulative_ablated_mask = np.zeros(n_z, dtype=np.bool_)
    dTl_dt_prev = np.zeros(n_z, dtype=np.float64)
    
    return (
        ne,
        Te,
        Tl,
        phase_state,
        latent_heat_acc,
        cumulative_ablated_mask,
        dTl_dt_prev,
    )


# ============================================================================
# メインループ
# ============================================================================


def _execute_time_loop(state: dict, config: EulerFDMConfig) -> dict:
    """メインタイムループを実行する。
    
    Args:
        state: 現在の状態辞書（in-place で更新される）
        config: euler_fdm 設定
    
    Returns:
        スナップショット履歴を含む辞書
    """
    history = _create_history_dict()
    
    while state["t"] < config.time.t_end:
        _execute_one_iteration(state, config, history)
    
    return history


def _execute_one_iteration(state: dict, config: EulerFDMConfig, history: dict) -> None:
    """1イテレーション（時間刻み決定 + 1ステップ実行 + 記録）を実行する。
    
    Args:
        state: 状態辞書（in-place で更新）
        config: euler_fdm 設定
        history: スナップショット履歴辞書（in-place で更新）
    """
    dt = config.time.dt_max

    # 1ステップ実行
    snapshot_data = _execute_one_step(state, dt)
    
    # スナップショット記録
    _record_snapshot_if_needed(state, snapshot_data, config, history)

    # 進捗コールバック（snapshot と同タイミング）
    _call_progress_if_needed(state, config, history)

    # 時刻とステップカウンタを更新
    state["t"] += dt
    state["step_index"] += 1


# ============================================================================
# 1ステップ実行
# ============================================================================


def _execute_one_step(state: dict, dt: float) -> dict:
    """1タイムステップの5段階処理を実行する。
    
    Args:
        state: 状態辞書（in-place で更新）
        dt: 時間刻み [s]
    
    Returns:
        スナップショット用の表面値を含む辞書
    """
    # Step 1: レーザー場計算
    optics_result = compute_laser_field(
        ne=state["ne"],
        Tl=state["Tl"],
        Te=state["Te"],
        phase_state=state["phase_state"],
        t=state["t"],
        config=state["optics_config"],
    )
    
    # Step 2: キャリア密度更新
    carrier_result = advance_carrier_density(
        ne=state["ne"],
        intensity=optics_result.intensity,
        Te=state["Te"],
        Tl=state["Tl"],
        phase_state=state["phase_state"],
        dt=dt,
        config=state["carrier_config"],
    )
    
    # Step 3: 温度更新（前ステップの Tl を保存）
    # Forward Euler: 全微分は時刻 t の値で計算するため更新前の ne を使用
    Tl_before = state["Tl"].copy()
    ttm_result = advance_temperatures(
        Te=state["Te"],
        Tl=state["Tl"],
        ne=state["ne"],
        dne_dt=carrier_result.dne_dt,
        source_term=optics_result.source_term,
        phase_state=state["phase_state"],
        latent_heat_accumulated=state["latent_heat_acc"],
        dt=dt,
        config=state["ttm_config"],
    )
    
    # Step 4: アブレーション判定
    ablation_result = evaluate_ablation(
        Tl=ttm_result.Tl,
        dz=state["optics_config"].dz,
        config=state["ablation_config"],
    )
    
    # Step 5: 状態更新
    _update_state_vectors(state, carrier_result, ttm_result, ablation_result, Tl_before, dt)
    
    # スナップショット用データを収集
    snapshot_data = _collect_snapshot_data(
        state,
        optics_result,
        ablation_result,
    )
    
    return snapshot_data


def _update_state_vectors(
    state: dict,
    carrier_result,
    ttm_result,
    ablation_result,
    Tl_before: NDArray[np.float64],
    dt: float,
) -> None:
    """状態ベクトルを更新する。
    
    Args:
        state: 状態辞書（in-place で更新）
        carrier_result: CarrierResult
        ttm_result: TTMResult
        ablation_result: AblationResult
        Tl_before: 更新前の Tl（dTl/dt 計算用）
        dt: 時間刻み [s]
    """
    state["ne"] = carrier_result.ne
    state["Te"] = ttm_result.Te
    state["Tl"] = ttm_result.Tl
    state["phase_state"] = ttm_result.phase_state
    state["latent_heat_acc"] = ttm_result.latent_heat_accumulated
    state["max_ablation_depth"] = max(
        state["max_ablation_depth"],
        ablation_result.ablation_depth,
    )
    state["cumulative_ablated_mask"] |= ablation_result.ablated_mask
    state["dTl_dt_prev"] = (ttm_result.Tl - Tl_before) / dt

    # アブレートしたセルを除去（気化した物質はグリッドから取り除く）
    _reset_ablated_cells(state)


def _ablation_front_index(ablated_mask: NDArray[np.bool_]) -> int:
    """アブレーション前線（最初の非アブレートセル）のインデックスを返す。

    アブレートセルがなければ 0（表面）を返す。
    """
    if not ablated_mask.any():
        return 0
    non_ablated = np.where(~ablated_mask)[0]
    return int(non_ablated[0]) if len(non_ablated) > 0 else 0


def _reset_ablated_cells(state: dict) -> None:
    """アブレートしたセルを室温・初期キャリア密度にリセットする。

    気化した物質は領域外に除去されたとみなし、次ステップのレーザー伝播で
    その下の層が新たな表面として扱われる。
    """
    mask = state["cumulative_ablated_mask"]
    if not mask.any():
        return
    T_room = state["ttm_config"].T_room
    state["Te"][mask] = T_room
    state["Tl"][mask] = T_room
    state["ne"][mask] = 1.0e12
    state["phase_state"][mask] = 0   # PhaseState.SOLID
    state["latent_heat_acc"][mask] = 0.0


# ============================================================================
# スナップショット記録
# ============================================================================


def _collect_snapshot_data(state: dict, optics_result, ablation_result) -> dict:
    """スナップショット記録用のデータを収集する。
    
    Args:
        state: 状態辞書
        optics_result: OpticsResult
        ablation_result: AblationResult
    
    Returns:
        スナップショット用データ辞書
    """
    ne_surface = state["ne"][0]
    auger_term = compute_auger_term_surface(ne_surface)
    front_idx = _ablation_front_index(state["cumulative_ablated_mask"])

    return {
        "t": state["t"],
        "Te_surface": state["Te"][0],
        "Tl_surface": state["Tl"][0],
        "ne_surface": ne_surface,
        "reflectivity": optics_result.reflectivity,
        "alpha_fca_surface": optics_result.alpha_fca[0],
        "auger_term_surface": auger_term,
        "ablation_depth_cm": ablation_result.ablation_depth,
        "Tl_front": state["Tl"][front_idx],
        "Te_front": state["Te"][front_idx],
    }


def _record_snapshot_if_needed(
    state: dict,
    snapshot_data: dict,
    config: EulerFDMConfig,
    history: dict,
) -> None:
    """必要に応じてスナップショットを記録する。
    
    Args:
        state: 状態辞書
        snapshot_data: スナップショット用データ
        config: euler_fdm 設定
        history: スナップショット履歴辞書（in-place で更新）
    """
    if not should_record_snapshot(state["step_index"], config.time.snapshot_interval):
        return
    
    history["time_points"].append(snapshot_data["t"])
    history["Te_surface_history"].append(snapshot_data["Te_surface"])
    history["Tl_surface_history"].append(snapshot_data["Tl_surface"])
    history["ne_surface_history"].append(snapshot_data["ne_surface"])
    history["reflectivity_history"].append(snapshot_data["reflectivity"])
    history["alpha_fca_surface_history"].append(snapshot_data["alpha_fca_surface"])
    history["auger_term_surface_history"].append(snapshot_data["auger_term_surface"])
    history["Tl_front_history"].append(snapshot_data["Tl_front"])
    history["Te_front_history"].append(snapshot_data["Te_front"])

    # アブレーション深さは nm に変換して記録
    ablation_depth_nm = convert_cm_to_nm(snapshot_data["ablation_depth_cm"])
    history["ablation_depth_history"].append(ablation_depth_nm)


def _create_history_dict() -> dict:
    """スナップショット履歴を格納する辞書を作成する。
    
    Returns:
        空のリストで初期化された履歴辞書
    """
    return {
        "time_points": [],
        "Te_surface_history": [],
        "Tl_surface_history": [],
        "ne_surface_history": [],
        "reflectivity_history": [],
        "alpha_fca_surface_history": [],
        "auger_term_surface_history": [],
        "ablation_depth_history": [],
        "Tl_front_history": [],
        "Te_front_history": [],
    }


# ============================================================================
# 結果収集
# ============================================================================


def _collect_results(state: dict, history: dict, config: EulerFDMConfig) -> SimulationResult:
    """最終結果を収集して SimulationResult を構築する。
    
    Args:
        state: 最終状態辞書
        history: スナップショット履歴辞書
        config: euler_fdm 設定
    
    Returns:
        SimulationResult インスタンス
    """
    # 履歴リストを numpy 配列に変換
    history_arrays = {
        key: np.array(values, dtype=np.float64)
        for key, values in history.items()
    }
    
    # アブレーション深さを nm に変換
    ablation_depth_nm = convert_cm_to_nm(state["max_ablation_depth"])
    
    return SimulationResult(
        # 最終状態
        Te_final=state["Te"],
        Tl_final=state["Tl"],
        ne_final=state["ne"],
        # アブレーション結果
        ablation_depth_cm=state["max_ablation_depth"],
        ablation_depth_nm=ablation_depth_nm,
        ablated_mask=state["cumulative_ablated_mask"],
        # 時間履歴
        time_points=history_arrays["time_points"],
        Te_surface_history=history_arrays["Te_surface_history"],
        Tl_surface_history=history_arrays["Tl_surface_history"],
        ne_surface_history=history_arrays["ne_surface_history"],
        reflectivity_history=history_arrays["reflectivity_history"],
        alpha_fca_surface_history=history_arrays["alpha_fca_surface_history"],
        auger_term_surface_history=history_arrays["auger_term_surface_history"],
        ablation_depth_history=history_arrays["ablation_depth_history"],
        # メタデータ
        total_steps=state["step_index"],
        fluence=config.fluence,
    )


# ============================================================================
# 補助ロジック（CFL条件、Config生成、判定）
# ============================================================================


def compute_safe_dt(
    Te: NDArray[np.float64],
    Tl: NDArray[np.float64],
    ne: NDArray[np.float64],
    dz: float,
    dt_max: float,
    source_term: NDArray[np.float64] | None = None,
) -> float:
    """CFL安定性条件に基づく安全な時間刻みを計算する。

    dt_cfl = 0.4 * dz² / max(alpha_e_max, alpha_l_max)
    dt     = min(dt_cfl, dt_max)

    Args:
        Te: 電子温度 [K], shape (n_z,)
        Tl: 格子温度 [K], shape (n_z,)
        ne: キャリア密度 [cm⁻³], shape (n_z,)
        dz: グリッド間隔 [cm]
        dt_max: 最大時間刻み [s]
        source_term: 未使用（シグネチャ互換性のために保持）

    Returns:
        安全な時間刻み [s]
    """
    from modules import material_properties

    phase_state = np.zeros_like(Te, dtype=np.int32)

    Cl = material_properties.compute_thermal_capacity_lattice(Tl, phase_state)
    Kl = material_properties.compute_thermal_conductivity_lattice(Tl, phase_state, SILICON.T_m)

    Cl_safe = np.maximum(Cl, 1e-30)
    alpha_l_max = float(np.max(Kl / Cl_safe))

    # 格子 CFL のみ評価。te_scheme="euler" 時は Te CFL を dt_max で外部管理すること。
    dt_cfl = 0.4 * dz**2 / max(alpha_l_max, 1e-30)

    return min(dt_cfl, dt_max)


def create_domain_configs(
    config: EulerFDMConfig,
) -> tuple[OpticsConfig, CarrierConfig, TTMConfig, AblationConfig]:
    """各ドメインモジュールの Config インスタンスを生成し、dz を注入する。
    
    Args:
        config: euler_fdm の最上位設定
    
    Returns:
        (optics_config, carrier_config, ttm_config, ablation_config) のタプル
    """
    dz = config.grid.dz
    fluence = config.fluence
    
    # 各ドメインの Config を生成（dz を注入）
    optics_config = OpticsConfig(dz=dz, fluence=fluence)
    carrier_config = CarrierConfig(dz=dz)
    ttm_config = TTMConfig(dz=dz, te_scheme=config.te_scheme)
    ablation_config = AblationConfig()
    
    return optics_config, carrier_config, ttm_config, ablation_config


def should_record_snapshot(step_index: int, snapshot_interval: int) -> bool:
    """このステップでスナップショットを記録すべきか判定する。
    
    Args:
        step_index: 現在のステップ番号（0始まり）
        snapshot_interval: スナップショット記録間隔 [ステップ数]
    
    Returns:
        True なら記録する
    """
    return step_index % snapshot_interval == 0


def _call_progress_if_needed(state: dict, config: EulerFDMConfig, history: dict) -> None:
    """進捗コールバックを snapshot_interval ごとに呼び出す。"""
    if config.progress_callback is None:
        return
    if not should_record_snapshot(state["step_index"], config.time.snapshot_interval):
        return
    config.progress_callback(
        state["step_index"],
        state["t"],
        config.time.t_end,
        history,
    )


def compute_auger_term_surface(ne_surface: float) -> float:
    """表面（z=0）のオージェ再結合項を計算する。
    
    Args:
        ne_surface: 表面のキャリア密度 [cm⁻³]
    
    Returns:
        γ × ne³ [cm⁻³/s]
    """
    gamma = SILICON.gamma_auger
    return gamma * ne_surface**3
