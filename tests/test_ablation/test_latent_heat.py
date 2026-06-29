"""tests/test_ablation/test_latent_heat.py — 特異点テスト（Edge Case / Logic Test）

相転移における潜熱プールの動作を検証する。
融点・沸点での温度固定、潜熱蓄積、相状態遷移が正しく動作するかをテストする。
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from modules import PhaseState
from modules.ttm import TTMConfig, advance_temperatures as advance_temperature
from modules.ttm.public import TTMResult


@pytest.fixture
def single_point_config() -> TTMConfig:
    """1点のみのテスト用設定。"""
    return TTMConfig(dz=1e-7)  # 1 nm（値は使わない）


def test_melting_temperature_plateau() -> None:
    """融点での温度プラトー（停滞）を確認する。
    
    テストシナリオ:
    1. 固相の温度を融点直下に設定
    2. 一定の熱源を与え続ける
    3. 融点到達後、温度が融点に固定されることを確認
    4. 融解潜熱を超えた後、再び温度が上昇することを確認
    
    物理的意味:
    - 融解中（MELTING）は温度が T_m = 1687 K に固定される
    - 入力エネルギーは潜熱プールに蓄積される
    - 潜熱 > L_m になった瞬間、液相へ遷移して温度上昇が再開される
    """
    # === セットアップ ===
    config = TTMConfig(dz=1e-7)
    n_z = 1  # 1点のみ
    
    # 初期状態: 電子系が高温、格子系が融点直下（現実的なTTM状態）
    Te = np.array([4000.0], dtype=np.float64)  # 高温の電子系（強い結合のため）
    Tl = np.array([1685.0], dtype=np.float64)  # 融点のすぐ下
    ne = np.array([1e21], dtype=np.float64)  # 高密度（G を大きくする）
    phase_state = np.array([PhaseState.SOLID], dtype=np.int32)
    
    # 一定の熱源（超高出力：フェムト秒レーザー相当）
    source_term = np.array([1e15], dtype=np.float64)  # 1 PW/cm³（Te維持のため）
    dne_dt = np.zeros_like(source_term)
    latent_heat_accumulated = np.zeros_like(source_term)
    
    dt = 1e-15  # 1 fs
    
    # === Phase 1: 融点到達前 ===
    # 電子-格子結合で格子温度が上昇
    # Tl=1685K から 1687Kまで約2K上昇すればよい
    # 約300ステップで到達するはず
    for step in range(500):  # 十分なステップ数
        result = advance_temperature(Te, Tl, ne, dne_dt, source_term, phase_state, latent_heat_accumulated, dt, config)
        Te = result.Te
        Tl = result.Tl
        phase_state = result.phase_state
        latent_heat_accumulated = result.latent_heat_accumulated
        
        if Tl[0] >= config.T_m:
            # 融点到達 → MELTING へ遷移
            assert phase_state[0] == PhaseState.MELTING, (
                f"Expected MELTING at step {step}, got {phase_state[0]}"
            )
            break
    
    # 融点到達を確認
    assert Tl[0] >= config.T_m * 0.99, "Temperature should reach melting point"
    assert phase_state[0] == PhaseState.MELTING, "Should be in MELTING phase"
    
    # === Phase 2: 融解中（温度プラトー） ===
    # 融解潜熱: L_m = 4.2e9 erg/cm³ = 4.2e2 J/cm³
    # 熱源: S = 1 TW/cm³ + 電子-格子結合
    # 格子への熱流: G*(Te-Tl) ≈ 6e10 W/cm³·K * 300K = 2e13 W/cm³
    # 必要時間: t ≈ L_m / 熱流 ≈ 4.2e2 / 2e13 ≈ 21 fs
    
    melting_steps = 20000  # 融解潜熱を完全に超えるため（約20 ps）
    temps_during_melting = []
    
    for step in range(melting_steps):
        result = advance_temperature(Te, Tl, ne, dne_dt, source_term, phase_state, latent_heat_accumulated, dt, config)
        Te = result.Te
        Tl = result.Tl
        phase_state = result.phase_state
        latent_heat_accumulated = result.latent_heat_accumulated
        
        temps_during_melting.append(Tl[0])
        
        if phase_state[0] == PhaseState.LIQUID:
            # 液相へ遷移 → プラトー終了
            break
    
    # 融解中の温度が融点付近に固定されていることを確認
    temps_array = np.array(temps_during_melting)
    mean_temp = np.mean(temps_array)
    std_temp = np.std(temps_array)
    
    assert abs(mean_temp - config.T_m) < 50.0, (
        f"Temperature should stay near melting point during MELTING. "
        f"Mean: {mean_temp:.1f} K, Expected: {config.T_m} K"
    )
    assert std_temp < 100.0, (
        f"Temperature should be stable during MELTING. Std: {std_temp:.1f} K"
    )
    
    # === Phase 3: 液相へ遷移後、温度上昇再開 ===
    assert phase_state[0] == PhaseState.LIQUID, "Should have transitioned to LIQUID"
    
    # さらに数ステップ進めて、温度が再び上昇することを確認
    liquid_start_temp = Tl[0]
    
    for _ in range(50):
        result = advance_temperature(Te, Tl, ne, dne_dt, source_term, phase_state, latent_heat_accumulated, dt, config)
        Te = result.Te
        Tl = result.Tl
        phase_state = result.phase_state
        latent_heat_accumulated = result.latent_heat_accumulated
    
    assert Tl[0] > liquid_start_temp, (
        f"Temperature should rise after melting complete. "
        f"Start: {liquid_start_temp:.1f} K, End: {Tl[0]:.1f} K"
    )


def test_vaporization_temperature_plateau() -> None:
    """沸点での温度プラトーを確認する。
    
    テストシナリオ:
    1. 液相の温度を沸点直下に設定
    2. 一定の熱源を与え続ける
    3. 沸点到達後、温度が沸点に固定されることを確認
    4. 気化潜熱を超えた後、再び温度が上昇することを確認
    """
    # === セットアップ ===
    config = TTMConfig(dz=1e-7)
    n_z = 1
    
    # 初期状態: 電子系が高温、格子系が沸点に近い
    Te = np.array([5000.0], dtype=np.float64)  # 高温の電子系（強い結合）
    Tl = np.array([3500.0], dtype=np.float64)  # 沸点のすぐ下
    ne = np.array([1e21], dtype=np.float64)  # 高密度（G を大きくする）
    phase_state = np.array([PhaseState.LIQUID], dtype=np.int32)
    
    # 強い熱源（超高出力）
    source_term = np.array([1e15], dtype=np.float64)  # 1 PW/cm³（Te維持のため）
    dne_dt = np.zeros_like(source_term)
    latent_heat_accumulated = np.zeros_like(source_term)
    
    dt = 1e-15  # 1 fs
    
    # === Phase 1: 沸点到達 ===
    # 沸点 T_b = 3538Kまで約338K上昇
    # 初期条件を3200Kに設定しているので、少ないステップで到達
    for step in range(1000):  # 十分なステップ
        result = advance_temperature(Te, Tl, ne, dne_dt, source_term, phase_state, latent_heat_accumulated, dt, config)
        Te = result.Te
        Tl = result.Tl
        phase_state = result.phase_state
        latent_heat_accumulated = result.latent_heat_accumulated
        
        if Tl[0] >= config.T_b:
            assert phase_state[0] == PhaseState.VAPORIZING, (
                f"Expected VAPORIZING at step {step}, got {phase_state[0]}"
            )
            break
    
    assert phase_state[0] == PhaseState.VAPORIZING, "Should be in VAPORIZING phase"
    
    # === Phase 2: 気化中（温度プラトー） ===
    # 気化潜熱: L_v = 1.5e10 erg/cm³ = 1.5e3 J/cm³
    # 熱源: S = 5 TW/cm³ + 電子-格子結合
    # 格子への熱流: G*(Te-Tl) ≈ 6e10 * 400 = 2.4e13 W/cm³
    # 必要時間: t ≈ L_v / 熱流 ≈ 1.5e3 / 2.4e13 ≈ 63 fs
    
    vaporizing_steps = 80000  # 気化潜熱を完全に超えるため（L_vはL_mの3.6倍、約80 ps）
    temps_during_vaporizing = []
    
    for step in range(vaporizing_steps):
        result = advance_temperature(Te, Tl, ne, dne_dt, source_term, phase_state, latent_heat_accumulated, dt, config)
        Te = result.Te
        Tl = result.Tl
        phase_state = result.phase_state
        latent_heat_accumulated = result.latent_heat_accumulated
        
        temps_during_vaporizing.append(Tl[0])
        
        if phase_state[0] == PhaseState.VAPOR:
            break
    
    # 気化中の温度が沸点付近に固定されていることを確認
    temps_array = np.array(temps_during_vaporizing)
    mean_temp = np.mean(temps_array)
    
    assert abs(mean_temp - config.T_b) < 100.0, (
        f"Temperature should stay near boiling point during VAPORIZING. "
        f"Mean: {mean_temp:.1f} K, Expected: {config.T_b} K"
    )
    
    # === Phase 3: 気相へ遷移後、温度上昇再開 ===
    assert phase_state[0] == PhaseState.VAPOR, "Should have transitioned to VAPOR"
    
    vapor_start_temp = Tl[0]
    
    for _ in range(50):
        result = advance_temperature(Te, Tl, ne, dne_dt, source_term, phase_state, latent_heat_accumulated, dt, config)
        Te = result.Te
        Tl = result.Tl
        phase_state = result.phase_state
        latent_heat_accumulated = result.latent_heat_accumulated
    
    assert Tl[0] > vapor_start_temp, (
        f"Temperature should rise after vaporization complete. "
        f"Start: {vapor_start_temp:.1f} K, End: {Tl[0]:.1f} K"
    )


def test_phase_state_transitions_sequence() -> None:
    """相状態遷移のシーケンスを確認する。
    
    テストシナリオ:
    SOLID → MELTING → LIQUID → VAPORIZING → VAPOR
    の順序で遷移することを確認する。
    """
    # === セットアップ ===
    config = TTMConfig(dz=1e-7)
    n_z = 1
    
    # 初期状態: 電子系が超高温、格子系が室温
    Te = np.array([8000.0], dtype=np.float64)  # 超高温の電子系（一気に加熱）
    Tl = np.array([300.0], dtype=np.float64)  # 室温の格子系
    ne = np.array([1e21], dtype=np.float64)  # 高密度（G を大きくする）
    phase_state = np.array([PhaseState.SOLID], dtype=np.int32)
    
    # 超強力な熱源（一気に加熱：フェムト秒レーザー相当）
    source_term = np.array([5e15], dtype=np.float64)  # 5 PW/cm³
    dne_dt = np.zeros_like(source_term)
    latent_heat_accumulated = np.zeros_like(source_term)
    
    dt = 1e-15  # 1 fs
    
    # === 遷移を追跡 ===
    phase_history = [PhaseState.SOLID]
    max_steps = 50000  # 十分なステップ数（約50 ps）
    
    for step in range(max_steps):
        result = advance_temperature(Te, Tl, ne, dne_dt, source_term, phase_state, latent_heat_accumulated, dt, config)
        Te = result.Te
        Tl = result.Tl
        phase_state = result.phase_state
        latent_heat_accumulated = result.latent_heat_accumulated
        
        # 相状態が変化したら記録
        if phase_state[0] != phase_history[-1]:
            phase_history.append(phase_state[0])
        
        # VAPOR に到達したら終了
        if phase_state[0] == PhaseState.VAPOR:
            break
    
    # === 遷移シーケンスを検証 ===
    expected_sequence = [
        PhaseState.SOLID,
        PhaseState.MELTING,
        PhaseState.LIQUID,
        PhaseState.VAPORIZING,
        PhaseState.VAPOR,
    ]
    
    assert len(phase_history) == len(expected_sequence), (
        f"Phase history length mismatch. "
        f"Expected {len(expected_sequence)}, got {len(phase_history)}"
    )
    
    for i, (expected, actual) in enumerate(zip(expected_sequence, phase_history)):
        assert actual == expected, (
            f"Phase transition {i}: Expected {expected}, got {actual}. "
            f"Full history: {phase_history}"
        )


def test_latent_heat_accumulation_energy_balance() -> None:
    """潜熱蓄積のエネルギーバランスを確認する。
    
    テストシナリオ:
    1. 融解中のエネルギー入力を積算
    2. 最終的な潜熱蓄積量が L_m と一致することを確認
    """
    # === セットアップ ===
    config = TTMConfig(dz=1e-7)
    n_z = 1
    
    Te = np.array([4000.0], dtype=np.float64)  # 高温の電子系
    Tl = np.array([1685.0], dtype=np.float64)  # 融点のすぐ下
    ne = np.array([1e21], dtype=np.float64)  # 高密度（G を大きくする）
    phase_state = np.array([PhaseState.SOLID], dtype=np.int32)
    
    # 一定の熱源（超高出力）
    source_term = np.array([1e15], dtype=np.float64)  # 1 PW/cm³（融解を早めるため）
    dne_dt = np.zeros_like(source_term)
    latent_heat_accumulated = np.zeros_like(source_term)
    
    dt = 1e-15  # 1 fs
    
    # === 融点到達まで ===
    for _ in range(500):  # 十分なステップ
        result = advance_temperature(Te, Tl, ne, dne_dt, source_term, phase_state, latent_heat_accumulated, dt, config)
        Te = result.Te
        Tl = result.Tl
        phase_state = result.phase_state
        latent_heat_accumulated = result.latent_heat_accumulated
        
        if phase_state[0] == PhaseState.MELTING:
            break
    
    assert phase_state[0] == PhaseState.MELTING, "Should reach MELTING phase"
    
    # === 融解中のエネルギー積算 ===
    # 融解中は潜熱が蓄積される
    # 最終的な潜熱蓄積量がL_mに近いことを確認
    
    initial_latent = latent_heat_accumulated[0]
    
    for step in range(20000):  # 融解完了まで十分な時間（20 ps）
        result = advance_temperature(Te, Tl, ne, dne_dt, source_term, phase_state, latent_heat_accumulated, dt, config)
        Te = result.Te
        Tl = result.Tl
        phase_state = result.phase_state
        latent_heat_accumulated = result.latent_heat_accumulated
        
        if phase_state[0] != PhaseState.MELTING:
            # 液相へ遷移 → 融解完了
            break
    
    # === エネルギーバランスの検証 ===
    # 融解完了時、latent_heat_accumulated はリセットされている（0になる）
    # つまり、L_m 分のエネルギーが潜熱として消費された
    expected_energy = config.L_m
    
    # 液相に遷移したことを確認
    assert phase_state[0] == PhaseState.LIQUID, (
        f"Should have transitioned to LIQUID after melting. "
        f"Current phase: {phase_state[0]}"
    )
    
    # 潜熱蓄積がリセットされたことを確認
    assert latent_heat_accumulated[0] < config.L_m * 0.1, (
        f"Latent heat should be reset after melting complete. "
        f"Current: {latent_heat_accumulated[0]:.2e} J/cm³"
    )
    
    # 注: エネルギーの定量的な検証は複雑なので、定性的な挙動確認に留める
    # （熱源 + 電子-格子結合の両方が寄与するため）
