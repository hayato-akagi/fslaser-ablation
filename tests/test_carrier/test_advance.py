"""tests/test_carrier/test_advance.py — Unit tests for carrier density evolution.

README §7.1 の全8テストケースを実装する。
"""

import numpy as np
import pytest

from modules import PhaseState
from modules.carrier import CarrierConfig, advance_carrier_density


@pytest.fixture
def base_config() -> CarrierConfig:
    """テスト用の基本設定。"""
    return CarrierConfig(dz=5e-7)  # 5 nm in cm


@pytest.fixture
def n_z() -> int:
    """グリッド点数。"""
    return 10


def test_spa_term(base_config: CarrierConfig, n_z: int) -> None:
    """SPA項が支配的な条件でキャリア増加を検証する。
    
    低強度でSPA項が支配的になることを確認する。
    """
    ne = np.zeros(n_z, dtype=np.float64)
    intensity = np.ones(n_z, dtype=np.float64) * 1e6  # 0.001 GW/cm² (低強度)
    Te = np.ones(n_z, dtype=np.float64) * 300.0
    Tl = np.ones(n_z, dtype=np.float64) * 300.0
    phase_state = np.full(n_z, PhaseState.SOLID, dtype=np.int32)
    dt = 1e-15  # 1 fs
    
    result = advance_carrier_density(ne, intensity, Te, Tl, phase_state, dt, base_config)
    
    alpha_spa_300K = (
        -58.95
        + 0.6226 * 300.0
        - 2.3e-3 * 300.0**2
        + 3.186e-6 * 300.0**3
        + 9.967e-10 * 300.0**4
        - 1.409e-13 * 300.0**5
    )
    hw_J = base_config.photon_energy_J
    spa_contribution = alpha_spa_300K * intensity[0] * dt / hw_J
    
    beta_cgs = base_config.beta_cgs
    tpa_contribution = beta_cgs * intensity[0]**2 * dt / (2.0 * hw_J)
    
    expected_delta_ne = spa_contribution + tpa_contribution
    actual_delta_ne = result.ne[0]
    
    assert actual_delta_ne > 0, "SPA項でキャリアが増加すること"
    assert spa_contribution > tpa_contribution * 10, "低強度でSPA項が支配的であること"
    np.testing.assert_allclose(actual_delta_ne, expected_delta_ne, rtol=1e-3)


def test_tpa_term(base_config: CarrierConfig, n_z: int) -> None:
    """TPA項が支配的な条件でキャリア増加を検証する。
    
    高強度でTPA項が支配的になることを確認する。
    """
    ne = np.zeros(n_z, dtype=np.float64)
    intensity = np.ones(n_z, dtype=np.float64) * 1e10  # 10 GW/cm² (非常に高強度)
    Te = np.ones(n_z, dtype=np.float64) * 300.0
    Tl = np.ones(n_z, dtype=np.float64) * 300.0
    phase_state = np.full(n_z, PhaseState.SOLID, dtype=np.int32)
    dt = 1e-15
    
    result = advance_carrier_density(ne, intensity, Te, Tl, phase_state, dt, base_config)
    
    alpha_spa_300K = (
        -58.95
        + 0.6226 * 300.0
        - 2.3e-3 * 300.0**2
        + 3.186e-6 * 300.0**3
        + 9.967e-10 * 300.0**4
        - 1.409e-13 * 300.0**5
    )
    hw_J = base_config.photon_energy_J
    spa_contribution = alpha_spa_300K * intensity[0] * dt / hw_J
    
    beta_cgs = base_config.beta_cgs
    tpa_contribution = beta_cgs * intensity[0]**2 * dt / (2.0 * hw_J)
    
    expected_delta_ne = spa_contribution + tpa_contribution
    actual_delta_ne = result.ne[0]
    
    assert actual_delta_ne > 0, "TPA項でキャリアが増加すること"
    assert tpa_contribution > spa_contribution, "高強度でTPA項が支配的であること"
    np.testing.assert_allclose(actual_delta_ne, expected_delta_ne, rtol=1e-3)


def test_auger_term(base_config: CarrierConfig, n_z: int) -> None:
    """Auger項のみでキャリア減少を検証する。
    
    Δn_e = -γ × n_e³ × Δt と一致することを確認する。
    """
    ne_init = 1e20  # cm⁻³
    ne = np.full(n_z, ne_init, dtype=np.float64)
    intensity = np.zeros(n_z, dtype=np.float64)  # I=0でSPA/TPAを無効化
    Te = np.ones(n_z, dtype=np.float64) * 300.0
    Tl = np.ones(n_z, dtype=np.float64) * 300.0
    phase_state = np.full(n_z, PhaseState.SOLID, dtype=np.int32)
    dt = 1e-15
    
    result = advance_carrier_density(ne, intensity, Te, Tl, phase_state, dt, base_config)
    
    gamma = base_config.gamma
    expected_delta_ne = -gamma * ne_init**3 * dt
    expected_ne_new = ne_init + expected_delta_ne
    
    actual_ne_new = result.ne[0]
    
    assert actual_ne_new < ne_init, "Auger項でキャリアが減少すること"
    np.testing.assert_allclose(actual_ne_new, expected_ne_new, rtol=1e-3)


def test_impact_term(base_config: CarrierConfig, n_z: int) -> None:
    """Impact項のみでキャリア増加を検証する。
    
    Δn_e = θ × n_e × Δt と一致することを確認する。
    """
    ne_init = 1e18  # cm⁻³
    ne = np.full(n_z, ne_init, dtype=np.float64)
    intensity = np.zeros(n_z, dtype=np.float64)
    Te = np.ones(n_z, dtype=np.float64) * 5000.0  # 高温で θ を有効化
    Tl = np.ones(n_z, dtype=np.float64) * 1000.0
    phase_state = np.full(n_z, PhaseState.SOLID, dtype=np.int32)
    dt = 1e-15
    
    result = advance_carrier_density(ne, intensity, Te, Tl, phase_state, dt, base_config)
    
    Eg = (
        1.16
        - 7.02e-4 * 1000.0**2 / (1000.0 + 1108.0)
        - 1.5e-8 * ne_init ** (1.0 / 3.0)
    )
    k_B_eV = base_config.k_B_eV
    theta = 3.6e10 * np.exp(-1.5 * Eg / (k_B_eV * 5000.0))
    
    expected_delta_ne = theta * ne_init * dt
    expected_ne_new = ne_init + expected_delta_ne
    
    actual_ne_new = result.ne[0]
    
    assert actual_ne_new > ne_init, "Impact項でキャリアが増加すること"
    np.testing.assert_allclose(actual_ne_new, expected_ne_new, rtol=1e-2)


def test_diffusion_conservation(base_config: CarrierConfig, n_z: int) -> None:
    """拡散項のみで総キャリア数が保存されることを検証する。
    
    断熱境界条件下では、∫n_e dz が保存される。
    """
    ne = np.zeros(n_z, dtype=np.float64)
    ne[n_z // 2] = 1e20  # 中央にピーク
    intensity = np.zeros(n_z, dtype=np.float64)
    Te = np.ones(n_z, dtype=np.float64) * 300.0
    Tl = np.ones(n_z, dtype=np.float64) * 300.0
    phase_state = np.full(n_z, PhaseState.SOLID, dtype=np.int32)
    dt = 1e-16  # 小さい時間刻み
    
    total_ne_before = np.sum(ne)
    
    result = advance_carrier_density(ne, intensity, Te, Tl, phase_state, dt, base_config)
    
    total_ne_after = np.sum(result.ne)
    
    np.testing.assert_allclose(total_ne_after, total_ne_before, rtol=1e-6)


def test_diffusion_gaussian(base_config: CarrierConfig, n_z: int) -> None:
    """初期ガウス分布の拡散で分散が増加することを検証する。
    
    分散の増加が 2 × D_0 × Δt に一致することを確認する（近似的）。
    """
    z = np.arange(n_z, dtype=np.float64) * base_config.dz
    z0 = z[n_z // 2]
    sigma0 = base_config.dz * 2.0
    ne = 1e19 * np.exp(-((z - z0) ** 2) / (2.0 * sigma0**2))
    
    intensity = np.zeros(n_z, dtype=np.float64)
    Te = np.ones(n_z, dtype=np.float64) * 300.0
    Tl = np.ones(n_z, dtype=np.float64) * 300.0
    phase_state = np.full(n_z, PhaseState.SOLID, dtype=np.int32)
    dt = 1e-14  # 10 fs
    
    var_before = np.sum((z - z0) ** 2 * ne) / np.sum(ne)
    
    result = advance_carrier_density(ne, intensity, Te, Tl, phase_state, dt, base_config)
    
    var_after = np.sum((z - z0) ** 2 * result.ne) / np.sum(result.ne)
    
    D0 = 18.0 * base_config.T_room / Tl[0]
    expected_var_increase = 2.0 * D0 * dt
    actual_var_increase = var_after - var_before
    
    assert actual_var_increase > 0, "拡散で分散が増加すること"
    np.testing.assert_allclose(actual_var_increase, expected_var_increase, rtol=0.5)


def test_eg_phase_switch(base_config: CarrierConfig, n_z: int) -> None:
    """固相→液相で E_g = 0 に切り替わることを検証する。
    
    θ（衝突電離係数）が液相で大幅に増加することを確認する。
    """
    ne_init = 1e18
    ne_solid = np.full(n_z, ne_init, dtype=np.float64)
    ne_liquid = np.full(n_z, ne_init, dtype=np.float64)
    
    intensity = np.zeros(n_z, dtype=np.float64)
    Te = np.ones(n_z, dtype=np.float64) * 5000.0
    Tl = np.ones(n_z, dtype=np.float64) * 2000.0
    
    phase_state_solid = np.full(n_z, PhaseState.SOLID, dtype=np.int32)
    phase_state_liquid = np.full(n_z, PhaseState.LIQUID, dtype=np.int32)
    
    dt = 1e-15
    
    result_solid = advance_carrier_density(
        ne_solid, intensity, Te, Tl, phase_state_solid, dt, base_config
    )
    result_liquid = advance_carrier_density(
        ne_liquid, intensity, Te, Tl, phase_state_liquid, dt, base_config
    )
    
    delta_ne_solid = result_solid.ne[0] - ne_init
    delta_ne_liquid = result_liquid.ne[0] - ne_init
    
    assert delta_ne_liquid > delta_ne_solid, "液相でθが大きくキャリアが増加すること"


def test_dne_dt_consistency(base_config: CarrierConfig, n_z: int) -> None:
    """dne_dt が (ne_new - ne_old) / dt と一致することを検証する。"""
    ne = np.ones(n_z, dtype=np.float64) * 1e18
    intensity = np.ones(n_z, dtype=np.float64) * 1e9
    Te = np.ones(n_z, dtype=np.float64) * 1000.0
    Tl = np.ones(n_z, dtype=np.float64) * 800.0
    phase_state = np.full(n_z, PhaseState.SOLID, dtype=np.int32)
    dt = 1e-15
    
    result = advance_carrier_density(ne, intensity, Te, Tl, phase_state, dt, base_config)
    
    expected_dne_dt = (result.ne - ne) / dt
    actual_dne_dt = result.dne_dt
    
    np.testing.assert_allclose(actual_dne_dt, expected_dne_dt, rtol=1e-6)
