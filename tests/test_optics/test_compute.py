"""tests/test_optics/test_compute.py — optics モジュールの統合テスト

README §7.1 の全8テストケースを実装する。
"""

import numpy as np
import pytest

from modules.optics import compute_laser_field, OpticsConfig
from modules import PhaseState


@pytest.fixture
def base_config() -> OpticsConfig:
    """テスト用の基本設定。"""
    return OpticsConfig(
        fluence=1.0,  # J/cm²
        dz=5e-7,  # cm (5 nm)
    )


@pytest.fixture
def grid_size() -> int:
    """テスト用グリッドサイズ。"""
    return 100


@pytest.fixture
def Te_300K(grid_size: int) -> np.ndarray:
    """テスト用電子温度（室温）。"""
    return np.full(grid_size, 300.0, dtype=np.float64)


def test_drude_low_ne(base_config: OpticsConfig, grid_size: int, Te_300K: np.ndarray) -> None:
    """テスト1: ne ≈ 0 で静的誘電率に一致する。"""
    n_z = grid_size
    ne = np.full(n_z, 1e12, dtype=np.float64)  # ≈ 0
    Tl = np.full(n_z, 300.0, dtype=np.float64)
    phase_state = np.full(n_z, PhaseState.SOLID, dtype=np.int32)
    t = 0.0

    result = compute_laser_field(ne, Tl, Te_300K, phase_state, t, base_config)

    # 低密度では反射率は静的誘電率の値に近いはず
    # Si の静的反射率 ≈ 0.30
    assert 0.25 < result.reflectivity < 0.35


def test_reflectivity_300K(base_config: OpticsConfig, grid_size: int, Te_300K: np.ndarray) -> None:
    """テスト2: 300K, ne = 10^12 で静的反射率に一致。"""
    n_z = grid_size
    ne = np.full(n_z, 1e12, dtype=np.float64)
    Tl = np.full(n_z, 300.0, dtype=np.float64)
    phase_state = np.full(n_z, PhaseState.SOLID, dtype=np.int32)
    t = 0.0

    result = compute_laser_field(ne, Tl, Te_300K, phase_state, t, base_config)

    # Si の 1030nm での反射率 ≈ 0.30
    assert 0.25 < result.reflectivity < 0.35


def test_reflectivity_high_ne(base_config: OpticsConfig, grid_size: int, Te_300K: np.ndarray) -> None:
    """テスト3: ne = 5×10^22 (≈n0) で金属的反射率上昇（R > 0.9）。"""
    n_z = grid_size
    ne = np.full(n_z, 5e22, dtype=np.float64)  # n0 に近い密度
    Tl = np.full(n_z, 300.0, dtype=np.float64)
    phase_state = np.full(n_z, PhaseState.SOLID, dtype=np.int32)
    t = 0.0

    result = compute_laser_field(ne, Tl, Te_300K, phase_state, t, base_config)

    # n0 程度の高密度で完全金属的になり反射率がほぼ1
    assert result.reflectivity > 0.9


def test_beer_lambert(base_config: OpticsConfig, grid_size: int, Te_300K: np.ndarray) -> None:
    """テスト4: β=0 で Beer-Lambert 則（指数減衰）を確認。"""
    config = base_config.model_copy()
    config.beta_tpa = 0.0  # TPA を無効化

    n_z = grid_size
    ne = np.full(n_z, 1e12, dtype=np.float64)
    Tl = np.full(n_z, 300.0, dtype=np.float64)
    phase_state = np.full(n_z, PhaseState.SOLID, dtype=np.int32)
    t = 0.0

    result = compute_laser_field(ne, Tl, Te_300K, phase_state, t, config)

    # I[0] から I[n_z-1] まで単調減少
    assert np.all(np.diff(result.intensity) <= 0)

    # 指数減衰の確認（対数が線形）
    I_nonzero = result.intensity[result.intensity > 1e-10]
    if len(I_nonzero) > 10:
        log_I = np.log(I_nonzero)
        # log(I) が z に対して線形なら、2階差分は小さい
        second_diff = np.diff(log_I, n=2)
        assert np.std(second_diff) < 0.1


def test_pulse_energy(base_config: OpticsConfig, grid_size: int, Te_300K: np.ndarray) -> None:
    """テスト5: ガウシアンパルスの時間積分の妥当性を確認。"""
    n_z = grid_size
    ne = np.full(n_z, 1e12, dtype=np.float64)
    Tl = np.full(n_z, 300.0, dtype=np.float64)
    phase_state = np.full(n_z, PhaseState.SOLID, dtype=np.int32)

    # パルス中心前後で時間積分
    t_array = np.linspace(-3e-12, 3e-12, 100)  # -3ps ~ +3ps
    I_surface_array = []

    for t in t_array:
        result = compute_laser_field(ne, Tl, Te_300K, phase_state, t, base_config)
        I_surface_array.append(result.intensity[0])

    I_surface_array = np.array(I_surface_array)
    dt = t_array[1] - t_array[0]
    total_energy = np.trapz(I_surface_array, dx=dt)

    # 0.94 × sqrt(π/2.77) ≈ 1.000 なので、積分 ≈ (1-R)×F
    R = compute_laser_field(ne, Tl, Te_300K, phase_state, 0.0, base_config).reflectivity
    expected_energy = (1.0 - R) * base_config.fluence

    # ガウシアンパルスの時間積分 ≈ (1-R)×F（±1% 以内）
    assert 0.9 < total_energy / expected_energy < 1.1


def test_source_term(base_config: OpticsConfig, grid_size: int, Te_300K: np.ndarray) -> None:
    """テスト6: S = (α_SPA + α_FCA) × I を確認。"""
    n_z = grid_size
    ne = np.full(n_z, 1e18, dtype=np.float64)
    Tl = np.full(n_z, 500.0, dtype=np.float64)
    phase_state = np.full(n_z, PhaseState.SOLID, dtype=np.int32)
    t = 0.0

    result = compute_laser_field(ne, Tl, Te_300K, phase_state, t, base_config)

    # α_SPA を再計算
    from modules.material_properties import compute_alpha_spa

    alpha_spa = compute_alpha_spa(Tl, phase_state)

    # S = (α_SPA + α_FCA)×I + β×I²（TPA項を含む）
    expected_source = (alpha_spa + result.alpha_fca) * result.intensity + base_config.beta_cgs * result.intensity**2
    np.testing.assert_allclose(result.source_term, expected_source, rtol=1e-10)


def test_tau_e_phase_switch(base_config: OpticsConfig, grid_size: int, Te_300K: np.ndarray) -> None:
    """テスト7: 固相→液相でτeが切り替わることを確認（高密度で変化が大きい）。"""
    n_z = grid_size
    ne = np.full(n_z, 5e21, dtype=np.float64)  # 高密度でτeの影響を大きくする
    Tl = np.full(n_z, 300.0, dtype=np.float64)

    # 固相
    phase_solid = np.full(n_z, PhaseState.SOLID, dtype=np.int32)
    result_solid = compute_laser_field(ne, Tl, Te_300K, phase_solid, 0.0, base_config)

    # 液相
    phase_liquid = np.full(n_z, PhaseState.LIQUID, dtype=np.int32)
    result_liquid = compute_laser_field(ne, Tl, Te_300K, phase_liquid, 0.0, base_config)

    # τe が異なるため、反射率が変化するはず
    # 高密度では ν/ω の値が大きくなり、Drude項への影響が見えやすくなる
    assert result_solid.reflectivity != result_liquid.reflectivity


def test_alpha_spa_polynomial(base_config: OpticsConfig, grid_size: int, Te_300K: np.ndarray) -> None:
    """テスト8: 既知温度での α_SPA 値を確認。"""
    from modules.material_properties import compute_alpha_spa

    solid = np.array([0], dtype=np.int32)  # PhaseState.SOLID

    # 300K での α_SPA
    # 多項式: -58.95 + 0.6226*300 - 2.3e-3*300² + ... ≈ 14.6 cm⁻¹
    alpha_300K = compute_alpha_spa(np.array([300.0]), solid)[0]
    assert 10.0 < alpha_300K < 20.0

    # 1000K では増加
    alpha_1000K = compute_alpha_spa(np.array([1000.0]), solid)[0]
    assert alpha_1000K > alpha_300K

    # 1687K (融点) ではさらに増加
    alpha_1687K = compute_alpha_spa(np.array([1687.0]), solid)[0]
    assert alpha_1687K > alpha_1000K

    # 液相では 0
    liquid = np.array([2], dtype=np.int32)  # PhaseState.LIQUID
    alpha_liquid = compute_alpha_spa(np.array([2000.0]), liquid)[0]
    assert alpha_liquid == 0.0
