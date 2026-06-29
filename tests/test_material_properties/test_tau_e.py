"""tests/test_material_properties/test_tau_e.py — compute_tau_e の単体テスト。

論文 Table 1 の経験式に基づく実装を検証する。
"""

import numpy as np
import pytest

from modules import PhaseState
from modules.material_properties.silicon import compute_tau_e


def _make_inputs(
    ne_values: list[float],
    phase: int,
) -> tuple[
    "np.ndarray[np.float64]",
    "np.ndarray[np.float64]",
    "np.ndarray[np.float64]",
    "np.ndarray[np.int32]",
]:
    """テスト用入力配列を生成する。"""
    ne = np.array(ne_values, dtype=np.float64)
    Te = np.ones_like(ne) * 300.0
    Tl = np.ones_like(ne) * 300.0
    phase_state = np.full(len(ne_values), phase, dtype=np.int32)
    return ne, Te, Tl, phase_state


# ── 固相テスト ──────────────────────────────────────────────────────────────

class TestSolidPhase:
    """固相 (SOLID) における τ_e の検証。"""

    NE_VALUES = [1e18, 1e20, 1e21, 1e22]
    # τ_e = 240e-15 * (1 + ne / 6e20)
    EXPECTED_FS = [
        240e-15 * (1.0 + 1e18 / 6.0e20),
        240e-15 * (1.0 + 1e20 / 6.0e20),
        240e-15 * (1.0 + 1e21 / 6.0e20),
        240e-15 * (1.0 + 1e22 / 6.0e20),
    ]

    def test_tau_e_solid_representative_values(self) -> None:
        """代表的なキャリア密度での τ_e が論文式に一致すること。

        期待値（固相）:
            ne = 1e18  -> 約240 fs
            ne = 1e20  -> 約280 fs
            ne = 1e21  -> 約640 fs
            ne = 1e22  -> 約4240 fs
        """
        ne, Te, Tl, phase_state = _make_inputs(self.NE_VALUES, PhaseState.SOLID)

        tau_e = compute_tau_e(ne, Te, Tl, phase_state)

        np.testing.assert_allclose(tau_e, self.EXPECTED_FS, rtol=1e-6)

    def test_tau_e_solid_approx_fs(self) -> None:
        """代表値の近似 fs 値が論文記載の概算と一致すること。"""
        ne, Te, Tl, phase_state = _make_inputs(self.NE_VALUES, PhaseState.SOLID)

        tau_e_fs = compute_tau_e(ne, Te, Tl, phase_state) * 1e15

        assert abs(tau_e_fs[0] - 240) < 5,   f"ne=1e18: expected ~240 fs, got {tau_e_fs[0]:.1f} fs"
        assert abs(tau_e_fs[1] - 280) < 5,   f"ne=1e20: expected ~280 fs, got {tau_e_fs[1]:.1f} fs"
        assert abs(tau_e_fs[2] - 640) < 10,  f"ne=1e21: expected ~640 fs, got {tau_e_fs[2]:.1f} fs"
        assert abs(tau_e_fs[3] - 4240) < 20, f"ne=1e22: expected ~4240 fs, got {tau_e_fs[3]:.1f} fs"

    def test_tau_e_melting_same_as_solid(self) -> None:
        """MELTING 相でも固相と同じ式が適用されること。"""
        ne_values = [1e20]
        ne_s, Te_s, Tl_s, ps_solid = _make_inputs(ne_values, PhaseState.SOLID)
        ne_m, Te_m, Tl_m, ps_melt = _make_inputs(ne_values, PhaseState.MELTING)

        tau_solid = compute_tau_e(ne_s, Te_s, Tl_s, ps_solid)
        tau_melt = compute_tau_e(ne_m, Te_m, Tl_m, ps_melt)

        np.testing.assert_allclose(tau_solid, tau_melt, rtol=1e-12)

    def test_tau_e_solid_increases_with_ne(self) -> None:
        """固相では ne が増大するほど τ_e が長くなること。"""
        ne, Te, Tl, phase_state = _make_inputs(self.NE_VALUES, PhaseState.SOLID)

        tau_e = compute_tau_e(ne, Te, Tl, phase_state)

        assert np.all(np.diff(tau_e) > 0), "τ_e は ne の増加関数であること"


# ── 液相・気相テスト ─────────────────────────────────────────────────────────

class TestLiquidVaporPhase:
    """液相・気相における τ_e の検証。"""

    NE_VALUES = [1e18, 1e20, 1e21, 1e22]
    EXPECTED_S = 1.0e-12
    EXPECTED_FS = 1000.0

    @pytest.mark.parametrize("phase", [PhaseState.LIQUID, PhaseState.VAPORIZING, PhaseState.VAPOR])
    def test_tau_e_liquid_is_1ps(self, phase: int) -> None:
        """液相・気相では ne によらず τ_e = 1 ps になること。"""
        ne, Te, Tl, phase_state = _make_inputs(self.NE_VALUES, phase)

        tau_e = compute_tau_e(ne, Te, Tl, phase_state)

        np.testing.assert_allclose(tau_e, self.EXPECTED_S, rtol=1e-12)

    @pytest.mark.parametrize("phase", [PhaseState.LIQUID, PhaseState.VAPORIZING, PhaseState.VAPOR])
    def test_tau_e_liquid_fs(self, phase: int) -> None:
        """液相では τ_e = 1000 fs であること。"""
        ne, Te, Tl, phase_state = _make_inputs(self.NE_VALUES, phase)

        tau_e_fs = compute_tau_e(ne, Te, Tl, phase_state) * 1e15

        np.testing.assert_allclose(tau_e_fs, self.EXPECTED_FS, rtol=1e-12)


# ── 混合相テスト ─────────────────────────────────────────────────────────────

class TestMixedPhase:
    """固相・液相が混在するグリッドでの検証。"""

    def test_tau_e_mixed_phase(self) -> None:
        """固相・液相が混在するとき、各セルが正しい式で計算されること。"""
        ne = np.array([1e20, 1e21, 1e20, 1e21], dtype=np.float64)
        Te = np.ones(4, dtype=np.float64) * 300.0
        Tl = np.ones(4, dtype=np.float64) * 300.0
        phase_state = np.array(
            [PhaseState.SOLID, PhaseState.SOLID, PhaseState.LIQUID, PhaseState.LIQUID],
            dtype=np.int32,
        )

        tau_e = compute_tau_e(ne, Te, Tl, phase_state)

        expected_solid_0 = 240e-15 * (1.0 + 1e20 / 6.0e20)
        expected_solid_1 = 240e-15 * (1.0 + 1e21 / 6.0e20)
        np.testing.assert_allclose(tau_e[0], expected_solid_0, rtol=1e-6)
        np.testing.assert_allclose(tau_e[1], expected_solid_1, rtol=1e-6)
        np.testing.assert_allclose(tau_e[2], 1.0e-12, rtol=1e-12)
        np.testing.assert_allclose(tau_e[3], 1.0e-12, rtol=1e-12)
