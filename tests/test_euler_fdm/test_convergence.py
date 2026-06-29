"""tests/test_euler_fdm/test_convergence.py — 収束テスト（Grid Convergence Test）

空間・時間解像度の変化に対する差分スキームの収束性を検証する。
中心差分法（2階精度）の場合、Δz を半分にすると誤差が 1/4 になることを確認する。
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from modules.euler_fdm import EulerFDMConfig, GridConfig, TimeConfig, run_simulation
from modules.euler_fdm.public import SimulationResult


_DOMAIN_DEPTH_CM = 500e-7  # 500 nm
_T_END = 1e-12              # 1 ps
_DT_MAX = 1e-15             # 1 fs
_SNAP_STEPS = 1000          # 1 snapshot per 1 ps run (record at end)


def _make_config(n_z: int, dt_max: float = _DT_MAX) -> EulerFDMConfig:
    """テスト用設定を生成する。"""
    dz = _DOMAIN_DEPTH_CM / n_z
    return EulerFDMConfig(
        fluence=0.5,
        grid=GridConfig(n_z=n_z, dz=dz),
        time=TimeConfig(t_end=_T_END, dt_max=dt_max, snapshot_interval=_SNAP_STEPS),
    )


def test_spatial_convergence_order() -> None:
    """空間解像度を変えて、収束オーダーが O(Δz²) であることを確認する。

    テストシナリオ:
    1. n_z = 50, 100, 200 の3つの解像度で計算
    2. 各解像度での最終状態（Te, Tl, ne）を記録
    3. 粗い解像度から細かい解像度への誤差の減少率を計算
    4. 収束オーダーが 2 に近いことを確認

    数学的背景:
    - 中心差分法: O(Δz²)
    - dz を 1/2 にすると、誤差は 1/4 になる
    - 収束オーダー p = log(E1/E2) / log(2) ≈ 2
    """
    resolutions = [50, 100, 200]
    results: dict[int, SimulationResult] = {}

    for n_z in resolutions:
        results[n_z] = run_simulation(_make_config(n_z))

    # 基準解像度（最も細かい）のグリッド
    ref_n = 200
    ref_z = np.linspace(0, _DOMAIN_DEPTH_CM, ref_n)
    ref_Te = results[ref_n].Te_final
    ref_Tl = results[ref_n].Tl_final
    ref_ne = results[ref_n].ne_final

    errors: dict[int, float] = {}

    for n_z in [50, 100]:
        coarse_z = np.linspace(0, _DOMAIN_DEPTH_CM, n_z)
        Te_interp = np.interp(ref_z, coarse_z, results[n_z].Te_final)
        Tl_interp = np.interp(ref_z, coarse_z, results[n_z].Tl_final)
        ne_interp = np.interp(ref_z, coarse_z, results[n_z].ne_final)

        error_Te = np.linalg.norm(Te_interp - ref_Te) / np.linalg.norm(ref_Te)
        error_Tl = np.linalg.norm(Tl_interp - ref_Tl) / np.linalg.norm(ref_Tl)
        error_ne = np.linalg.norm(ne_interp - ref_ne) / np.linalg.norm(ref_ne)
        errors[n_z] = (error_Te + error_Tl + error_ne) / 3.0

    error_50 = errors[50]
    error_100 = errors[100]
    error_ratio = error_50 / error_100
    convergence_order = np.log(error_ratio) / np.log(2.0)

    assert convergence_order > 1.5, (
        f"Convergence order too low: {convergence_order:.2f}. "
        f"Error(50): {error_50:.2e}, Error(100): {error_100:.2e}"
    )
    assert error_ratio > 2.0, (
        f"Error reduction insufficient: {error_ratio:.2f}x. "
        f"Expected at least 2x reduction."
    )


@pytest.mark.slow
def test_temporal_convergence() -> None:
    """時間解像度を変えて、収束性を確認する。

    テストシナリオ:
    1. dt_max = 2 fs, 1 fs, 0.5 fs の3つで計算
    2. 最終状態の差を比較
    3. dt を小さくすると誤差が減少することを確認

    注意:
    - オイラー法は1階精度なので、収束オーダーは O(Δt)
    """
    dt_values = [2e-15, 1e-15, 5e-16]
    results: dict[float, SimulationResult] = {}

    for dt_max in dt_values:
        snap = max(1, int(_T_END / dt_max))
        config = EulerFDMConfig(
            fluence=0.5,
            grid=GridConfig(n_z=100, dz=5e-7),
            time=TimeConfig(t_end=_T_END, dt_max=dt_max, snapshot_interval=snap),
        )
        results[dt_max] = run_simulation(config)

    ref = results[5e-16]
    errors: dict[float, float] = {}

    for dt_max in [2e-15, 1e-15]:
        r = results[dt_max]
        error_Te = np.linalg.norm(r.Te_final - ref.Te_final) / np.linalg.norm(ref.Te_final)
        error_Tl = np.linalg.norm(r.Tl_final - ref.Tl_final) / np.linalg.norm(ref.Tl_final)
        error_ne = np.linalg.norm(r.ne_final - ref.ne_final) / np.linalg.norm(ref.ne_final)
        errors[dt_max] = (error_Te + error_Tl + error_ne) / 3.0

    error_ratio = errors[2e-15] / errors[1e-15]

    assert error_ratio > 1.5, (
        f"Temporal error reduction insufficient: {error_ratio:.2f}x. "
        f"Error(2fs): {errors[2e-15]:.2e}, Error(1fs): {errors[1e-15]:.2e}"
    )


def test_grid_independence_check() -> None:
    """グリッド非依存性のチェック（簡易版）。

    テストシナリオ:
    1. 2つの解像度（n_z = 100, 200）で計算
    2. 主要な積分量（総エネルギーなど）が近い値になることを確認

    物理的意味:
    - 解像度が十分に高ければ、結果は解像度に依存しない
    - 積分量は差分式の誤差が相殺されやすいため、良い指標
    """
    results: dict[int, SimulationResult] = {}

    for n_z in [100, 200]:
        results[n_z] = run_simulation(_make_config(n_z))

    dz_100 = _DOMAIN_DEPTH_CM / 100
    dz_200 = _DOMAIN_DEPTH_CM / 200

    total_ne_100 = np.sum(results[100].ne_final) * dz_100
    total_ne_200 = np.sum(results[200].ne_final) * dz_200

    avg_Te_100 = np.mean(results[100].Te_final)
    avg_Te_200 = np.mean(results[200].Te_final)

    avg_Tl_100 = np.mean(results[100].Tl_final)
    avg_Tl_200 = np.mean(results[200].Tl_final)

    depth_100 = results[100].ablation_depth_nm
    depth_200 = results[200].ablation_depth_nm

    ne_relative_diff = abs(total_ne_100 - total_ne_200) / (total_ne_200 + 1e-30)
    Te_relative_diff = abs(avg_Te_100 - avg_Te_200) / avg_Te_200
    Tl_relative_diff = abs(avg_Tl_100 - avg_Tl_200) / avg_Tl_200
    depth_relative_diff = abs(depth_100 - depth_200) / (depth_200 + 1e-10)

    assert ne_relative_diff < 0.1, (
        f"Total carrier count differs too much: {ne_relative_diff:.2%}"
    )
    assert Te_relative_diff < 0.1, (
        f"Average electron temperature differs too much: {Te_relative_diff:.2%}"
    )
    assert Tl_relative_diff < 0.1, (
        f"Average lattice temperature differs too much: {Tl_relative_diff:.2%}"
    )
    assert depth_relative_diff < 0.2, (
        f"Ablation depth differs too much: {depth_relative_diff:.2%}. "
        f"Depth(100): {depth_100:.2e} nm, Depth(200): {depth_200:.2e} nm"
    )


def test_slightly_larger_dt_stable() -> None:
    """dt_max を5倍にしても発散しないことを確認する（前進オイラーの安定限界チェック）。

    注意: ソルバーは適応的時間刻みを持たないため、dt_max はユーザー責任で設定する。
    デフォルト 1 fs の 5 倍（5 fs）ならば数値的に安定することを確認する。
    """
    config = EulerFDMConfig(
        fluence=0.5,
        grid=GridConfig(n_z=100, dz=5e-7),
        time=TimeConfig(t_end=_T_END, dt_max=5e-15, snapshot_interval=200),
    )

    result = run_simulation(config)

    assert np.all(np.isfinite(result.Te_final)), "Electron temperature contains NaN/Inf"
    assert np.all(np.isfinite(result.Tl_final)), "Lattice temperature contains NaN/Inf"
    assert np.all(np.isfinite(result.ne_final)), "Carrier density contains NaN/Inf"

    assert np.all(result.Te_final >= 0), "Negative electron temperature detected"
    assert np.all(result.Te_final <= 1e7), "Electron temperature above clip ceiling"
    assert np.all(result.Tl_final >= 0), "Negative lattice temperature detected"
    assert np.all(result.ne_final >= 0), "Negative carrier density detected"


@pytest.mark.slow
def test_convergence_with_increasing_resolution() -> None:
    """解像度を段階的に上げて、解が収束していくことを確認する。

    テストシナリオ:
    1. n_z = 25, 50, 100, 200 の4つで計算
    2. 表面格子温度の最終値を比較
    3. 解像度を上げるごとに、前の結果との差が小さくなることを確認
    """
    resolutions = [25, 50, 100, 200]
    surface_Tl: dict[int, float] = {}

    for n_z in resolutions:
        result = run_simulation(_make_config(n_z))
        surface_Tl[n_z] = float(result.Tl_final[0])

    changes = []
    prev = surface_Tl[25]

    for n_z in [50, 100, 200]:
        current = surface_Tl[n_z]
        changes.append(abs(current - prev) / prev)
        prev = current

    assert changes[1] < changes[0], (
        f"Change rate should decrease. Change(50→100): {changes[1]:.2%}, "
        f"Change(25→50): {changes[0]:.2%}"
    )
    assert changes[2] < 0.05, (
        f"Solution not converged. Final change rate: {changes[2]:.2%}"
    )
