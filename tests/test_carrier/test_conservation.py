"""tests/test_carrier/test_conservation.py — 保存則テスト（Conservation Test）

拡散項のみを動かし、キャリア密度の総量が保存されることを検証する。
レーザー入力・オージェ再結合・衝突電離をすべて0にして、
両極性拡散による空間再分配だけが起きる状態を作る。
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from modules import PhaseState
from modules.carrier import CarrierConfig, advance_carrier_density


@pytest.fixture
def diffusion_only_config() -> CarrierConfig:
    """拡散項のみのテスト用設定。
    
    生成・消滅項をすべて無効化:
    - SPA: alpha_spa ≈ 0 (室温で低値)
    - TPA: beta = 0
    - Auger: C_A = 0
    - Impact: 無視（Te = 300K で theta ≈ 0）
    """
    return CarrierConfig(
        dz=5e-7,  # 5 nm in cm
        beta_cgs=0.0,  # TPAを無効化
        C_A=0.0,  # オージェを無効化
    )


def test_diffusion_preserves_total_carrier_count(
    diffusion_only_config: CarrierConfig,
) -> None:
    """拡散項のみを動かして、キャリア総量が保存されることを確認する。
    
    テストシナリオ:
    1. 空間の中央（z=50）にキャリアを集中配置
    2. 他の点は ne = 0
    3. 拡散のみで時間発展させる（100ステップ）
    4. 各ステップで総キャリア量が初期値と一致することを確認
    
    物理的意味:
    - 境界条件が正しく実装されていれば、キャリアは空間から漏れない
    - 差分式のインデックスがズレていれば、総和が変化する
    """
    # === セットアップ ===
    n_z = 100
    ne = np.zeros(n_z, dtype=np.float64)
    ne[50] = 1.0e20  # 中央にキャリアを配置 [cm⁻³]
    
    intensity = np.zeros(n_z, dtype=np.float64)  # レーザーなし
    Te = np.full(n_z, 300.0, dtype=np.float64)  # 室温（impact ≈ 0）
    Tl = np.full(n_z, 300.0, dtype=np.float64)
    phase_state = np.full(n_z, PhaseState.SOLID, dtype=np.int32)
    
    dt = 1e-15  # 1 fs
    dz = diffusion_only_config.dz  # cm
    
    # === 初期総量を計算 ===
    # 空間積分: ∫ ne dz ≈ Σ ne[i] × dz
    initial_total = np.sum(ne) * dz  # [cm⁻²]
    
    # === 時間発展（100ステップ） ===
    n_steps = 100
    for step in range(n_steps):
        result = advance_carrier_density(
            ne, intensity, Te, Tl, phase_state, dt, diffusion_only_config
        )
        ne = result.ne
        
        # 各ステップで総量が保存されているか検証
        current_total = np.sum(ne) * dz
        relative_error = abs(current_total - initial_total) / initial_total
        
        # 数値誤差を考慮して 0.1% 以内の一致を要求
        assert relative_error < 1e-3, (
            f"Step {step}: Total carrier count not conserved. "
            f"Initial: {initial_total:.6e}, Current: {current_total:.6e}, "
            f"Relative error: {relative_error:.2e}"
        )
    
    # === 最終結果の検証 ===
    final_total = np.sum(ne) * dz
    assert np.isclose(final_total, initial_total, rtol=1e-3), (
        f"Final total {final_total:.6e} differs from initial {initial_total:.6e}"
    )
    
    # キャリア分布が拡散していることを確認（中央から広がる）
    assert ne[50] < 1.0e20, "Central peak should decrease due to diffusion"
    assert ne[40] > 0, "Carriers should diffuse to neighboring cells"
    assert ne[60] > 0, "Carriers should diffuse to neighboring cells"


def test_diffusion_boundary_condition_no_flux(
    diffusion_only_config: CarrierConfig,
) -> None:
    """境界でのフラックスがゼロ（ノイマン境界条件）であることを確認する。
    
    テストシナリオ:
    1. 端点（z=0, z=n_z-1）にキャリアを配置
    2. 拡散を実行
    3. 端点のキャリア密度が減少しない（漏れない）ことを確認
    """
    # === セットアップ ===
    n_z = 50
    ne = np.zeros(n_z, dtype=np.float64)
    ne[0] = 5.0e19  # 左端にキャリア
    ne[-1] = 5.0e19  # 右端にキャリア
    
    intensity = np.zeros(n_z, dtype=np.float64)
    Te = np.full(n_z, 300.0, dtype=np.float64)
    Tl = np.full(n_z, 300.0, dtype=np.float64)
    phase_state = np.full(n_z, PhaseState.SOLID, dtype=np.int32)
    
    dt = 1e-15  # 1 fs
    dz = diffusion_only_config.dz
    
    initial_total = np.sum(ne) * dz
    
    # === 時間発展（50ステップ） ===
    for _ in range(50):
        result = advance_carrier_density(
            ne, intensity, Te, Tl, phase_state, dt, diffusion_only_config
        )
        ne = result.ne
    
    # === 総量が保存されていることを確認 ===
    final_total = np.sum(ne) * dz
    relative_error = abs(final_total - initial_total) / initial_total
    assert relative_error < 1e-3, (
        f"Boundary flux is not zero. "
        f"Initial: {initial_total:.6e}, Final: {final_total:.6e}, "
        f"Relative error: {relative_error:.2e}"
    )
    
    # 端点のキャリアが完全に消えていないことを確認（多少は残る）
    assert ne[0] > 0, "Carriers at left boundary should not disappear"
    assert ne[-1] > 0, "Carriers at right boundary should not disappear"


def test_diffusion_symmetry(diffusion_only_config: CarrierConfig) -> None:
    """拡散の対称性を確認する。
    
    テストシナリオ:
    1. 中央にキャリアを配置
    2. 拡散を実行
    3. 左右対称な分布になることを確認
    """
    # === セットアップ ===
    n_z = 101  # 奇数にして中央を明確に
    ne = np.zeros(n_z, dtype=np.float64)
    center = n_z // 2
    ne[center] = 1.0e20  # 中央にキャリア
    
    intensity = np.zeros(n_z, dtype=np.float64)
    Te = np.full(n_z, 300.0, dtype=np.float64)
    Tl = np.full(n_z, 300.0, dtype=np.float64)
    phase_state = np.full(n_z, PhaseState.SOLID, dtype=np.int32)
    
    dt = 1e-15  # 1 fs
    
    # === 時間発展（100ステップ） ===
    for _ in range(100):
        result = advance_carrier_density(
            ne, intensity, Te, Tl, phase_state, dt, diffusion_only_config
        )
        ne = result.ne
    
    # === 対称性の検証 ===
    # 中央から左右で対称な位置の値を比較
    for offset in range(1, center):
        left_idx = center - offset
        right_idx = center + offset
        relative_diff = abs(ne[left_idx] - ne[right_idx]) / (ne[left_idx] + 1e-30)
        
        assert relative_diff < 1e-2, (
            f"Asymmetry detected at offset {offset}: "
            f"ne[{left_idx}] = {ne[left_idx]:.6e}, "
            f"ne[{right_idx}] = {ne[right_idx]:.6e}"
        )
