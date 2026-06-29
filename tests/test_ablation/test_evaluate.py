"""tests/test_ablation/test_evaluate.py — evaluate_ablation のテスト"""

import numpy as np
import pytest

from modules.ablation import evaluate_ablation, AblationConfig


# フィクスチャ: デフォルト設定
@pytest.fixture
def default_config() -> AblationConfig:
    """デフォルトのアブレーション設定（threshold = 7132.5 K）。"""
    return AblationConfig()


@pytest.fixture
def default_dz() -> float:
    """デフォルトのグリッド間隔 5 nm in cm。"""
    return 5e-7  # cm


# ヘルパー関数
def make_Tl(
    n_z: int,
    hot_count: int,
    hot_temp: float = 8000.0,
    cold_temp: float = 3000.0,
) -> np.ndarray:
    """先頭 hot_count グリッドを hot_temp、残りを cold_temp にした配列を生成。"""
    Tl = np.full(n_z, cold_temp)
    Tl[:hot_count] = hot_temp
    return Tl


# テストケース1: 全グリッドが閾値未満
def test_no_ablation(default_config: AblationConfig, default_dz: float) -> None:
    """全グリッドが閾値未満の場合、アブレーション深さは0。"""
    n_z = 1000
    Tl = np.full(n_z, 5000.0)  # 閾値未満
    
    result = evaluate_ablation(Tl, default_dz, default_config)
    
    assert result.ablation_depth == 0.0
    assert np.all(~result.ablated_mask)  # 全て False


# テストケース2: 表面から3グリッドがアブレーション
def test_surface_ablation(default_config: AblationConfig, default_dz: float) -> None:
    """表面から3グリッドが閾値を超える場合、depth=3*dz。"""
    n_z = 1000
    Tl = make_Tl(n_z, hot_count=3, hot_temp=8000.0, cold_temp=3000.0)
    
    result = evaluate_ablation(Tl, default_dz, default_config)
    
    assert result.ablation_depth == 3 * default_dz
    assert np.sum(result.ablated_mask) == 3
    assert np.all(result.ablated_mask[:3])
    assert np.all(~result.ablated_mask[3:])


# テストケース3: 不連続な高温領域（連続性ルール）
def test_discontinuous(default_config: AblationConfig, default_dz: float) -> None:
    """中間に閾値未満があると、そこで打ち切られる。"""
    n_z = 1000
    Tl = np.full(n_z, 3000.0)
    Tl[0] = 8000.0
    Tl[1] = 8000.0
    Tl[2] = 5000.0  # 閾値未満
    Tl[3] = 8000.0  # 不連続なので除外
    
    result = evaluate_ablation(Tl, default_dz, default_config)
    
    assert result.ablation_depth == 2 * default_dz
    assert np.sum(result.ablated_mask) == 2
    assert result.ablated_mask[0]
    assert result.ablated_mask[1]
    assert not result.ablated_mask[2]
    assert not result.ablated_mask[3]


# テストケース4: 表面が閾値未満
def test_surface_below(default_config: AblationConfig, default_dz: float) -> None:
    """表面が閾値未満なら、深部が高温でもアブレーションなし。"""
    n_z = 1000
    Tl = np.full(n_z, 3000.0)
    Tl[0] = 5000.0  # 閾値未満
    Tl[1:4] = 8000.0  # 深部は高温
    
    result = evaluate_ablation(Tl, default_dz, default_config)
    
    assert result.ablation_depth == 0.0
    assert np.all(~result.ablated_mask)


# テストケース5: ちょうど閾値（>= 判定）
def test_exact_threshold(default_config: AblationConfig, default_dz: float) -> None:
    """温度がちょうど閾値の場合、閾値以上なのでアブレーション。"""
    n_z = 1000
    threshold = default_config.threshold_temperature  # 7132.5 K
    Tl = np.full(n_z, 3000.0)
    Tl[0] = threshold  # ちょうど閾値
    
    result = evaluate_ablation(Tl, default_dz, default_config)
    
    assert result.ablation_depth == default_dz
    assert np.sum(result.ablated_mask) == 1
    assert result.ablated_mask[0]


# テストケース6: 閾値未満（ぎりぎり）
def test_just_below(default_config: AblationConfig, default_dz: float) -> None:
    """閾値より僅かに低い温度では、アブレーションなし。"""
    n_z = 1000
    threshold = default_config.threshold_temperature  # 7132.5 K
    Tl = np.full(n_z, 3000.0)
    Tl[0] = threshold - 0.1  # わずかに閾値未満
    
    result = evaluate_ablation(Tl, default_dz, default_config)
    
    assert result.ablation_depth == 0.0
    assert np.all(~result.ablated_mask)


# テストケース7: 全グリッドがアブレーション
def test_all_ablated(default_config: AblationConfig, default_dz: float) -> None:
    """全グリッドが閾値を超える場合、全てアブレーション。"""
    n_z = 1000
    Tl = np.full(n_z, 8000.0)
    
    result = evaluate_ablation(Tl, default_dz, default_config)
    
    assert result.ablation_depth == n_z * default_dz
    assert np.all(result.ablated_mask)


# テストケース8: 異なるグリッドサイズ
def test_different_grid(default_config: AblationConfig) -> None:
    """グリッドサイズが異なっても正しく計算される。"""
    n_z = 500
    dz = 10e-7  # 10 nm in cm
    Tl = make_Tl(n_z, hot_count=10, hot_temp=8000.0, cold_temp=3000.0)
    
    result = evaluate_ablation(Tl, dz, default_config)
    
    assert result.ablation_depth == 10 * dz
    assert np.sum(result.ablated_mask) == 10


# テストケース9: ステートレス（同一入力で同一結果）
def test_stateless(default_config: AblationConfig, default_dz: float) -> None:
    """同じ入力で2回呼び出すと、同じ結果が得られる。"""
    n_z = 1000
    Tl = make_Tl(n_z, hot_count=5, hot_temp=8000.0, cold_temp=3000.0)
    
    result1 = evaluate_ablation(Tl, default_dz, default_config)
    result2 = evaluate_ablation(Tl, default_dz, default_config)
    
    assert result1.ablation_depth == result2.ablation_depth
    assert np.array_equal(result1.ablated_mask, result2.ablated_mask)


# テストケース10: 入力配列の不変性
def test_immutability(default_config: AblationConfig, default_dz: float) -> None:
    """evaluate_ablation は入力配列 Tl を変更しない。"""
    n_z = 1000
    Tl = make_Tl(n_z, hot_count=5, hot_temp=8000.0, cold_temp=3000.0)
    Tl_original = Tl.copy()
    
    result = evaluate_ablation(Tl, default_dz, default_config)
    
    assert np.array_equal(Tl, Tl_original)
