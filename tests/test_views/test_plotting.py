"""tests/test_views/test_plotting.py — views/plotting.py のテスト

グラフファイル生成確認を行う。
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from modules.euler_fdm.config import EulerFDMConfig, GridConfig, TimeConfig
from modules.euler_fdm.public import SimulationResult
from views.io import save_result
from views.plotting import plot_fluence_comparison, plot_single_run


@pytest.fixture
def dummy_result() -> SimulationResult:
    """テスト用のダミー SimulationResult を生成する。"""
    n_z = 100
    n_snapshots = 50
    fluence = 1.5
    
    return SimulationResult(
        Te_final=np.full(n_z, 500.0),
        Tl_final=np.full(n_z, 400.0),
        ne_final=np.full(n_z, 1e18),
        ablation_depth_cm=5e-6,
        ablation_depth_nm=50.0,
        ablated_mask=np.zeros(n_z, dtype=bool),
        time_points=np.linspace(-1e-12, 5e-10, n_snapshots),
        Te_surface_history=np.linspace(300, 5000, n_snapshots),
        Tl_surface_history=np.linspace(300, 2000, n_snapshots),
        ne_surface_history=np.logspace(12, 22, n_snapshots),
        reflectivity_history=np.linspace(0.3, 0.8, n_snapshots),
        alpha_fca_surface_history=np.linspace(100, 1e5, n_snapshots),
        auger_term_surface_history=np.logspace(20, 30, n_snapshots),
        ablation_depth_history=np.linspace(0, 50, n_snapshots),
        total_steps=50000,
        fluence=fluence,
    )


def make_dummy_result(fluence: float) -> SimulationResult:
    """指定フルエンスのダミー SimulationResult を生成する。"""
    n_z = 100
    n_snapshots = 50
    
    return SimulationResult(
        Te_final=np.full(n_z, 500.0 * fluence),
        Tl_final=np.full(n_z, 400.0 * fluence),
        ne_final=np.full(n_z, 1e18 * fluence),
        ablation_depth_cm=5e-6 * fluence,
        ablation_depth_nm=50.0 * fluence,
        ablated_mask=np.zeros(n_z, dtype=bool),
        time_points=np.linspace(-1e-12, 5e-10, n_snapshots),
        Te_surface_history=np.linspace(300, 5000 * fluence, n_snapshots),
        Tl_surface_history=np.linspace(300, 2000 * fluence, n_snapshots),
        ne_surface_history=np.logspace(12, 22, n_snapshots) * fluence,
        reflectivity_history=np.linspace(0.3, 0.8, n_snapshots),
        alpha_fca_surface_history=np.linspace(100, 1e5 * fluence, n_snapshots),
        auger_term_surface_history=np.logspace(20, 30, n_snapshots) * fluence,
        ablation_depth_history=np.linspace(0, 50 * fluence, n_snapshots),
        total_steps=50000,
        fluence=fluence,
    )


@pytest.fixture
def dummy_config() -> EulerFDMConfig:
    """テスト用のダミー EulerFDMConfig を生成する。"""
    return EulerFDMConfig(
        fluence=1.5,
        grid=GridConfig(n_z=100, dz=5e-7),
        time=TimeConfig(t_end=5e-10, dt_max=1e-15),
    )


@pytest.fixture
def temp_results_dir():
    """テスト用の一時ディレクトリを作成する。"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # テスト後にクリーンアップ
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


class TestPlotSingleRun:
    """plot_single_run のテスト"""
    
    def test_plot_single_run_creates_plots_directory(self, dummy_result, dummy_config, temp_results_dir):
        """plots ディレクトリが作成されることを確認"""
        run_dir = save_result(dummy_result, dummy_config, temp_results_dir)
        plot_single_run(run_dir)
        
        plots_dir = run_dir / "plots"
        assert plots_dir.exists()
        assert plots_dir.is_dir()
    
    def test_plot_single_run_creates_8_graphs(self, dummy_result, dummy_config, temp_results_dir):
        """グラフ8枚が生成されることを確認"""
        run_dir = save_result(dummy_result, dummy_config, temp_results_dir)
        plot_single_run(run_dir)
        
        plots_dir = run_dir / "plots"
        
        expected_files = [
            "temperature_history.png",
            "carrier_density_history.png",
            "reflectivity_history.png",
            "alpha_fca_history.png",
            "Te_ne_dual.png",
            "auger_ne_history.png",
            "spatial_profiles.png",
            "summary.png",
        ]
        
        for filename in expected_files:
            filepath = plots_dir / filename
            assert filepath.exists(), f"{filename} was not created"
            # ファイルサイズが0でないことを確認
            assert filepath.stat().st_size > 0, f"{filename} is empty"


class TestPlotFluenceComparison:
    """plot_fluence_comparison のテスト"""
    
    def test_plot_fluence_comparison_creates_output_directory(self, temp_results_dir):
        """output_dir が作成されることを確認"""
        # 複数フルエンスの結果を保存
        fluences = [0.5, 1.0, 1.5, 2.0]
        run_dirs = []
        
        for F in fluences:
            result = make_dummy_result(F)
            config = EulerFDMConfig(
                fluence=F,
                grid=GridConfig(n_z=100, dz=5e-7),
                time=TimeConfig(t_end=5e-10, dt_max=1e-15),
            )
            run_dir = save_result(result, config, temp_results_dir)
            run_dirs.append(run_dir)
        
        # 比較グラフ生成
        output_dir = temp_results_dir / "fluence_scan"
        plot_fluence_comparison(run_dirs, output_dir)
        
        assert output_dir.exists()
        assert output_dir.is_dir()
    
    def test_plot_fluence_comparison_creates_6_graphs(self, temp_results_dir):
        """比較グラフ6枚が生成されることを確認"""
        # 複数フルエンスの結果を保存
        fluences = [0.5, 1.0, 1.5, 2.0]
        run_dirs = []
        
        for F in fluences:
            result = make_dummy_result(F)
            config = EulerFDMConfig(
                fluence=F,
                grid=GridConfig(n_z=100, dz=5e-7),
                time=TimeConfig(t_end=5e-10, dt_max=1e-15),
            )
            run_dir = save_result(result, config, temp_results_dir)
            run_dirs.append(run_dir)
        
        # 比較グラフ生成
        output_dir = temp_results_dir / "fluence_scan"
        plot_fluence_comparison(run_dirs, output_dir)
        
        expected_files = [
            "Tl_surface_compare.png",
            "Te_surface_compare.png",
            "reflectivity_compare.png",
            "alpha_fca_compare.png",
            "auger_ne_compare.png",
            "ablation_depth_vs_fluence.png",
        ]
        
        for filename in expected_files:
            filepath = output_dir / filename
            assert filepath.exists(), f"{filename} was not created"
            # ファイルサイズが0でないことを確認
            assert filepath.stat().st_size > 0, f"{filename} is empty"
    
    def test_plot_fluence_comparison_with_experimental_data(self, temp_results_dir):
        """実験データ付きの比較グラフが生成されることを確認"""
        # 複数フルエンスの結果を保存
        fluences = [0.5, 1.0, 1.5, 2.0]
        run_dirs = []
        
        for F in fluences:
            result = make_dummy_result(F)
            config = EulerFDMConfig(
                fluence=F,
                grid=GridConfig(n_z=100, dz=5e-7),
                time=TimeConfig(t_end=5e-10, dt_max=1e-15),
            )
            run_dir = save_result(result, config, temp_results_dir)
            run_dirs.append(run_dir)
        
        # 実験データ（ダミー）
        experimental_data = {
            0.5: 10.0,
            1.0: 30.0,
            1.5: 60.0,
            2.0: 100.0,
        }
        
        # 比較グラフ生成
        output_dir = temp_results_dir / "fluence_scan"
        plot_fluence_comparison(run_dirs, output_dir, experimental_data)
        
        # ablation_depth_vs_fluence.png が生成されていること
        filepath = output_dir / "ablation_depth_vs_fluence.png"
        assert filepath.exists()
        assert filepath.stat().st_size > 0
