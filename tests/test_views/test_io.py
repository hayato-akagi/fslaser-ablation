"""tests/test_views/test_io.py — views/io.py のテスト

save/load のラウンドトリップとメタデータ検証を行う。
"""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from modules.euler_fdm.config import EulerFDMConfig, GridConfig, TimeConfig
from modules.euler_fdm.public import SimulationResult
from views.io import load_result, save_result


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


class TestSaveLoad:
    """save_result / load_result のテスト"""
    
    def test_save_creates_directory(self, dummy_result, dummy_config, temp_results_dir):
        """run ディレクトリが作成されることを確認"""
        run_dir = save_result(dummy_result, dummy_config, temp_results_dir)
        
        assert run_dir.exists()
        assert run_dir.is_dir()
        assert run_dir.parent == temp_results_dir
    
    def test_save_creates_files(self, dummy_result, dummy_config, temp_results_dir):
        """metadata.json と arrays.npz が作成されることを確認"""
        run_dir = save_result(dummy_result, dummy_config, temp_results_dir)
        
        metadata_path = run_dir / "metadata.json"
        arrays_path = run_dir / "arrays.npz"
        
        assert metadata_path.exists()
        assert arrays_path.exists()
    
    def test_run_id_format(self, dummy_result, dummy_config, temp_results_dir):
        """run_id のフォーマットが正しいことを確認"""
        run_dir = save_result(dummy_result, dummy_config, temp_results_dir)
        run_id = run_dir.name
        
        # {YYYYMMDD_HHMMSS}_F{fluence:.2f} 形式
        import re
        pattern = r"^\d{8}_\d{6}_F\d+\.\d{2}$"
        assert re.match(pattern, run_id), f"run_id '{run_id}' does not match expected format"
        
        # フルエンスが含まれていること
        assert "F1.50" in run_id
    
    def test_save_load_roundtrip(self, dummy_result, dummy_config, temp_results_dir):
        """保存→読込でデータが一致することを確認"""
        # 保存
        run_dir = save_result(dummy_result, dummy_config, temp_results_dir)
        
        # 読込
        loaded_result, loaded_metadata = load_result(run_dir)
        
        # 配列データの検証
        assert np.allclose(loaded_result.Te_final, dummy_result.Te_final)
        assert np.allclose(loaded_result.Tl_final, dummy_result.Tl_final)
        assert np.allclose(loaded_result.ne_final, dummy_result.ne_final)
        assert np.allclose(loaded_result.time_points, dummy_result.time_points)
        assert np.allclose(loaded_result.Te_surface_history, dummy_result.Te_surface_history)
        assert np.allclose(loaded_result.Tl_surface_history, dummy_result.Tl_surface_history)
        assert np.allclose(loaded_result.ne_surface_history, dummy_result.ne_surface_history)
        assert np.allclose(loaded_result.reflectivity_history, dummy_result.reflectivity_history)
        assert np.allclose(loaded_result.alpha_fca_surface_history, dummy_result.alpha_fca_surface_history)
        assert np.allclose(loaded_result.auger_term_surface_history, dummy_result.auger_term_surface_history)
        assert np.allclose(loaded_result.ablation_depth_history, dummy_result.ablation_depth_history)
        assert np.array_equal(loaded_result.ablated_mask, dummy_result.ablated_mask)
        
        # スカラーデータの検証
        assert loaded_result.ablation_depth_nm == pytest.approx(dummy_result.ablation_depth_nm)
        assert loaded_result.total_steps == dummy_result.total_steps
        assert loaded_result.fluence == pytest.approx(dummy_result.fluence)


class TestMetadata:
    """metadata.json の検証"""
    
    def test_metadata_fields(self, dummy_result, dummy_config, temp_results_dir):
        """metadata.json に必須フィールドが存在することを確認"""
        run_dir = save_result(dummy_result, dummy_config, temp_results_dir)
        
        metadata_path = run_dir / "metadata.json"
        with metadata_path.open("r") as f:
            metadata = json.load(f)
        
        # 必須フィールド
        assert "created_at" in metadata
        assert "fluence_J_cm2" in metadata
        assert "grid" in metadata
        assert "time" in metadata
        assert "ablation_depth_nm" in metadata
        
        # grid 内のフィールド
        assert "n_z" in metadata["grid"]
        assert "dz_cm" in metadata["grid"]
        
        # time 内のフィールド
        assert "t_start_s" in metadata["time"]
        assert "t_end_s" in metadata["time"]
        assert "dt_max_s" in metadata["time"]
        assert "total_steps" in metadata["time"]
    
    def test_metadata_values(self, dummy_result, dummy_config, temp_results_dir):
        """metadata.json の値が正しいことを確認"""
        run_dir = save_result(dummy_result, dummy_config, temp_results_dir)
        
        _, metadata = load_result(run_dir)
        
        assert metadata["fluence_J_cm2"] == pytest.approx(1.5)
        assert metadata["grid"]["n_z"] == 100
        assert metadata["grid"]["dz_cm"] == pytest.approx(5e-7)
        assert metadata["time"]["total_steps"] == 50000
        assert metadata["ablation_depth_nm"] == pytest.approx(50.0)


class TestArrays:
    """arrays.npz の検証"""
    
    def test_arrays_keys(self, dummy_result, dummy_config, temp_results_dir):
        """arrays.npz に必須キーが存在することを確認"""
        run_dir = save_result(dummy_result, dummy_config, temp_results_dir)
        
        arrays_path = run_dir / "arrays.npz"
        data = np.load(arrays_path)
        
        # 必須キー
        expected_keys = {
            "Te_final",
            "Tl_final",
            "ne_final",
            "ablated_mask",
            "time_points",
            "Te_surface_history",
            "Tl_surface_history",
            "ne_surface_history",
            "reflectivity_history",
            "alpha_fca_surface_history",
            "auger_term_surface_history",
            "ablation_depth_history",
        }
        
        actual_keys = set(data.keys())
        assert actual_keys == expected_keys
    
    def test_arrays_shapes(self, dummy_result, dummy_config, temp_results_dir):
        """arrays.npz の配列 shape が正しいことを確認"""
        run_dir = save_result(dummy_result, dummy_config, temp_results_dir)
        
        arrays_path = run_dir / "arrays.npz"
        data = np.load(arrays_path)
        
        n_z = 100
        n_snapshots = 50
        
        # 空間配列
        assert data["Te_final"].shape == (n_z,)
        assert data["Tl_final"].shape == (n_z,)
        assert data["ne_final"].shape == (n_z,)
        assert data["ablated_mask"].shape == (n_z,)
        
        # 時間配列
        assert data["time_points"].shape == (n_snapshots,)
        assert data["Te_surface_history"].shape == (n_snapshots,)
        assert data["Tl_surface_history"].shape == (n_snapshots,)
        assert data["ne_surface_history"].shape == (n_snapshots,)
        assert data["reflectivity_history"].shape == (n_snapshots,)
        assert data["alpha_fca_surface_history"].shape == (n_snapshots,)
        assert data["auger_term_surface_history"].shape == (n_snapshots,)
        assert data["ablation_depth_history"].shape == (n_snapshots,)
