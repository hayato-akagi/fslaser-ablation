"""views/io.py — SimulationResult の保存・読み込み

euler_fdm の SimulationResult を results/ ディレクトリに永続化し、
後から読み込んで可視化できるようにする。
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from modules.euler_fdm.config import EulerFDMConfig
from modules.euler_fdm.public import SimulationResult


def save_result(
    result: SimulationResult,
    config: EulerFDMConfig,
    output_dir: Path = Path("results"),
) -> Path:
    """SimulationResult を results/ に保存する。
    
    ディレクトリ構造:
        results/{YYYYMMDD_HHMMSS}_F{fluence:.2f}/
        ├── metadata.json   # 設定・メタ情報
        └── arrays.npz      # numpy 配列データ
    
    Args:
        result: シミュレーション結果
        config: シミュレーション設定
        output_dir: 保存先ベースディレクトリ（デフォルト: results/）
    
    Returns:
        作成された run ディレクトリの Path
    """
    # run_id ディレクトリ名を生成
    run_id = _create_run_id(result.fluence)
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # metadata.json を書き出し
    _save_metadata(run_dir, result, config)
    
    # arrays.npz を書き出し
    _save_arrays(run_dir, result)
    
    return run_dir


def load_result(run_dir: Path) -> tuple[SimulationResult, dict]:
    """保存済み結果を読み込む。
    
    Args:
        run_dir: run ディレクトリの Path
    
    Returns:
        (SimulationResult, metadata_dict) のタプル
    """
    # metadata.json を読み込み
    metadata = _load_metadata(run_dir)
    
    # arrays.npz を読み込み
    result = _load_arrays(run_dir, metadata)
    
    return result, metadata


def _create_run_id(fluence: float) -> str:
    """run_id ディレクトリ名を生成する。
    
    Args:
        fluence: フルエンス [J/cm²]
    
    Returns:
        {YYYYMMDD_HHMMSS}_F{fluence:.2f} 形式の文字列
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_F{fluence:.2f}"


def _save_metadata(run_dir: Path, result: SimulationResult, config: EulerFDMConfig) -> None:
    """metadata.json を書き出す。
    
    Args:
        run_dir: run ディレクトリ
        result: シミュレーション結果
        config: シミュレーション設定
    """
    metadata = {
        "created_at": datetime.now().isoformat(),
        "fluence_J_cm2": float(result.fluence),
        "grid": {
            "n_z": int(config.grid.n_z),
            "dz_cm": float(config.grid.dz),
        },
        "time": {
            "t_start_s": float(config.time.t_start),
            "t_end_s": float(config.time.t_end),
            "dt_max_s": float(config.time.dt_max),
            "total_steps": int(result.total_steps),
        },
        "ablation_depth_nm": float(result.ablation_depth_nm),
    }
    
    metadata_path = run_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def _save_arrays(run_dir: Path, result: SimulationResult) -> None:
    """arrays.npz を書き出す。
    
    Args:
        run_dir: run ディレクトリ
        result: シミュレーション結果
    """
    arrays_path = run_dir / "arrays.npz"
    
    np.savez_compressed(
        arrays_path,
        # 最終状態
        Te_final=result.Te_final,
        Tl_final=result.Tl_final,
        ne_final=result.ne_final,
        ablated_mask=result.ablated_mask,
        # 時間履歴
        time_points=result.time_points,
        Te_surface_history=result.Te_surface_history,
        Tl_surface_history=result.Tl_surface_history,
        ne_surface_history=result.ne_surface_history,
        reflectivity_history=result.reflectivity_history,
        alpha_fca_surface_history=result.alpha_fca_surface_history,
        auger_term_surface_history=result.auger_term_surface_history,
        ablation_depth_history=result.ablation_depth_history,
    )


def _load_metadata(run_dir: Path) -> dict:
    """metadata.json を読み込む。
    
    Args:
        run_dir: run ディレクトリ
    
    Returns:
        メタデータの辞書
    """
    metadata_path = run_dir / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_arrays(run_dir: Path, metadata: dict) -> SimulationResult:
    """arrays.npz を読み込んで SimulationResult を復元する。
    
    Args:
        run_dir: run ディレクトリ
        metadata: メタデータ辞書
    
    Returns:
        SimulationResult インスタンス
    """
    arrays_path = run_dir / "arrays.npz"
    data = np.load(arrays_path)
    
    # ablation_depth_cm は ablation_depth_nm から逆算
    ablation_depth_nm = metadata["ablation_depth_nm"]
    ablation_depth_cm = ablation_depth_nm * 1e-7
    
    return SimulationResult(
        # 最終状態
        Te_final=data["Te_final"],
        Tl_final=data["Tl_final"],
        ne_final=data["ne_final"],
        # アブレーション結果
        ablation_depth_cm=ablation_depth_cm,
        ablation_depth_nm=ablation_depth_nm,
        ablated_mask=data["ablated_mask"],
        # 時間履歴
        time_points=data["time_points"],
        Te_surface_history=data["Te_surface_history"],
        Tl_surface_history=data["Tl_surface_history"],
        ne_surface_history=data["ne_surface_history"],
        reflectivity_history=data["reflectivity_history"],
        alpha_fca_surface_history=data["alpha_fca_surface_history"],
        auger_term_surface_history=data["auger_term_surface_history"],
        ablation_depth_history=data["ablation_depth_history"],
        # メタデータ
        total_steps=metadata["time"]["total_steps"],
        fluence=metadata["fluence_J_cm2"],
    )
