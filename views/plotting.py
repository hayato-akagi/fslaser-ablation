"""views/plotting.py — グラフ生成

matplotlib を用いて SimulationResult から14種類のグラフを生成する。
物理計算は行わず、データの可視化のみを担当する。
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from views.io import load_result

# ヘッドレス環境対応（テスト時など）
matplotlib.use("Agg")

# グラフ共通設定
DPI = 150
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 10
COLOR_TE = "red"
COLOR_TL = "blue"
COLOR_NE = "green"


def plot_single_run(run_dir: Path) -> None:
    """1回のシミュレーション結果からグラフ8枚を生成・保存する。
    
    生成グラフ:
        1. temperature_history.png
        2. carrier_density_history.png
        3. reflectivity_history.png
        4. alpha_fca_history.png
        5. Te_ne_dual.png
        6. auger_ne_history.png
        7. spatial_profiles.png
        8. summary.png
    
    Args:
        run_dir: run ディレクトリの Path
    """
    result, metadata = load_result(run_dir)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # 時間軸を ps に変換
    time_ps = _convert_time_to_ps(result.time_points)
    
    # グラフ1: 温度履歴
    _plot_temperature_history(time_ps, result, plots_dir)
    
    # グラフ2: キャリア密度履歴
    _plot_carrier_density_history(time_ps, result, plots_dir)
    
    # グラフ3: 反射率履歴
    _plot_reflectivity_history(time_ps, result, plots_dir)
    
    # グラフ4: α_FCA 履歴
    _plot_alpha_fca_history(time_ps, result, plots_dir)
    
    # グラフ5: Te + ne デュアルY軸
    _plot_te_ne_dual(time_ps, result, plots_dir)
    
    # グラフ6: γne³ + ne デュアルY軸
    _plot_auger_ne_history(time_ps, result, plots_dir)
    
    # グラフ7: 空間プロファイル
    _plot_spatial_profiles(result, metadata, plots_dir)
    
    # グラフ8: サマリー（2×2）
    _plot_summary(time_ps, result, metadata, plots_dir)


def plot_fluence_comparison(
    run_dirs: list[Path],
    output_dir: Path = Path("results/fluence_scan"),
    experimental_data: dict[float, float] | None = None,
) -> None:
    """複数 run の結果を重ね描きした比較グラフ6枚を生成・保存する。
    
    生成グラフ:
        9. Tl_surface_compare.png
        10. Te_surface_compare.png
        11. reflectivity_compare.png
        12. alpha_fca_compare.png
        13. auger_ne_compare.png
        14. ablation_depth_vs_fluence.png
    
    Args:
        run_dirs: 各フルエンスの run ディレクトリリスト
        output_dir: 比較グラフの保存先
        experimental_data: {fluence: depth_nm} の辞書（論文実験値）
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 全 run を読み込み
    runs = [load_result(d) for d in run_dirs]
    results = [r[0] for r in runs]
    fluences = [r.fluence for r in results]
    
    # グラフ9: Tl 比較
    _plot_tl_surface_compare(results, fluences, output_dir)
    
    # グラフ10: Te 比較
    _plot_te_surface_compare(results, fluences, output_dir)
    
    # グラフ11: 反射率比較
    _plot_reflectivity_compare(results, fluences, output_dir)
    
    # グラフ12: α_FCA 比較
    _plot_alpha_fca_compare(results, fluences, output_dir)
    
    # グラフ13: γne³ + ne 比較
    _plot_auger_ne_compare(results, fluences, output_dir)
    
    # グラフ14: アブレーション深さ vs フルエンス
    _plot_ablation_depth_vs_fluence(results, fluences, output_dir, experimental_data)


# ========== 単位変換ヘルパー ==========

def _convert_time_to_ps(time_s: np.ndarray) -> np.ndarray:
    """時間を s から ps に変換する。"""
    return time_s * 1e12


def _convert_depth_to_nm(depth_cm: np.ndarray) -> np.ndarray:
    """深さを cm から nm に変換する。"""
    return depth_cm * 1e7


# ========== 単一 run グラフ（#1〜#8） ==========

def _plot_temperature_history(time_ps: np.ndarray, result, plots_dir: Path) -> None:
    """グラフ#1: 温度履歴 Te(0,t), Tl(0,t)"""
    plt.figure(figsize=(10, 6))
    plt.plot(time_ps, result.Te_surface_history, COLOR_TE, label="$T_e$", linewidth=2)
    plt.plot(time_ps, result.Tl_surface_history, COLOR_TL, label="$T_l$", linewidth=2)
    plt.xlabel("Time [ps]", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Temperature [K]", fontsize=LABEL_FONTSIZE)
    plt.title("Surface Temperature History", fontsize=TITLE_FONTSIZE)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / "temperature_history.png", dpi=DPI)
    plt.close()


def _plot_carrier_density_history(time_ps: np.ndarray, result, plots_dir: Path) -> None:
    """グラフ#2: キャリア密度履歴 ne(0,t)（対数Y軸）"""
    plt.figure(figsize=(10, 6))
    plt.semilogy(time_ps, result.ne_surface_history, COLOR_NE, linewidth=2)
    plt.xlabel("Time [ps]", fontsize=LABEL_FONTSIZE)
    plt.ylabel("$n_e$ [cm$^{-3}$]", fontsize=LABEL_FONTSIZE)
    plt.title("Surface Carrier Density History", fontsize=TITLE_FONTSIZE)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / "carrier_density_history.png", dpi=DPI)
    plt.close()


def _plot_reflectivity_history(time_ps: np.ndarray, result, plots_dir: Path) -> None:
    """グラフ#3: 反射率履歴 R(t)"""
    plt.figure(figsize=(10, 6))
    plt.plot(time_ps, result.reflectivity_history, "purple", linewidth=2)
    plt.xlabel("Time [ps]", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Reflectivity R", fontsize=LABEL_FONTSIZE)
    plt.title("Surface Reflectivity History", fontsize=TITLE_FONTSIZE)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / "reflectivity_history.png", dpi=DPI)
    plt.close()


def _plot_alpha_fca_history(time_ps: np.ndarray, result, plots_dir: Path) -> None:
    """グラフ#4: α_FCA 履歴"""
    plt.figure(figsize=(10, 6))
    plt.plot(time_ps, result.alpha_fca_surface_history, "orange", linewidth=2)
    plt.xlabel("Time [ps]", fontsize=LABEL_FONTSIZE)
    plt.ylabel("$\\alpha_{FCA}$ [cm$^{-1}$]", fontsize=LABEL_FONTSIZE)
    plt.title("Surface FCA Absorption Coefficient History", fontsize=TITLE_FONTSIZE)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / "alpha_fca_history.png", dpi=DPI)
    plt.close()


def _plot_te_ne_dual(time_ps: np.ndarray, result, plots_dir: Path) -> None:
    """グラフ#5: Te + ne デュアルY軸"""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 左軸: Te
    ax1.plot(time_ps, result.Te_surface_history, COLOR_TE, label="$T_e$", linewidth=2)
    ax1.set_xlabel("Time [ps]", fontsize=LABEL_FONTSIZE)
    ax1.set_ylabel("$T_e$ [K]", fontsize=LABEL_FONTSIZE, color=COLOR_TE)
    ax1.tick_params(axis="y", labelcolor=COLOR_TE)
    ax1.grid(True)
    
    # 右軸: ne（対数）
    ax2 = ax1.twinx()
    ax2.semilogy(time_ps, result.ne_surface_history, COLOR_NE, label="$n_e$", linewidth=2, linestyle="--")
    ax2.set_ylabel("$n_e$ [cm$^{-3}$]", fontsize=LABEL_FONTSIZE, color=COLOR_NE)
    ax2.tick_params(axis="y", labelcolor=COLOR_NE)
    
    plt.title("$T_e$ and $n_e$ Dual-Axis", fontsize=TITLE_FONTSIZE)
    plt.tight_layout()
    plt.savefig(plots_dir / "Te_ne_dual.png", dpi=DPI)
    plt.close()


def _plot_auger_ne_history(time_ps: np.ndarray, result, plots_dir: Path) -> None:
    """グラフ#6: γne³ + ne デュアルY軸"""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 左軸: γne³
    ax1.semilogy(time_ps, result.auger_term_surface_history, "brown", label="$\\gamma n_e^3$", linewidth=2)
    ax1.set_xlabel("Time [ps]", fontsize=LABEL_FONTSIZE)
    ax1.set_ylabel("$\\gamma n_e^3$ [cm$^{-3}$ s$^{-1}$]", fontsize=LABEL_FONTSIZE)
    ax1.grid(True)
    
    # 右軸: ne（対数）
    ax2 = ax1.twinx()
    ax2.semilogy(time_ps, result.ne_surface_history, COLOR_NE, label="$n_e$", linewidth=2, linestyle="--")
    ax2.set_ylabel("$n_e$ [cm$^{-3}$]", fontsize=LABEL_FONTSIZE, color=COLOR_NE)
    ax2.tick_params(axis="y", labelcolor=COLOR_NE)
    
    plt.title("Auger Recombination and Carrier Density", fontsize=TITLE_FONTSIZE)
    plt.tight_layout()
    plt.savefig(plots_dir / "auger_ne_history.png", dpi=DPI)
    plt.close()


def _plot_spatial_profiles(result, metadata: dict, plots_dir: Path) -> None:
    """グラフ#7: 最終状態の空間プロファイル Te(z), Tl(z), ne(z)"""
    n_z = metadata["grid"]["n_z"]
    dz_cm = metadata["grid"]["dz_cm"]
    depth_nm = np.arange(n_z) * dz_cm * 1e7
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    
    # Te(z)
    ax1.plot(depth_nm, result.Te_final, COLOR_TE, linewidth=2)
    ax1.set_ylabel("$T_e$ [K]", fontsize=LABEL_FONTSIZE)
    ax1.set_title("Final Spatial Profiles", fontsize=TITLE_FONTSIZE)
    ax1.grid(True)
    
    # Tl(z)
    ax2.plot(depth_nm, result.Tl_final, COLOR_TL, linewidth=2)
    ax2.set_ylabel("$T_l$ [K]", fontsize=LABEL_FONTSIZE)
    ax2.grid(True)
    
    # ne(z)（対数）
    ax3.semilogy(depth_nm, result.ne_final, COLOR_NE, linewidth=2)
    ax3.set_xlabel("Depth [nm]", fontsize=LABEL_FONTSIZE)
    ax3.set_ylabel("$n_e$ [cm$^{-3}$]", fontsize=LABEL_FONTSIZE)
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "spatial_profiles.png", dpi=DPI)
    plt.close()


def _plot_summary(time_ps: np.ndarray, result, metadata: dict, plots_dir: Path) -> None:
    """グラフ#8: サマリー（#1, #2, #3, #7 を 2×2）"""
    fig = plt.figure(figsize=(16, 12))
    
    # サブプロット1: 温度履歴
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(time_ps, result.Te_surface_history, COLOR_TE, label="$T_e$", linewidth=1.5)
    ax1.plot(time_ps, result.Tl_surface_history, COLOR_TL, label="$T_l$", linewidth=1.5)
    ax1.set_xlabel("Time [ps]", fontsize=LABEL_FONTSIZE)
    ax1.set_ylabel("Temperature [K]", fontsize=LABEL_FONTSIZE)
    ax1.set_title("Temperature History", fontsize=TITLE_FONTSIZE)
    ax1.legend(fontsize=LEGEND_FONTSIZE)
    ax1.grid(True)
    
    # サブプロット2: キャリア密度履歴
    ax2 = plt.subplot(2, 2, 2)
    ax2.semilogy(time_ps, result.ne_surface_history, COLOR_NE, linewidth=1.5)
    ax2.set_xlabel("Time [ps]", fontsize=LABEL_FONTSIZE)
    ax2.set_ylabel("$n_e$ [cm$^{-3}$]", fontsize=LABEL_FONTSIZE)
    ax2.set_title("Carrier Density", fontsize=TITLE_FONTSIZE)
    ax2.grid(True)
    
    # サブプロット3: 反射率履歴
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(time_ps, result.reflectivity_history, "purple", linewidth=1.5)
    ax3.set_xlabel("Time [ps]", fontsize=LABEL_FONTSIZE)
    ax3.set_ylabel("Reflectivity R", fontsize=LABEL_FONTSIZE)
    ax3.set_title("Reflectivity", fontsize=TITLE_FONTSIZE)
    ax3.set_ylim(0, 1)
    ax3.grid(True)
    
    # サブプロット4: 空間プロファイル（簡略版）
    ax4 = plt.subplot(2, 2, 4)
    n_z = metadata["grid"]["n_z"]
    dz_cm = metadata["grid"]["dz_cm"]
    depth_nm = np.arange(n_z) * dz_cm * 1e7
    ax4.plot(depth_nm, result.Tl_final, COLOR_TL, label="$T_l$", linewidth=1.5)
    ax4_twin = ax4.twinx()
    ax4_twin.semilogy(depth_nm, result.ne_final, COLOR_NE, label="$n_e$", linewidth=1.5, linestyle="--", alpha=0.7)
    ax4.set_xlabel("Depth [nm]", fontsize=LABEL_FONTSIZE)
    ax4.set_ylabel("$T_l$ [K]", fontsize=LABEL_FONTSIZE, color=COLOR_TL)
    ax4_twin.set_ylabel("$n_e$ [cm$^{-3}$]", fontsize=LABEL_FONTSIZE, color=COLOR_NE)
    ax4.set_title("Final Profiles", fontsize=TITLE_FONTSIZE)
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "summary.png", dpi=DPI)
    plt.close()


# ========== 複数 run 比較グラフ（#9〜#14） ==========

def _plot_tl_surface_compare(results: list, fluences: list[float], output_dir: Path) -> None:
    """グラフ#9: Tl(0,t) 比較"""
    plt.figure(figsize=(10, 6))
    for result, F in zip(results, fluences):
        time_ps = _convert_time_to_ps(result.time_points)
        plt.plot(time_ps, result.Tl_surface_history, label=f"F = {F:.2f} J/cm²", linewidth=2)
    plt.xlabel("Time [ps]", fontsize=LABEL_FONTSIZE)
    plt.ylabel("$T_l$ [K]", fontsize=LABEL_FONTSIZE)
    plt.title("Surface Lattice Temperature Comparison", fontsize=TITLE_FONTSIZE)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "Tl_surface_compare.png", dpi=DPI)
    plt.close()


def _plot_te_surface_compare(results: list, fluences: list[float], output_dir: Path) -> None:
    """グラフ#10: Te(0,t) 比較"""
    plt.figure(figsize=(10, 6))
    for result, F in zip(results, fluences):
        time_ps = _convert_time_to_ps(result.time_points)
        plt.plot(time_ps, result.Te_surface_history, label=f"F = {F:.2f} J/cm²", linewidth=2)
    plt.xlabel("Time [ps]", fontsize=LABEL_FONTSIZE)
    plt.ylabel("$T_e$ [K]", fontsize=LABEL_FONTSIZE)
    plt.title("Surface Electron Temperature Comparison", fontsize=TITLE_FONTSIZE)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "Te_surface_compare.png", dpi=DPI)
    plt.close()


def _plot_reflectivity_compare(results: list, fluences: list[float], output_dir: Path) -> None:
    """グラフ#11: R(t) 比較"""
    plt.figure(figsize=(10, 6))
    for result, F in zip(results, fluences):
        time_ps = _convert_time_to_ps(result.time_points)
        plt.plot(time_ps, result.reflectivity_history, label=f"F = {F:.2f} J/cm²", linewidth=2)
    plt.xlabel("Time [ps]", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Reflectivity R", fontsize=LABEL_FONTSIZE)
    plt.title("Surface Reflectivity Comparison", fontsize=TITLE_FONTSIZE)
    plt.ylim(0, 1)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "reflectivity_compare.png", dpi=DPI)
    plt.close()


def _plot_alpha_fca_compare(results: list, fluences: list[float], output_dir: Path) -> None:
    """グラフ#12: α_FCA(t) 比較"""
    plt.figure(figsize=(10, 6))
    for result, F in zip(results, fluences):
        time_ps = _convert_time_to_ps(result.time_points)
        plt.plot(time_ps, result.alpha_fca_surface_history, label=f"F = {F:.2f} J/cm²", linewidth=2)
    plt.xlabel("Time [ps]", fontsize=LABEL_FONTSIZE)
    plt.ylabel("$\\alpha_{FCA}$ [cm$^{-1}$]", fontsize=LABEL_FONTSIZE)
    plt.title("FCA Absorption Coefficient Comparison", fontsize=TITLE_FONTSIZE)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "alpha_fca_compare.png", dpi=DPI)
    plt.close()


def _plot_auger_ne_compare(results: list, fluences: list[float], output_dir: Path) -> None:
    """グラフ#13: γne³ + ne 比較"""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    for result, F in zip(results, fluences):
        time_ps = _convert_time_to_ps(result.time_points)
        ax1.semilogy(time_ps, result.auger_term_surface_history, label=f"F = {F:.2f} J/cm²", linewidth=2)
    
    ax1.set_xlabel("Time [ps]", fontsize=LABEL_FONTSIZE)
    ax1.set_ylabel("$\\gamma n_e^3$ [cm$^{-3}$ s$^{-1}$]", fontsize=LABEL_FONTSIZE)
    ax1.set_title("Auger Recombination Term Comparison", fontsize=TITLE_FONTSIZE)
    ax1.legend(fontsize=LEGEND_FONTSIZE)
    ax1.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / "auger_ne_compare.png", dpi=DPI)
    plt.close()


def _plot_ablation_depth_vs_fluence(
    results: list,
    fluences: list[float],
    output_dir: Path,
    experimental_data: dict[float, float] | None,
) -> None:
    """グラフ#14: アブレーション深さ vs フルエンス"""
    plt.figure(figsize=(10, 6))
    
    # シミュレーション結果
    depths_nm = [r.ablation_depth_nm for r in results]
    plt.plot(fluences, depths_nm, "o-", label="Simulation", linewidth=2, markersize=8)
    
    # 実験データ（あれば）
    if experimental_data:
        exp_fluences = sorted(experimental_data.keys())
        exp_depths = [experimental_data[f] for f in exp_fluences]
        plt.plot(exp_fluences, exp_depths, "s--", label="Experimental", linewidth=2, markersize=8, color="gray")
    
    plt.xlabel("Fluence [J/cm²]", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Ablation Depth [nm]", fontsize=LABEL_FONTSIZE)
    plt.title("Ablation Depth vs Fluence", fontsize=TITLE_FONTSIZE)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "ablation_depth_vs_fluence.png", dpi=DPI)
    plt.close()
