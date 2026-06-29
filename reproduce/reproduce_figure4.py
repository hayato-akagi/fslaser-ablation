"""Figure 4 の再現（進捗表示・中間プロット・タイムスタンプ付き出力）

使用方法:
    docker compose run --rm sim python reproduce/reproduce_figure4.py
"""

import datetime
import os
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# プロジェクトルートを sys.path に追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.euler_fdm.config import EulerFDMConfig, GridConfig, TimeConfig
from modules.euler_fdm.public import SimulationResult, run_simulation

# ============================================================
# 定数
# ============================================================

FLUENCES = [0.25, 0.8, 1.5]          # J/cm²
COLORS = ["black", "black", "black"]
LINESTYLES = ["-", "--", ":"]
T_END_DEFAULT = 150e-12                # 150 ps [s]
T_END_HIGH = 200e-12                   # 200 ps [s]（F=1.5 J/cm² のみ）
T_END = T_END_HIGH                     # 後方互換（最終プロット用）
SNAPSHOT_INTERVAL = 1000               # ステップごとにスナップショット記録
PRINT_EVERY_N_SNAPSHOTS = 10          # このスナップショット数ごとに進捗を表示
CHECKPOINT_INTERVAL_PS = 40.0         # この間隔 [ps] ごとに中間プロットを保存


# ============================================================
# 進捗表示ユーティリティ
# ============================================================


def _fmt_sec(secs: float) -> str:
    """秒を HH:MM:SS 形式に変換する。"""
    if not np.isfinite(secs):
        return "--:--:--"
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _make_bar(frac: float, width: int = 28) -> str:
    """テキスト進捗バーを返す。"""
    filled = int(width * max(0.0, min(1.0, frac)))
    return "[" + "=" * filled + "-" * (width - filled) + "]"


# ============================================================
# ProgressTracker
# ============================================================


class ProgressTracker:
    """1フルエンス分のシミュレーション進捗を管理する。

    シグネチャ: (step_index, t, t_end, history) -> None
    snapshot_interval ごとに solver から呼び出される。
    """

    def __init__(
        self,
        fluence: float,
        t_start: float,
        t_end: float,
        progress_dir: str,
    ) -> None:
        self.fluence = fluence
        self.t_start = t_start
        self.t_end = t_end
        self.progress_dir = progress_dir
        self.wall_start = time.monotonic()
        self.snapshot_count = 0
        self.last_checkpoint_idx = -1
        os.makedirs(progress_dir, exist_ok=True)

    def __call__(
        self,
        step_index: int,
        t: float,
        t_end: float,
        history: dict,
    ) -> None:
        self.snapshot_count += 1
        t_ps = t * 1e12

        self._maybe_save_checkpoint(t_ps, history)

        if self.snapshot_count % PRINT_EVERY_N_SNAPSHOTS == 0:
            self._print_progress(step_index, t_ps, history)

    def _maybe_save_checkpoint(self, t_ps: float, history: dict) -> None:
        """CHECKPOINT_INTERVAL_PS ごとに中間プロットを保存する。"""
        checkpoint_idx = int(max(0.0, t_ps) / CHECKPOINT_INTERVAL_PS)
        if checkpoint_idx <= self.last_checkpoint_idx:
            return
        self.last_checkpoint_idx = checkpoint_idx
        _save_checkpoint_plot(self.fluence, t_ps, history, self.progress_dir)

    def _print_progress(
        self, step_index: int, t_ps: float, history: dict
    ) -> None:
        elapsed = time.monotonic() - self.wall_start
        frac = (t_ps * 1e-12 - self.t_start) / (self.t_end - self.t_start)
        frac = max(0.0, min(1.0, frac))
        eta = elapsed / frac * (1.0 - frac) if frac > 1e-3 else float("inf")

        Tl = history["Tl_surface_history"][-1] if history["Tl_surface_history"] else float("nan")
        Te = history["Te_surface_history"][-1] if history["Te_surface_history"] else float("nan")

        print(
            f"  {_make_bar(frac)} {frac*100:5.1f}%"
            f" | t={t_ps:7.1f}/{self.t_end*1e12:.0f} ps"
            f" | step={step_index:,}"
            f" | {_fmt_sec(elapsed)} → ETA {_fmt_sec(eta)}"
            f" | Tl={Tl/1e3:.2f}  Te={Te/1e3:.2f}  (×10³K)"
        )


# ============================================================
# チェックポイントプロット（1フルエンス・中間）
# ============================================================


def _save_checkpoint_plot(
    fluence: float,
    t_now_ps: float,
    history: dict,
    progress_dir: str,
) -> None:
    """現在の履歴から Tl / Te / ne の時間発展プロットを保存する。"""
    if not history["time_points"]:
        return

    time_ps = np.array(history["time_points"]) * 1e12
    Tl = np.array(history["Tl_surface_history"]) / 1000
    Te = np.array(history["Te_surface_history"]) / 1000
    ne = np.array(history["ne_surface_history"])
    mask = time_ps > 0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    _plot_log(axes[0], time_ps[mask], Tl[mask], "Tl (10³ K)", "Lattice Temp")
    _plot_log(axes[1], time_ps[mask], Te[mask], "Te (10³ K)", "Electron Temp")
    _plot_ne(axes[2], time_ps[mask], ne[mask])

    fig.suptitle(
        f"F = {fluence} J/cm²  |  t = {t_now_ps:.0f} ps",
        fontsize=12,
    )
    plt.tight_layout()

    fname = os.path.join(progress_dir, f"cp_{int(t_now_ps):04d}ps.png")
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    → チェックポイント保存: {os.path.basename(fname)}")


def _plot_log(ax, time_ps, values, ylabel: str, title: str) -> None:
    if len(time_ps) > 0:
        ax.plot(time_ps, values, lw=1.5)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def _plot_ne(ax, time_ps, ne) -> None:
    if len(time_ps) > 0 and (ne > 0).any():
        ax.semilogy(time_ps, np.maximum(ne, 1e10), lw=1.5, color="green")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("ne (cm⁻³)")
    ax.set_title("Carrier Density")
    ax.grid(True, alpha=0.3)


# ============================================================
# 途中経過 Figure 4(a)（完了済みフルエンスのみ）
# ============================================================


def _save_partial_figure4(results: dict, output_dir: str) -> None:
    """完了済みフルエンスで Figure 4(a) の途中経過を保存する。"""
    fig, ax = plt.subplots(figsize=(7, 5))

    for fluence, result in results.items():
        idx = FLUENCES.index(fluence)
        time_ps = result.time_points * 1e12
        Tl_k3 = result.Tl_surface_history / 1000
        mask = time_ps > 0
        ax.plot(
            time_ps[mask],
            Tl_k3[mask],
            color=COLORS[idx],
            linestyle=LINESTYLES[idx],
            lw=2,
            label=f"{fluence} J/cm²",
        )

    ax.set_xlabel("Time (ps)", fontsize=12)
    ax.set_ylabel("Tl (10³ K)", fontsize=12)
    ax.set_xscale("log")
    ax.set_xlim(1, 200)
    ax.set_ylim(0, 9)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Figure 4(a) — {len(results)}/{len(FLUENCES)} fluences done")
    plt.tight_layout()

    fname = os.path.join(output_dir, "figure4a_partial.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → 途中経過プロット更新: {os.path.basename(fname)}")


# ============================================================
# 最終プロット（Figure 4 完成版）
# ============================================================


def _save_final_figure4(results: dict, output_dir: str) -> str:
    """Figure 4 (a)(b) を最終出力として保存する。"""
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5))

    for fluence, result in results.items():
        idx = FLUENCES.index(fluence)
        time_ps = result.time_points * 1e12
        Tl_k3 = result.Tl_surface_history / 1000
        mask = time_ps > 0
        ax_a.plot(
            time_ps[mask],
            Tl_k3[mask],
            color=COLORS[idx],
            linestyle=LINESTYLES[idx],
            lw=2,
            label=f"{fluence} J/cm²",
        )

    ax_a.set_xlabel("Time (ps)", fontsize=12)
    ax_a.set_ylabel("Lattice temperature (10³ K)", fontsize=12)
    ax_a.set_xscale("log")
    ax_a.set_xlim(1, 200)
    ax_a.set_ylim(0, 9)
    ax_a.legend(fontsize=11, loc="upper left")
    ax_a.grid(True, alpha=0.3)
    ax_a.text(0.05, 0.95, "(a)", transform=ax_a.transAxes, fontsize=14, fontweight="bold", va="top")

    result_1p5 = results[1.5]
    time_ps = result_1p5.time_points * 1e12
    mask_b = (time_ps >= 80) & (time_ps <= 200)
    ax_b.plot(time_ps[mask_b], result_1p5.Te_surface_history[mask_b] / 1000, color="blue", lw=2, label="Te")
    ax_b.plot(time_ps[mask_b], result_1p5.Tl_surface_history[mask_b] / 1000, color="red", linestyle="--", lw=2, label="Tl")

    ax_b.set_xlabel("Time (ps)", fontsize=12)
    ax_b.set_ylabel("Temperature (10³ K)", fontsize=12)
    ax_b.set_xlim(80, 200)
    ax_b.set_ylim(5, 9)
    ax_b.legend(fontsize=11, loc="upper right")
    ax_b.grid(True, alpha=0.3)
    ax_b.text(0.05, 0.95, "(b)", transform=ax_b.transAxes, fontsize=14, fontweight="bold", va="top")

    plt.tight_layout()
    fname = os.path.join(output_dir, "figure4_reproduction.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fname


# ============================================================
# 1フルエンス実行
# ============================================================


def _run_one_fluence(fluence: float, output_dir: str) -> SimulationResult:
    """1フルエンス分のシミュレーションを実行して結果を返す。"""
    progress_dir = os.path.join(output_dir, "progress", f"F{fluence:.2f}")
    t_end = T_END_HIGH if fluence == 1.5 else T_END_DEFAULT
    t_start = TimeConfig(t_end=t_end).t_start

    config = EulerFDMConfig(
        fluence=fluence,
        grid=GridConfig(n_z=1000, dz=5.0e-7),
        time=TimeConfig(t_end=t_end, dt_max=1e-15, snapshot_interval=SNAPSHOT_INTERVAL),
        progress_callback=ProgressTracker(
            fluence=fluence,
            t_start=t_start,
            t_end=t_end,
            progress_dir=progress_dir,
        ),
    )

    return run_simulation(config)


def _save_data(result: SimulationResult, fluence: float, data_dir: str) -> None:
    """シミュレーション結果を npz で保存する。"""
    os.makedirs(data_dir, exist_ok=True)
    fname = os.path.join(data_dir, f"F{fluence:.2f}Jcm2.npz")
    np.savez(
        fname,
        time_points=result.time_points,
        Te_surface=result.Te_surface_history,
        Tl_surface=result.Tl_surface_history,
        ne_surface=result.ne_surface_history,
        ablation_depth_nm=np.array([result.ablation_depth_nm]),
    )
    print(f"  データ保存: {os.path.basename(fname)}")


# ============================================================
# main
# ============================================================


def _print_header(output_dir: str) -> None:
    print("=" * 72)
    print("  Figure 4 再現スクリプト")
    print(f"  出力先         : {output_dir}/")
    print(f"  フルエンス     : {FLUENCES} J/cm²")
    print(f"  終了時刻       : {T_END_DEFAULT*1e12:.0f} ps (F=0.25,0.8)  /  {T_END_HIGH*1e12:.0f} ps (F=1.5)")
    print(f"  スナップショット: {SNAPSHOT_INTERVAL} ステップごと")
    print(f"  チェックポイント: {CHECKPOINT_INTERVAL_PS:.0f} ps ごと（中間プロット）")
    print("=" * 72)


def main() -> None:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"{timestamp}_figure4")
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(output_dir, exist_ok=True)

    _print_header(output_dir)

    results: dict = {}
    wall_total_start = time.monotonic()

    for i, fluence in enumerate(FLUENCES):
        print(f"\n{'='*72}")
        print(f"  [{i+1}/{len(FLUENCES)}]  F = {fluence} J/cm²")
        print(f"{'='*72}")

        t_fluence_start = time.monotonic()
        result = _run_one_fluence(fluence, output_dir)
        elapsed_fluence = time.monotonic() - t_fluence_start

        results[fluence] = result

        print(f"\n  完了: {result.total_steps:,} ステップ | {_fmt_sec(elapsed_fluence)}")
        print(f"  最終 Tl={result.Tl_final[0]:.1f} K  ablation={result.ablation_depth_nm:.1f} nm")
        _save_data(result, fluence, data_dir)
        _save_partial_figure4(results, output_dir)

    final_plot = _save_final_figure4(results, output_dir)
    elapsed_total = time.monotonic() - wall_total_start

    print(f"\n{'='*72}")
    print(f"  全フルエンス完了 | 総実行時間: {_fmt_sec(elapsed_total)}")
    print(f"  最終プロット    : {final_plot}")
    print(f"  出力ディレクトリ: {output_dir}/")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
