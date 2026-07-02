"""reproduce/fig6.py — 論文 Figure 6 の再現。

論文 Figure 6:
  (a) 電子温度 Te [10³ K] + キャリア密度 ne [10²⁰ cm⁻³]  vs 時間 0〜6 ps
      + レーザー強度プロファイル（形状のみ）
  (b) (a) の拡大: 時間 0〜1 ps, Te 0〜3×10³ K（レーザー強度なし）

  F = 0.25 J/cm² のみ

【パルス調整パラメータ】
  PULSE_DURATION : パルス幅 FWHM [s]
  T_PEAK_PS      : レーザーピーク時刻 [ps]（論文時間軸）
  FLUENCE        : フルエンス [J/cm²]

実行:
  docker compose run --rm sim python reproduce/fig6.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modules.euler_fdm.config import EulerFDMConfig, GridConfig, TimeConfig, InitialCondition
from modules.euler_fdm.solver import initialize_state_vectors, create_domain_configs, compute_safe_dt
from modules.optics.config import OpticsConfig
from modules.optics.public import compute_laser_field
from modules.carrier.public import advance_carrier_density
from modules.ttm.public import advance_temperatures

# ─────────────────────────────────────────────────────────────
# ▼ ここを変えてパルスを調整
PULSE_DURATION = 421e-15   # パルス幅 FWHM [s]
T_PEAK_PS      = 1.5     # レーザーピーク時刻 [ps]（論文時間軸）
FLUENCE        = 0.25      # フルエンス [J/cm²]
# ─────────────────────────────────────────────────────────────

N_Z          = 50
DZ           = 5e-7
T_PAPER_END  = 6e-12                        # 論文時間軸の終端 [s]
T_START      = -T_PEAK_PS * 1e-12           # シミュレーション開始（= 論文 t=0）
T_END        = T_PAPER_END + T_START        # シミュレーション終了
DT_MAX       = 5e-20
RECORD_EVERY = 200


def run_simulation(fluence: float) -> dict:
    """指定フルエンスでシミュレーションを実行し、Te・ne の時系列を返す。"""
    config = EulerFDMConfig(
        fluence=fluence,
        grid=GridConfig(n_z=N_Z, dz=DZ),
        time=TimeConfig(t_end=T_END, dt_max=DT_MAX, snapshot_interval=RECORD_EVERY),
        initial=InitialCondition(),
        te_scheme="euler",  # 前進オイラー（論文再現用）。CFL条件は DT_MAX で管理。
    )

    optics_cfg = OpticsConfig(dz=DZ, fluence=fluence, pulse_duration=PULSE_DURATION)
    _, carrier_cfg, ttm_cfg, _ = create_domain_configs(config)

    ne, Te, Tl, phase_state, latent_acc, _, _ = initialize_state_vectors(
        config.grid, config.initial
    )

    t      = T_START
    step   = 0
    t0_sim = time.time()
    t_span = T_END - T_START
    times_list = []
    Te_list    = []
    ne_list    = []

    while t < T_END:
        opt = compute_laser_field(
            ne=ne, Tl=Tl, Te=Te, phase_state=phase_state, t=t, config=optics_cfg
        )
        S_max = np.abs(opt.source_term).max()
        dt = (
            DT_MAX if S_max < 1e6
            else compute_safe_dt(
                Te=Te, Tl=Tl, ne=ne, dz=DZ, dt_max=DT_MAX,
                source_term=opt.source_term,
            )
        )

        if step % RECORD_EVERY == 0:
            paper_t = (t - T_START) * 1e12
            times_list.append(paper_t)
            Te_list.append(Te[0])
            ne_list.append(ne[0])

            elapsed = time.time() - t0_sim
            done    = t - T_START
            pct     = done / t_span * 100
            eta_str = f"{elapsed / done * (t_span - done):.0f}s" if done > 0 else "--"
            print(
                f"\r  {pct:5.1f}%  t={paper_t:6.3f}/{T_PAPER_END*1e12:.1f} ps"
                f"  Te={Te[0]/1e3:.1f}e3K  ne={ne[0]:.2e}"
                f"  elapsed={elapsed:.0f}s  ETA={eta_str}   ",
                end="", flush=True,
            )

        car = advance_carrier_density(
            ne=ne, intensity=opt.intensity, Te=Te, Tl=Tl,
            phase_state=phase_state, dt=dt, config=carrier_cfg,
        )
        ttm = advance_temperatures(
            Te=Te, Tl=Tl, ne=ne, dne_dt=car.dne_dt,
            source_term=opt.source_term, phase_state=phase_state,
            latent_heat_accumulated=latent_acc, dt=dt, config=ttm_cfg,
        )
        ne, Te, Tl = car.ne, ttm.Te, ttm.Tl
        phase_state, latent_acc = ttm.phase_state, ttm.latent_heat_accumulated
        t    += dt
        step += 1

    print(f"\r  100.0%  完了  elapsed={time.time()-t0_sim:.0f}s" + " " * 40)
    return {
        "t":  np.array(times_list),
        "Te": np.array(Te_list) / 1e3,    # [10³ K]
        "ne": np.array(ne_list) / 1e20,   # [10²⁰ cm⁻³]
    }


def _laser_intensity_profile(
    t_peak_ps: float,
    pulse_duration: float,
    t_range_ps: tuple[float, float],
    n_pts: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    tp_ps  = pulse_duration * 1e12
    t_ps   = np.linspace(t_range_ps[0], t_range_ps[1], n_pts)
    I_norm = np.exp(-2.77 * ((t_ps - t_peak_ps) / tp_ps) ** 2)
    return t_ps, I_norm


def plot_figure6(
    d: dict,
    t_peak_ps: float,
    pulse_duration: float,
    out_path: str,
) -> None:
    """論文 Figure 6 (a)(b) を再現するグラフを描画する。"""
    BLACK  = "#000000"
    ORANGE = "#E86B00"
    RED    = "#CC0000"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── (a) 全体 0〜6 ps ──────────────────────────────────────
    ax  = axes[0]
    axr = ax.twinx()

    xlim_a    = (0, 6)
    Te_ylim_a = 200.0
    ne_ylim_a = 6.0

    ax.plot(d["t"], d["Te"], color=BLACK, lw=2, label=r"$T_e$")
    axr.plot(d["t"], d["ne"], color=ORANGE, lw=2, label=r"$n_e$")

    t_I, I_norm = _laser_intensity_profile(t_peak_ps, pulse_duration, xlim_a)
    I_scaled = I_norm * ne_ylim_a * 0.25
    axr.fill_between(t_I, I_scaled, alpha=0.25, color=RED, label="Laser intensity (a.u.)")

    ax.set_xlabel("Time (ps)", fontsize=12)
    ax.set_ylabel(r"Temperature ($\times 10^3$ K)", color=BLACK, fontsize=11)
    axr.set_ylabel(r"Carrier density ($10^{20}$ cm$^{-3}$)", color=ORANGE, fontsize=11)
    ax.set_xlim(*xlim_a)
    ax.set_ylim(0, Te_ylim_a)
    axr.set_ylim(0, ne_ylim_a)
    ax.tick_params(axis="y", labelcolor=BLACK)
    axr.tick_params(axis="y", labelcolor=ORANGE)
    ax.set_title("(a) F = 0.25 J/cm²  (0–6 ps)", fontsize=12)

    lines  = ax.get_lines() + axr.get_lines()
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=9, loc="upper right")

    # ── (b) 拡大 0〜1 ps ─────────────────────────────────────
    ax  = axes[1]
    axr = ax.twinx()

    xlim_b    = (0, 1)
    Te_ylim_b = 3.0
    ne_ylim_b = 6.0

    mask = (d["t"] >= xlim_b[0]) & (d["t"] <= xlim_b[1])
    ax.plot(d["t"][mask], d["Te"][mask], color=BLACK, lw=2, label=r"$T_e$")
    axr.plot(d["t"][mask], d["ne"][mask], color=ORANGE, lw=2, label=r"$n_e$")

    ax.set_xlabel("Time (ps)", fontsize=12)
    ax.set_ylabel(r"Temperature ($\times 10^3$ K)", color=BLACK, fontsize=11)
    axr.set_ylabel(r"Carrier density ($10^{20}$ cm$^{-3}$)", color=ORANGE, fontsize=11)
    ax.set_xlim(*xlim_b)
    ax.set_ylim(0, Te_ylim_b)
    axr.set_ylim(0, ne_ylim_b)
    ax.tick_params(axis="y", labelcolor=BLACK)
    axr.tick_params(axis="y", labelcolor=ORANGE)
    ax.set_title("(b) F = 0.25 J/cm²  (0–1 ps, zoom)", fontsize=12)

    lines  = ax.get_lines() + axr.get_lines()
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=9, loc="upper right")

    plt.suptitle(
        f"Figure 6  (tp={pulse_duration*1e15:.0f} fs, peak={t_peak_ps:.3f} ps)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"保存: {out_path}")


def _print_summary(d: dict) -> None:
    idx_Te = np.argmax(d["Te"])
    idx_ne = np.argmax(d["ne"])
    print(f"  Te_max = {d['Te'].max():.1f} ×10³ K  @ t = {d['t'][idx_Te]:.3f} ps")
    print(f"  ne_max = {d['ne'].max():.4f} ×10²⁰ cm⁻³  @ t = {d['t'][idx_ne]:.3f} ps")


if __name__ == "__main__":
    print(f"PULSE_DURATION = {PULSE_DURATION*1e15:.0f} fs")
    print(f"T_PEAK_PS      = {T_PEAK_PS:.3f} ps")
    print(f"T_START = {T_START*1e12:.3f} ps  T_END = {T_END*1e12:.3f} ps")
    print(f"FLUENCE = {FLUENCE} J/cm²")
    print()

    print(f"=== F = {FLUENCE} J/cm² ===")
    t0 = time.time()
    d  = run_simulation(FLUENCE)
    print(f"  実行時間: {time.time()-t0:.1f}s")
    _print_summary(d)

    out_path = os.path.join(os.path.dirname(__file__), "fig6_output.png")
    print()
    plot_figure6(d, T_PEAK_PS, PULSE_DURATION, out_path)
