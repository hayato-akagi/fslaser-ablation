"""reproduce/fig8.py — 論文 Figure 8 の再現。

論文 Figure 8:
  左軸: Auger再結合率 γn³ [10³⁴ s⁻¹ cm⁻³] (緑)
  右軸: キャリア密度 ne  [10²¹ cm⁻³]       (橙)
  時間軸: 0〜4 ps  (F=0.25 と F=3.06 J/cm²)

論文記載値:
  F=0.25: γn³_max = 7.4×10³¹ s⁻¹cm⁻³,  ne_max ≈ 0.6×10²¹
  F=3.06: γn³_max = 6.7×10³⁴ s⁻¹cm⁻³,  ne_max ≈ 6×10²¹  (t≈1.3〜1.6ps に2ピーク)

【パルス調整パラメータ】  ← 下の定数ブロックを参照
  PULSE_DURATION : パルス幅 FWHM [s]
  T_PEAK_PS      : レーザーピーク時刻 [ps]（論文時間軸）
  FLUENCE_025    : F=低フルエンス側 [J/cm²]  →  ピーク強度 ∝ F / PULSE_DURATION
  FLUENCE_306    : F=高フルエンス側 [J/cm²]

実行:
  docker compose run --rm sim python reproduce/fig8.py
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
from modules.material_properties.constants import SILICON

# ─────────────────────────────────────────────────────────────
# ▼ ここを変えてパルスを調整
PULSE_DURATION = 421e-15   # パルス幅 FWHM [s]
T_PEAK_PS      = 1.5     # レーザーピーク時刻 [ps]（論文時間軸）
FLUENCE_025    = 0.25      # 低フルエンス [J/cm²]  ピーク強度 ≈ 0.94×F/tp
FLUENCE_306    = 3.06      # 高フルエンス [J/cm²]
# ─────────────────────────────────────────────────────────────

N_Z          = 50
DZ           = 5e-7
T_PAPER_END  = 4e-12                        # 論文時間軸の終端 [s]
T_START      = -T_PEAK_PS * 1e-12           # シミュレーション開始（= 論文 t=0）
T_END        = T_PAPER_END + T_START        # シミュレーション終了
DT_MAX       = 1e-17
RECORD_EVERY = 200


def run_simulation(fluence: float) -> dict:
    """指定フルエンスでシミュレーションを実行し、γn³ と ne の時系列を返す。"""
    config = EulerFDMConfig(
        fluence=fluence,
        grid=GridConfig(n_z=N_Z, dz=DZ),
        time=TimeConfig(t_end=T_END, dt_max=DT_MAX, snapshot_interval=RECORD_EVERY),
        initial=InitialCondition(),
    )

    # OpticsConfig に PULSE_DURATION を明示注入
    optics_cfg = OpticsConfig(dz=DZ, fluence=fluence, pulse_duration=PULSE_DURATION)
    _, carrier_cfg, ttm_cfg, _ = create_domain_configs(config)

    ne, Te, Tl, phase_state, latent_acc, _, _ = initialize_state_vectors(
        config.grid, config.initial
    )

    t    = T_START   # T_PEAK_PS から導出した開始時刻を使用
    step = 0
    times_list  = []
    ne_list     = []
    auger_list  = []

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
            paper_t    = (t - T_START) * 1e12   # 論文時間軸（T_START = 論文 t=0）
            auger_rate = SILICON.gamma_auger * ne[0] ** 3
            times_list.append(paper_t)
            ne_list.append(ne[0])
            auger_list.append(auger_rate)

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

    return {
        "t":     np.array(times_list),
        "ne":    np.array(ne_list)    / 1e21,   # [10²¹ cm⁻³]
        "auger": np.array(auger_list) / 1e34,   # [10³⁴ s⁻¹ cm⁻³]
    }


def _laser_intensity_profile(
    t_peak_ps: float,
    pulse_duration: float,
    t_range_ps: tuple[float, float],
    n_pts: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """論文時間軸上のレーザー強度プロファイル（正規化、形状のみ）。

    Args:
        t_peak_ps:     ピーク時刻 [ps]
        pulse_duration: パルス幅 FWHM [s]

    Returns:
        t_ps:   論文時間 [ps], shape (n_pts,)
        I_norm: 正規化強度 0〜1, shape (n_pts,)
    """
    tp_ps  = pulse_duration * 1e12
    t_ps   = np.linspace(t_range_ps[0], t_range_ps[1], n_pts)
    I_norm = np.exp(-2.77 * ((t_ps - t_peak_ps) / tp_ps) ** 2)
    return t_ps, I_norm


def plot_figure8(
    d025: dict,
    d306: dict,
    t_peak_ps: float,
    pulse_duration: float,
    out_path: str,
) -> None:
    """論文 Figure 8 を再現するグラフを描画する。"""
    GREEN  = "#2E7D32"
    ORANGE = "#E86B00"
    RED    = "#CC0000"

    fig, ax = plt.subplots(figsize=(7, 5))
    axr = ax.twinx()

    xlim        = (0, 4)
    axr_ylim    = 6.0

    # γn³（左軸、緑）
    ax.plot(d306["t"], d306["auger"], color=GREEN, lw=2,          label=r"$\gamma n^3$  3.06 J/cm²")
    ax.plot(d025["t"], d025["auger"], color=GREEN, lw=2, ls="--", label=r"$\gamma n^3$  0.25 J/cm²")

    # ne（右軸、橙）
    axr.plot(d306["t"], d306["ne"], color=ORANGE, lw=2,          label=r"$n_e$  3.06 J/cm²")
    axr.plot(d025["t"], d025["ne"], color=ORANGE, lw=2, ls="--", label=r"$n_e$  0.25 J/cm²")

    # レーザー強度（右軸に正規化、形状のみ）
    t_I, I_norm = _laser_intensity_profile(t_peak_ps, pulse_duration, xlim)
    I_scaled = I_norm * axr_ylim * 0.25   # 右軸最大値の 25% にスケール
    axr.fill_between(t_I, I_scaled, alpha=0.25, color=RED, label="Laser intensity (a.u.)")

    ax.set_xlabel("Time (ps)", fontsize=12)
    ax.set_ylabel(r"Auger recombination ($10^{34}$ s$^{-1}$cm$^{-3}$)", color=GREEN, fontsize=11)
    axr.set_ylabel(r"Carrier density ($10^{21}$ cm$^{-3}$)", color=ORANGE, fontsize=11)
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 7)
    axr.set_ylim(0, axr_ylim)
    ax.tick_params(axis="y", labelcolor=GREEN)
    axr.tick_params(axis="y", labelcolor=ORANGE)
    ax.set_title(
        f"Figure 8  (tp={pulse_duration*1e15:.0f} fs, peak={t_peak_ps:.3f} ps)",
        fontsize=11,
    )

    lines  = ax.get_lines() + axr.get_lines()
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"保存: {out_path}")


def _print_summary(label: str, d: dict) -> None:
    idx_auger = np.argmax(d["auger"])
    idx_ne    = np.argmax(d["ne"])
    print(f"  γn³_max = {d['auger'].max():.4f} ×10³⁴  @ t = {d['t'][idx_auger]:.3f} ps")
    print(f"  ne_max  = {d['ne'].max():.4f} ×10²¹  @ t = {d['t'][idx_ne]:.3f} ps")

    if "0.25" in label:
        print(f"  論文: γn³_max = 7.4e31 = 0.0074×10³⁴,  ne_max ≈ 0.6×10²¹")
        print(f"  比率: γn³ {d['auger'].max()/0.0074:.2f}x,  ne {d['ne'].max()/0.6:.2f}x")
    else:
        print(f"  論文: γn³_max = 6.7×10³⁴,  ne_max ≈ 6.0×10²¹")
        print(f"  比率: γn³ {d['auger'].max()/6.7:.2f}x,  ne {d['ne'].max()/6.0:.2f}x")

    ne_at_peak     = d["ne"][idx_auger] * 1e21
    auger_at_peak  = d["auger"][idx_auger] * 1e34
    if ne_at_peak > 1e10:
        gamma_calc = auger_at_peak / (ne_at_peak ** 3)
        print(f"  逆算 γ = {gamma_calc:.3e} cm⁶/s  (設定値: {SILICON.gamma_auger:.3e})")


if __name__ == "__main__":
    print(f"PULSE_DURATION = {PULSE_DURATION*1e15:.0f} fs")
    print(f"T_PEAK_PS      = {T_PEAK_PS:.3f} ps")
    print(f"T_START = {T_START*1e12:.3f} ps  T_END = {T_END*1e12:.3f} ps")
    print(f"FLUENCE_025 = {FLUENCE_025} J/cm²  FLUENCE_306 = {FLUENCE_306} J/cm²")
    print()

    print(f"=== F = {FLUENCE_025} J/cm² ===")
    t0   = time.time()
    d025 = run_simulation(FLUENCE_025)
    print(f"  実行時間: {time.time()-t0:.1f}s")
    _print_summary("0.25", d025)

    print()
    print(f"=== F = {FLUENCE_306} J/cm² ===")
    t0   = time.time()
    d306 = run_simulation(FLUENCE_306)
    print(f"  実行時間: {time.time()-t0:.1f}s")
    _print_summary("3.06", d306)

    out_path = os.path.join(os.path.dirname(__file__), "fig8_output.png")
    print()
    plot_figure8(d025, d306, T_PEAK_PS, PULSE_DURATION, out_path)
