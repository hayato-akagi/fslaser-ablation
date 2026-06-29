"""reproduce/fig5.py — 論文 Figure 5 の再現。

論文 Figure 5:
  (a) 動的反射率 R + キャリア密度 ne  vs 時間 (F=0.25, 3.06 J/cm²)
  (b) FCA 係数 αFCA + キャリア密度 ne vs 時間 (F=0.25, 3.06 J/cm²)

時間軸の対応:
  論文の t=0 = シミュレーション開始時刻 (t_start = -3×tp = -1.263 ps)
  論文の t=6ps = our t = 4.737 ps

パラメータ: 論文準拠
  m_eff = me = 9.11e-31 kg (Table 1)

実行:
  docker run --rm -v "$(pwd):/app" -w /app fslaser-sim python reproduce/fig5.py
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
from modules.euler_fdm.solver import (
    initialize_state_vectors,
    create_domain_configs,
    compute_safe_dt,
)
from modules.optics.public import compute_laser_field
from modules.carrier.public import advance_carrier_density
from modules.ttm.public import advance_temperatures

# ─────────────────────────────────────────
# シミュレーション設定
# ─────────────────────────────────────────
N_Z          = 50
DZ           = 5e-7       # 5 nm セル [cm]
T_END        = 4.737e-12  # our t=4.737ps → 論文 t=6ps
DT_MAX       = 1e-15      # 1 fs
RECORD_EVERY = 200        # 何ステップごとにデータ記録するか

# 論文の時間軸オフセット: paper_t = our_t - t_start
PULSE_DURATION = 421e-15  # 421 fs (FWHM)
T_START        = -3.0 * PULSE_DURATION  # -1.263 ps


def run_simulation(fluence: float) -> dict:
    """指定フルエンスでシミュレーションを実行し、時系列データを返す。

    Returns:
        dict with keys:
            t    [ps]          : 論文時間軸 (paper_t = our_t - T_START)
            R    [-]           : 表面反射率
            ne   [10²¹ cm⁻³]  : 表面キャリア密度
            fca  [µm⁻¹]       : 表面 FCA 係数 (α_FCA / 10⁴)
    """
    config = EulerFDMConfig(
        fluence=fluence,
        grid=GridConfig(n_z=N_Z, dz=DZ),
        time=TimeConfig(t_end=T_END, dt_max=DT_MAX, snapshot_interval=200),
        initial=InitialCondition(),
    )

    optics_cfg, carrier_cfg, ttm_cfg, _ = create_domain_configs(config)

    # Drude モデル: 光学有効質量 m* = 0.26 me を使用（デフォルト値をそのまま使う）

    ne, Te, Tl, phase_state, latent_acc, _, _ = initialize_state_vectors(
        config.grid, config.initial
    )

    t    = config.time.t_start  # = -1.263e-12 s
    step = 0

    times_list  = []
    R_list      = []
    ne_list     = []
    fca_list    = []

    while t < T_END:
        opt = compute_laser_field(
            ne=ne, Tl=Tl, Te=Te, phase_state=phase_state, t=t, config=optics_cfg
        )

        S_max = np.abs(opt.source_term).max()
        dt = (
            DT_MAX
            if S_max < 1e6
            else compute_safe_dt(
                Te=Te, Tl=Tl, ne=ne, dz=DZ, dt_max=DT_MAX,
                source_term=opt.source_term,
            )
        )

        if step % RECORD_EVERY == 0:
            paper_t = (t - config.time.t_start) * 1e12   # [ps]
            times_list.append(paper_t)
            R_list.append(opt.reflectivity)
            ne_list.append(ne[0])
            fca_list.append(opt.alpha_fca[0])

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
        "t":   np.array(times_list),
        "R":   np.array(R_list),
        "ne":  np.array(ne_list) / 1e21,     # 10²¹ cm⁻³
        "fca": np.array(fca_list) / 1e4,     # cm⁻¹ → µm⁻¹
    }


def plot_figure5(d025: dict, d306: dict, out_path: str) -> None:
    """論文 Figure 5 (a)(b) を 1 枚に描画する。"""
    BLUE   = "#0072BD"
    ORANGE = "#E86B00"
    PURPLE = "#7E2F8E"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── (a) 反射率 + キャリア密度 ──────────────────────────
    ax  = axes[0]
    axr = ax.twinx()

    ax.plot(d025["t"], d025["R"], color=BLUE, lw=2,          label="R  0.25 J/cm²")
    ax.plot(d306["t"], d306["R"], color=BLUE, lw=2, ls="--", label="R  3.06 J/cm²")
    axr.plot(d025["t"], d025["ne"], color=ORANGE, lw=2,          label="ne 0.25 J/cm²")
    axr.plot(d306["t"], d306["ne"], color=ORANGE, lw=2, ls="--", label="ne 3.06 J/cm²")

    ax.set_xlabel("Time (ps)", fontsize=12)
    ax.set_ylabel("Reflectivity", color=BLUE, fontsize=12)
    axr.set_ylabel("Carrier density (10²¹ cm⁻³)", color=ORANGE, fontsize=12)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1)
    axr.set_ylim(0, 7)
    ax.tick_params(axis="y", labelcolor=BLUE)
    axr.tick_params(axis="y", labelcolor=ORANGE)
    ax.set_title("(a) Reflectivity & Carrier Density", fontsize=12)

    all_lines = ax.get_lines() + axr.get_lines()
    ax.legend(all_lines, [l.get_label() for l in all_lines], fontsize=9, loc="center right")

    # ── (b) FCA + キャリア密度 ─────────────────────────────
    ax  = axes[1]
    axr = ax.twinx()

    ax.plot(d025["t"], d025["fca"], color=PURPLE, lw=2,          label="αFCA 0.25 J/cm²")
    ax.plot(d306["t"], d306["fca"], color=PURPLE, lw=2, ls="--", label="αFCA 3.06 J/cm²")
    axr.plot(d025["t"], d025["ne"], color=ORANGE, lw=2,          label="ne   0.25 J/cm²")
    axr.plot(d306["t"], d306["ne"], color=ORANGE, lw=2, ls="--", label="ne   3.06 J/cm²")

    ax.set_xlabel("Time (ps)", fontsize=12)
    ax.set_ylabel("FCA coefficient (µm⁻¹)", color=PURPLE, fontsize=12)
    axr.set_ylabel("Carrier density (10²¹ cm⁻³)", color=ORANGE, fontsize=12)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 50)
    axr.set_ylim(0, 7)
    ax.tick_params(axis="y", labelcolor=PURPLE)
    axr.tick_params(axis="y", labelcolor=ORANGE)
    ax.set_title("(b) FCA Coefficient & Carrier Density", fontsize=12)

    all_lines = ax.get_lines() + axr.get_lines()
    ax.legend(all_lines, [l.get_label() for l in all_lines], fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"保存: {out_path}")


def _print_summary(label: str, d: dict) -> None:
    idx_max = d["ne"].argmax()
    t18_idx = np.argmin(np.abs(d["t"] - 1.8))
    print(f"  ne_max   = {d['ne'].max():.4f} × 10²¹ cm⁻³  @ t = {d['t'][idx_max]:.3f} ps")
    print(f"  ne @ 1.8ps = {d['ne'][t18_idx]:.4f} × 10²¹ cm⁻³")
    print(f"  R  @ t=0 = {d['R'][0]:.4f}")
    print(f"  R  peak  = {d['R'].max():.4f}  @ t = {d['t'][d['R'].argmax()]:.3f} ps")
    print(f"  αFCA peak = {d['fca'].max():.4f} µm⁻¹  @ t = {d['t'][d['fca'].argmax()]:.3f} ps")
    print(f"  αFCA @ 1.8ps = {d['fca'][t18_idx]:.4f} µm⁻¹")


if __name__ == "__main__":
    print(f"N_Z={N_Z}  DZ={DZ*1e7:.0f}nm  T_END=6ps(論文時間)  m_eff=me")
    print()

    print("=== F = 0.25 J/cm² ===")
    t0   = time.time()
    d025 = run_simulation(0.25)
    print(f"  実行時間: {time.time()-t0:.1f}s")
    _print_summary("0.25", d025)

    print()
    print("=== F = 3.06 J/cm² ===")
    t0   = time.time()
    d306 = run_simulation(3.06)
    print(f"  実行時間: {time.time()-t0:.1f}s")
    _print_summary("3.06", d306)

    print()
    print("── 論文 Figure 5 比較 ─────────────────────────")
    print("論文 F=0.25 J/cm²: ne_max ≈ 0.6×10²¹ cm⁻³, R→0.85, αFCA peak ≈ 15 µm⁻¹(?)")
    print("論文 F=3.06 J/cm²: ne_max ≈ 6×10²¹ cm⁻³,  R→0.95, αFCA peak ≈ 50 µm⁻¹(?)")

    out_path = os.path.join(os.path.dirname(__file__), "fig5_output.png")
    plot_figure5(d025, d306, out_path)

def print_drude_diagnostics(label: str, d: dict) -> None:
    """Drude診断情報を表示する。"""

    idx_fca = np.argmax(d["fca"])
    idx_ne = np.argmax(d["ne"])

    print()
    print(f"=== Drude Diagnostic ({label}) ===")

    print(
        f"FCA peak       = {d['fca'][idx_fca]:.4f} µm⁻¹"
    )
    print(
        f"t(FCA peak)    = {d['t'][idx_fca]:.3f} ps"
    )

    print(
        f"tau_min        = {d['tau_min_fs'][idx_fca]:.4f} fs"
    )
    print(
        f"tau_max        = {d['tau_max_fs'][idx_fca]:.4f} fs"
    )

    print(
        f"nu_max         = {d['nu_max'][idx_fca]:.4e} s⁻¹"
    )
    print(
        f"nu/omega       = {d['nu_over_omega'][idx_fca]:.4e}"
    )

    print()

    print(
        f"ne peak        = {d['ne'][idx_ne]:.4f} ×10²¹ cm⁻³"
    )
    print(
        f"t(ne peak)     = {d['t'][idx_ne]:.3f} ps"
    )

    print(
        f"tau_min@ne     = {d['tau_min_fs'][idx_ne]:.4f} fs"
    )
    print(
        f"tau_max@ne     = {d['tau_max_fs'][idx_ne]:.4f} fs"
    )
    print(
        f"nu/omega@ne    = {d['nu_over_omega'][idx_ne]:.4e}"
    )

    print(
        f"global tau_min = {d['tau_min_fs'].min():.4f} fs"
    )
    print(
        f"global tau_max = {d['tau_max_fs'].max():.4f} fs"
    )
    print(
        f"global max(nu/omega) = "
        f"{d['nu_over_omega'].max():.4e}"
    )