"""reproduce/fig5_debug.py — Drude モデルチェーンの段階的デバッグ。

n_e → τ_e → ω_p/ω → ε → k → α_FCA → S → Te の全ステップを時系列でプロット。
どのステップまで想定通りかを目視で確認する。

実行:
  docker compose run --rm sim python reproduce/fig5_debug.py
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
from modules import material_properties

# ─────────────────────────────────────────────────────────────
# ▼ ここを変えてパルスを調整
PULSE_DURATION = 421e-15   # パルス幅 FWHM [s]
T_PEAK_PS      = 1.684     # レーザーピーク時刻 [ps]（論文時間軸）
FLUENCE_025    = 0.25      # 低フルエンス [J/cm²]
FLUENCE_306    = 3.06      # 高フルエンス [J/cm²]
# ─────────────────────────────────────────────────────────────

N_Z          = 50
DZ           = 5e-7
T_PAPER_END  = 6e-12
T_START      = -T_PEAK_PS * 1e-12
T_END        = T_PAPER_END + T_START
DT_MAX       = 1e-17
RECORD_EVERY = 200


def run_simulation(fluence: float) -> dict:
    """指定フルエンスで Drude チェーン全量を時系列収集する。"""
    config = EulerFDMConfig(
        fluence=fluence,
        grid=GridConfig(n_z=N_Z, dz=DZ),
        time=TimeConfig(t_end=T_END, dt_max=DT_MAX, snapshot_interval=RECORD_EVERY),
        initial=InitialCondition(),
    )
    optics_cfg = OpticsConfig(dz=DZ, fluence=fluence, pulse_duration=PULSE_DURATION)
    _, carrier_cfg, ttm_cfg, _ = create_domain_configs(config)

    ne, Te, Tl, phase_state, latent_acc, _, _ = initialize_state_vectors(
        config.grid, config.initial
    )

    omega = optics_cfg.omega
    e2    = optics_cfg.e_charge ** 2
    eps0  = optics_cfg.epsilon_0
    meff  = optics_cfg.m_eff_drude

    t    = T_START
    step = 0
    rec = {k: [] for k in ["t", "ne", "Te", "Tl",
                            "tau_e", "omega_p_ratio",
                            "re_eps", "im_eps",
                            "k_ext", "alpha_fca", "S"]}

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

            # τ_e at surface
            tau_e_surf = material_properties.compute_tau_e(ne, Te, Tl, phase_state)[0]

            # Drude chain at surface (手動再現でチェーンを明示)
            ne_m3   = ne[0] * 1e6                              # cm⁻³ → m⁻³
            nu      = 1.0 / tau_e_surf
            omega_p = np.sqrt(ne_m3 * e2 / (eps0 * meff))     # [rad/s]
            denom   = 1.0 + 1j * nu / omega
            drude   = (ne_m3 * e2) / (eps0 * meff * omega**2 * denom)
            eps     = optics_cfg.epsilon_r.real - drude
            re_e    = float(eps.real)
            im_e    = float(eps.imag)
            k_ext   = float(np.sqrt(max(0.0, (-re_e + np.sqrt(re_e**2 + im_e**2)) / 2.0)))

            rec["t"].append(paper_t)
            rec["ne"].append(ne[0] / 1e20)                    # [10²⁰ cm⁻³]
            rec["Te"].append(Te[0] / 1e3)                     # [10³ K]
            rec["Tl"].append(Tl[0] / 1e3)                     # [10³ K]
            rec["tau_e"].append(tau_e_surf * 1e15)            # [fs]
            rec["omega_p_ratio"].append(omega_p / omega)
            rec["re_eps"].append(re_e)
            rec["im_eps"].append(im_e)
            rec["k_ext"].append(k_ext)
            rec["alpha_fca"].append(opt.alpha_fca[0] / 1e4)  # [µm⁻¹]
            rec["S"].append(opt.source_term[0])               # [W/cm³]

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

    return {k: np.array(v) for k, v in rec.items()}


def _n_crit_cm3(optics_cfg: OpticsConfig) -> float:
    """臨界キャリア密度 n_crit = ε₀ m_eff ω² / e²  [cm⁻³]。"""
    n_crit_m3 = (optics_cfg.epsilon_0 * optics_cfg.m_eff_drude
                 * optics_cfg.omega ** 2 / optics_cfg.e_charge ** 2)
    return n_crit_m3 * 1e-6


def plot_drude_chain(
    d025: dict,
    d306: dict,
    n_crit: float,
    t_peak_ps: float,
    pulse_duration: float,
    out_path: str,
) -> None:
    """8 パネルで Drude チェーン全ステップを描画する。"""
    C025 = "#1565C0"   # 青: F=0.25
    C306 = "#B71C1C"   # 赤: F=3.06
    xlim = (0, 6)

    fig, axes = plt.subplots(4, 2, figsize=(13, 18))
    fig.suptitle(
        f"Drude chain debug  (tp={pulse_duration*1e15:.0f} fs, peak={t_peak_ps:.3f} ps)",
        fontsize=12,
    )

    def _base_plot(ax, key, ylabel, yscale="linear", ref_y=None, ref_label=None):
        ax.plot(d025["t"], d025[key], color=C025, lw=1.5, label="F=0.25")
        ax.plot(d306["t"], d306[key], color=C306, lw=1.5, ls="--", label="F=3.06")
        if ref_y is not None:
            ax.axhline(ref_y, color="gray", lw=1, ls=":", label=ref_label)
        ax.set_xlim(*xlim)
        ax.set_yscale(yscale)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel("Time (ps)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # ① ne
    _base_plot(axes[0, 0], "ne",
               r"$n_e$ ($10^{20}$ cm$^{-3}$)",
               ref_y=n_crit / 1e20,
               ref_label=f"$n_{{crit}}$={n_crit/1e20:.2f}×10²⁰")
    axes[0, 0].set_title("① キャリア密度 ne", fontsize=10)

    # ② τ_e
    _base_plot(axes[0, 1], "tau_e", r"$\tau_e$ (fs)")
    axes[0, 1].set_title("② 電子衝突時間 τ_e", fontsize=10)

    # ③ ω_p/ω
    _base_plot(axes[1, 0], "omega_p_ratio",
               r"$\omega_p / \omega$",
               ref_y=1.0, ref_label=r"critical ($\omega_p=\omega$)")
    axes[1, 0].set_title("③ プラズマ周波数比 ω_p/ω", fontsize=10)

    # ④ Re(ε) [左軸] + Im(ε) [右軸]
    ax  = axes[1, 1]
    axr = ax.twinx()
    ax.plot(d025["t"],  d025["re_eps"], color=C025,  lw=1.5,         label="Re(ε) 0.25")
    ax.plot(d306["t"],  d306["re_eps"], color=C306,  lw=1.5, ls="--",label="Re(ε) 3.06")
    axr.plot(d025["t"], d025["im_eps"], color=C025,  lw=1.5, ls="-.",  label="Im(ε) 0.25")
    axr.plot(d306["t"], d306["im_eps"], color=C306,  lw=1.5, ls=":",   label="Im(ε) 3.06")
    ax.axhline(0, color="gray", lw=1, ls=":")
    ax.set_xlim(*xlim)
    ax.set_xlabel("Time (ps)", fontsize=9)
    ax.set_ylabel("Re(ε)", fontsize=10)
    axr.set_ylabel("Im(ε)", fontsize=10, color="gray")
    lines  = ax.get_lines() + axr.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_title("④ Drude 誘電率 ε", fontsize=10)

    # ⑤ k
    _base_plot(axes[2, 0], "k_ext", "k (extinction coeff.)")
    axes[2, 0].set_title("⑤ 消衰係数 k", fontsize=10)

    # ⑥ α_FCA
    _base_plot(axes[2, 1], "alpha_fca", r"$\alpha_{FCA}$ (µm$^{-1}$)")
    axes[2, 1].set_title("⑥ FCA 吸収係数 α_FCA", fontsize=10)

    # ⑦ S (log)
    ax = axes[3, 0]
    clip = 1e10
    ax.plot(d025["t"], np.clip(d025["S"], clip, None), color=C025, lw=1.5, label="F=0.25")
    ax.plot(d306["t"], np.clip(d306["S"], clip, None), color=C306, lw=1.5, ls="--", label="F=3.06")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_xlabel("Time (ps)", fontsize=9)
    ax.set_ylabel(r"S (W/cm³)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("⑦ 熱源項 S（対数スケール）", fontsize=10)

    # ⑧ Te
    _base_plot(axes[3, 1], "Te", r"$T_e$ ($10^3$ K)")
    axes[3, 1].set_title("⑧ 電子温度 Te", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"保存: {out_path}")


def _print_chain_summary(d: dict, n_crit: float) -> None:
    idx_ne  = np.argmax(d["ne"])
    idx_Te  = np.argmax(d["Te"])
    idx_fca = np.argmax(d["alpha_fca"])
    n_ratio = d["ne"].max() / (n_crit / 1e20)
    print(f"  ne_max       = {d['ne'].max():.3f}×10²⁰  @ t={d['t'][idx_ne]:.3f} ps  "
          f"(n_crit={n_crit/1e20:.3f}×10²⁰, ratio={n_ratio:.2f})")
    print(f"  τ_e @ ne_max = {d['tau_e'][idx_ne]:.3f} fs")
    print(f"  ω_p/ω @ ne_max = {d['omega_p_ratio'][idx_ne]:.3f}")
    print(f"  Re(ε) @ ne_max = {d['re_eps'][idx_ne]:.3f}")
    print(f"  Im(ε) @ ne_max = {d['im_eps'][idx_ne]:.3f}")
    print(f"  k     @ ne_max = {d['k_ext'][idx_ne]:.4f}")
    print(f"  αFCA_max  = {d['alpha_fca'].max():.3f} µm⁻¹  @ t={d['t'][idx_fca]:.3f} ps")
    print(f"  Te_max    = {d['Te'].max():.1f}×10³ K  @ t={d['t'][idx_Te]:.3f} ps")


if __name__ == "__main__":
    optics_ref = OpticsConfig(dz=DZ, fluence=FLUENCE_025, pulse_duration=PULSE_DURATION)
    n_crit = _n_crit_cm3(optics_ref)

    print(f"PULSE_DURATION = {PULSE_DURATION*1e15:.0f} fs,  T_PEAK_PS = {T_PEAK_PS:.3f} ps")
    print(f"n_crit = {n_crit:.3e} cm⁻³ = {n_crit/1e20:.3f}×10²⁰  (m_eff = 0.26 me)")
    print()

    print(f"=== F = {FLUENCE_025} J/cm² ===")
    t0   = time.time()
    d025 = run_simulation(FLUENCE_025)
    print(f"  実行時間: {time.time()-t0:.1f}s")
    _print_chain_summary(d025, n_crit)

    print()
    print(f"=== F = {FLUENCE_306} J/cm² ===")
    t0   = time.time()
    d306 = run_simulation(FLUENCE_306)
    print(f"  実行時間: {time.time()-t0:.1f}s")
    _print_chain_summary(d306, n_crit)

    out_path = os.path.join(os.path.dirname(__file__), "fig5_debug_output.png")
    print()
    plot_drude_chain(d025, d306, n_crit, T_PEAK_PS, PULSE_DURATION, out_path)
