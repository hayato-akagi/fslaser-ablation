"""reproduce/fig6_debug.py — 4大デバッグチェックポイント。

シミュレーション内部の中間変数を収集・可視化し、Te 上昇不足の根本原因を特定する。

  CP1: S(z=0, t) の絶対値スケールチェック           [W/cm³]
  CP2: α_ext 内訳（α_SPA / α_FCA / β×I）と τ_e / ν 内訳  [cm⁻¹ / s⁻¹]
  CP3: Ce の異常値チェック（数値不安定の予兆検出）  [J/(cm³·K)]
  CP4: エネルギー保存チェック（∫S dz ≈ I(0) − I(L)）  [W/cm²]

実行:
  docker compose run --rm sim python reproduce/fig6_debug.py
"""

import os
import sys
import time as _time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")

from modules import material_properties
from modules.carrier.config import CarrierConfig
from modules.carrier.public import advance_carrier_density
from modules.euler_fdm.config import EulerFDMConfig, GridConfig, InitialCondition, TimeConfig
from modules.euler_fdm.solver import compute_safe_dt, create_domain_configs, initialize_state_vectors
from modules.material_properties.constants import PHYSICAL
from modules.material_properties.drude_plasma import (
    compute_nu_ee,
    compute_nu_ei_spitzer,
    compute_nu_phonon,
)
from modules.optics.config import OpticsConfig
from modules.optics.public import OpticsResult, compute_laser_field
from modules.ttm.config import TTMConfig
from modules.ttm.public import advance_temperatures

# ─────────────────────────────────────────────────────────────
PULSE_DURATION = 421e-15   # パルス幅 FWHM [s]
T_PEAK_PS      = 1.5       # レーザーピーク時刻 [ps]（論文時間軸）
FLUENCE        = 0.25      # フルエンス [J/cm²]
# ─────────────────────────────────────────────────────────────

N_Z          = 20
DZ           = 5e-7
T_PAPER_END  = 6e-12
T_START      = -T_PEAK_PS * 1e-12
T_END        = T_PAPER_END + T_START
DT_MAX       = 1e-17
RECORD_EVERY = 50


def run_debug_simulation(fluence: float) -> dict:
    """診断データを収集しながらシミュレーションを実行する。"""
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

    t      = T_START
    step   = 0
    records: list[dict] = []

    while t < T_END:
        opt = compute_laser_field(
            ne=ne, Tl=Tl, Te=Te, phase_state=phase_state, t=t, config=optics_cfg
        )
        S_max = float(np.abs(opt.source_term).max())
        dt = (
            DT_MAX if S_max < 1e6
            else compute_safe_dt(
                Te=Te, Tl=Tl, ne=ne, dz=DZ, dt_max=DT_MAX,
                source_term=opt.source_term,
            )
        )

        car = advance_carrier_density(
            ne=ne, intensity=opt.intensity, Te=Te, Tl=Tl,
            phase_state=phase_state, dt=dt, config=carrier_cfg,
        )

        if step % RECORD_EVERY == 0:
            records.append(
                _collect_snapshot(ne, Te, Tl, phase_state, opt, t, optics_cfg, car.dne_dt, ttm_cfg, carrier_cfg)
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

    return _aggregate_records(records)


def _collect_snapshot(
    ne: NDArray[np.float64],
    Te: NDArray[np.float64],
    Tl: NDArray[np.float64],
    phase_state: NDArray[np.int32],
    opt: OpticsResult,
    t: float,
    optics_cfg: OpticsConfig,
    dne_dt: NDArray[np.float64],
    ttm_cfg: TTMConfig,
    carrier_cfg: CarrierConfig,
) -> dict:
    """1ステップ分の診断データを z=0（表面）で収集する。

    スナップショットは advance_carrier_density 直後に取得する。
    これにより dne_dt（キャリア生成率）を同じ pre-update 状態で比較できる。
    """
    z0 = 0

    alpha_spa = material_properties.compute_alpha_spa(Tl, phase_state)
    tau_e     = material_properties.compute_tau_e(ne, Te, Tl, phase_state)
    nu_ph     = compute_nu_phonon(Tl)
    nu_ei_arr = compute_nu_ei_spitzer(ne, Te)
    nu_ee_arr = compute_nu_ee(ne, Te)
    Ce        = material_properties.compute_thermal_capacity_electron(Te, ne, phase_state)
    Eg        = material_properties.compute_bandgap(Tl, ne, phase_state)

    # k_ext: alpha_fca = 4π k / λ の逆算 [無次元]
    lambda_cm  = optics_cfg.wavelength_nm * 1e-7
    k_ext_z0   = float(opt.alpha_fca[z0]) * lambda_cm / (4.0 * np.pi)

    # キャリア冷却項: -(Eg + 3 kB Te) × dne_dt [W/cm³]
    # TTM 電子方程式で S と逆符号で競合する項（符号付きで収集）
    cooling_coeff_eV = float(Eg[z0]) + 3.0 * PHYSICAL.k_B_eV * float(Te[z0])
    cooling_term_z0  = -(cooling_coeff_eV * PHYSICAL.e_charge) * float(dne_dt[z0])

    # dn_e/dt の5項分解 at z=0
    I_z0          = float(opt.intensity[z0])
    hw_J          = carrier_cfg.photon_energy_J
    theta         = material_properties.compute_impact_ionization_rate(Te, Eg, carrier_cfg.k_B_eV)
    D0            = material_properties.compute_diffusion_coefficient(Tl, carrier_cfg.T_room)
    D0_half_z0    = 0.5 * (D0[0] + D0[1])
    spa_z0  = float(alpha_spa[z0]) * I_z0 / hw_J
    tpa_z0  = carrier_cfg.beta_cgs * I_z0**2 / (2.0 * hw_J)
    auger_z0   = -carrier_cfg.gamma * float(ne[z0])**3
    impact_z0  = float(theta[z0]) * float(ne[z0])
    diffusion_z0 = D0_half_z0 * (float(ne[1]) - float(ne[z0])) / (DZ * DZ)

    # キャリアエネルギー項の内訳: _compute_carrier_energy_term の term_ne / term_tl
    dEg_dne  = material_properties.compute_bandgap_derivative(ne, phase_state)
    dEg_dTl  = material_properties.compute_bandgap_derivative_tl(Tl, phase_state)
    Cl       = material_properties.compute_thermal_capacity_lattice(Tl, phase_state)
    G        = material_properties.compute_electron_lattice_coupling(Ce, ne, phase_state)
    Kl       = material_properties.compute_thermal_conductivity_lattice(
        Tl, phase_state, ttm_cfg.T_m
    )
    # dTl/dt at z=0 (Neumann BC: ghost cell で Kl_{-1/2} = 0)
    Kl_half_z0  = 0.5 * (Kl[0] + Kl[1])
    kl_diff_z0  = Kl_half_z0 * (Tl[1] - Tl[0]) / (DZ * DZ)
    rhs_l_z0    = float(G[z0]) * (float(Te[z0]) - float(Tl[z0])) + float(kl_diff_z0)
    dTl_dt_z0   = rhs_l_z0 / float(max(Cl[z0], 1e-30))

    eV_to_J     = 1.602e-19
    coeff_ne_z0 = (
        float(Eg[z0])
        + 3.0 * ttm_cfg.k_B_eV * float(Te[z0])
        + float(ne[z0]) * float(dEg_dne[z0])
    ) * eV_to_J
    term_ne_z0  = -coeff_ne_z0 * float(dne_dt[z0])
    term_tl_z0  = -float(ne[z0]) * float(dEg_dTl[z0]) * eV_to_J * dTl_dt_z0

    return {
        "t_ps":         (t - T_START) * 1e12,
        # CP1
        "S_z0":         float(opt.source_term[z0]),
        "I_z0":         float(opt.intensity[z0]),
        "cooling_term": cooling_term_z0,
        # CP2 — α内訳（いずれも有効吸収係数として比較可能な [cm⁻¹] 単位）
        "alpha_spa":    float(alpha_spa[z0]),
        "alpha_fca":    float(opt.alpha_fca[z0]),
        "beta_I":       float(optics_cfg.beta_cgs * opt.intensity[z0]),
        # CP2 — τ_e / k_ext / ν内訳
        "tau_e_fs":     float(tau_e[z0]) * 1e15,
        "k_ext":        k_ext_z0,
        "nu_ph":        float(nu_ph[z0]),
        "nu_ei":        float(nu_ei_arr[z0]),
        "nu_ee":        float(nu_ee_arr[z0]),
        # CP3
        "Ce":           float(Ce[z0]),
        # CP4
        "integrated_S": float(np.sum(opt.source_term) * DZ),
        "I_diff":       float(opt.intensity[z0] - opt.intensity[-1]),
        # 参考: 温度・密度
        "Te":           float(Te[z0]),
        "ne":           float(ne[z0]),
        "Tl":           float(Tl[z0]),
        # キャリアエネルギー項内訳
        "dne_dt":       float(dne_dt[z0]),
        "Eg_z0":        float(Eg[z0]),
        "term_ne":      term_ne_z0,
        "term_tl":      term_tl_z0,
        # dn_e/dt 5項分解
        "spa":          spa_z0,
        "tpa":          tpa_z0,
        "auger":        auger_z0,
        "impact":       impact_z0,
        "diffusion":    diffusion_z0,
    }


def _aggregate_records(records: list[dict]) -> dict:
    """list[dict] → dict[str, ndarray] に変換する。"""
    keys = records[0].keys()
    return {k: np.array([r[k] for r in records]) for k in keys}


def _print_diagnostics(d: dict) -> None:
    """主要な診断値をコンソールに出力する。"""
    idx_peak = int(np.argmax(d["I_z0"]))
    t_peak   = d["t_ps"][idx_peak]

    print("=" * 60)
    print(f"  パルスピーク: t = {t_peak:.3f} ps  (記録点数: {len(d['t_ps'])})")
    print("=" * 60)

    print("\n[CP1] S(z=0) vs キャリア冷却項 スケール")
    S_peak   = d["S_z0"][idx_peak]
    cool     = d["cooling_term"][idx_peak]
    net      = S_peak + cool
    print(f"  S @ peak             = {S_peak:.3e} W/cm³")
    print(f"  S_max (全期間)       = {d['S_z0'].max():.3e} W/cm³")
    print(f"  I(z=0) @ peak        = {d['I_z0'][idx_peak]:.3e} W/cm²")
    print(f"  冷却項 @ peak        = {cool:.3e} W/cm³  (負 = 電子を冷やす)")
    print(f"  S + 冷却項 (正味)    = {net:.3e} W/cm³")

    print("\n[CP2a] α_ext 内訳 @ peak")
    alpha_s  = d["alpha_spa"][idx_peak]
    alpha_f  = d["alpha_fca"][idx_peak]
    beta_i   = d["beta_I"][idx_peak]
    dominant = max(
        [("α_SPA", alpha_s), ("α_FCA", alpha_f), ("β×I", beta_i)],
        key=lambda x: abs(x[1]),
    )
    print(f"  α_SPA = {alpha_s:.3e} cm⁻¹")
    print(f"  α_FCA = {alpha_f:.3e} cm⁻¹")
    print(f"  β×I   = {beta_i:.3e} cm⁻¹")
    print(f"  → 支配項: {dominant[0]}")

    print("\n[CP2b] τ_e / k_ext / ν 内訳 @ peak")
    nu_total = d["nu_ph"][idx_peak] + d["nu_ei"][idx_peak] + d["nu_ee"][idx_peak]
    print(f"  τ_e      = {d['tau_e_fs'][idx_peak]:.3f} fs")
    print(f"  k_ext    = {d['k_ext'][idx_peak]:.4e}  (消衰係数、無次元)")
    print(f"  ν_phonon = {d['nu_ph'][idx_peak]:.3e} s⁻¹")
    print(f"  ν_ei     = {d['nu_ei'][idx_peak]:.3e} s⁻¹")
    print(f"  ν_ee     = {d['nu_ee'][idx_peak]:.3e} s⁻¹")
    print(f"  ν_total  = {nu_total:.3e} s⁻¹")

    print("\n[CP3] Ce 異常値チェック")
    idx_min_Ce = int(np.argmin(d["Ce"]))
    print(f"  Ce @ peak = {d['Ce'][idx_peak]:.3e} J/(cm³·K)")
    print(f"  Ce_min    = {d['Ce'][idx_min_Ce]:.3e} J/(cm³·K)"
          f"  @ t = {d['t_ps'][idx_min_Ce]:.3f} ps")
    print(f"  Ce_max    = {d['Ce'].max():.3e} J/(cm³·K)")

    print("\n[CP4] エネルギー保存チェック @ peak")
    int_S  = d["integrated_S"][idx_peak]
    I_diff = d["I_diff"][idx_peak]
    err    = abs(int_S - I_diff) / (abs(I_diff) + 1e-30) * 100.0
    print(f"  ∫S dz       = {int_S:.3e} W/cm²")
    print(f"  I(0) − I(L) = {I_diff:.3e} W/cm²")
    print(f"  誤差率       = {err:.2f}%")

    print("\n[CP5] キャリアエネルギー項 内訳 @ peak  (z=0, 電子系 RHS への寄与)")
    print(f"  ne       = {d['ne'][idx_peak]:.3e} cm⁻³")
    print(f"  dne_dt   = {d['dne_dt'][idx_peak]:.3e} cm⁻³/s")
    print(f"  Eg       = {d['Eg_z0'][idx_peak]:.4f} eV")
    print(f"  Te       = {d['Te'][idx_peak]:.2f} K")
    print(f"  term_ne  = {d['term_ne'][idx_peak]:.3e} W/cm³  [-(Eg + 3kBTe + ne·∂Eg/∂ne)·dne/dt]")
    print(f"  term_tl  = {d['term_tl'][idx_peak]:.3e} W/cm³  [-ne·∂Eg/∂Tl·dTl/dt]")

    print("\n[CP6] dn_e/dt 5項分解 @ peak  (z=0)")
    spa_p    = d["spa"][idx_peak]
    tpa_p    = d["tpa"][idx_peak]
    auger_p  = d["auger"][idx_peak]
    impact_p = d["impact"][idx_peak]
    diff_p   = d["diffusion"][idx_peak]
    total_p  = spa_p + tpa_p + auger_p + impact_p + diff_p
    print(f"  SPA  generation     = {spa_p:+.3e} cm⁻³/s   [α_SPA·I / hω]")
    print(f"  TPA  generation     = {tpa_p:+.3e} cm⁻³/s   [β·I² / (2hω)]")
    print(f"  Auger recombination = {auger_p:+.3e} cm⁻³/s   [-γ·ne³]")
    print(f"  Impact ionization   = {impact_p:+.3e} cm⁻³/s   [θ·ne]")
    print(f"  Diffusion           = {diff_p:+.3e} cm⁻³/s   [∇(D₀∇ne)]")
    print(f"  ── total (dne_dt)   = {total_p:+.3e} cm⁻³/s")
    print(f"  I(z=0) @ peak       = {d['I_z0'][idx_peak]:.3e} W/cm²")
    print()


def _add_vline(ax: plt.Axes, t_peak: float) -> None:
    """パルスピーク時刻の参照縦線を追加する。"""
    ax.axvline(t_peak, color="gray", ls="--", lw=1.0, alpha=0.7)


def plot_debug(d: dict, out_path: str) -> None:
    """4大チェックポイントを 3×2 サブプロットで可視化する。"""
    idx_peak = int(np.argmax(d["I_z0"]))
    t_peak   = d["t_ps"][idx_peak]
    t        = d["t_ps"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle(
        f"Debug Checkpoints — F = {FLUENCE} J/cm²,  "
        f"tp = {PULSE_DURATION * 1e15:.0f} fs,  peak = {t_peak:.2f} ps",
        fontsize=12,
    )

    _plot_cp1_source(axes[0, 0], t, d, t_peak)
    _plot_cp2a_alpha(axes[0, 1], t, d, t_peak)
    _plot_cp2b_tau_e(axes[1, 0], t, d, t_peak)
    _plot_cp2c_nu(   axes[1, 1], t, d, t_peak)
    _plot_cp3_ce(    axes[2, 0], t, d, t_peak)
    _plot_cp4_energy(axes[2, 1], t, d, t_peak)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


def _plot_cp1_source(ax: plt.Axes, t: np.ndarray, d: dict, t_peak: float) -> None:
    """CP1: S vs carrier cooling term competition and I_surface."""
    axr = ax.twinx()
    ax.plot(t, d["S_z0"],         color="crimson",   lw=2,   label="S(z=0)  [heat source]")
    ax.plot(t, d["cooling_term"], color="dodgerblue", lw=2,   label="cooling term(z=0)  [W/cm3]")
    net = d["S_z0"] + d["cooling_term"]
    ax.plot(t, net,               color="black",     lw=1.5, ls=":", label="S + cooling (net)")
    axr.plot(t, d["I_z0"],        color="royalblue", lw=1.2, ls="--", label="I(z=0) [W/cm2]")
    ax.axhline(0, color="gray", lw=0.8, ls="-")
    _add_vline(ax, t_peak)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("[W/cm3]")
    axr.set_ylabel("I [W/cm2]", color="royalblue")
    ax.set_title("[CP1] Source S vs carrier cooling term (net electron heating)")
    ax.legend( loc="upper left",  fontsize=7)
    axr.legend(loc="upper right", fontsize=8)


def _plot_cp2a_alpha(ax: plt.Axes, t: np.ndarray, d: dict, t_peak: float) -> None:
    """CP2a: dominant absorption mechanism check (log)."""
    ax.semilogy(t, np.maximum(d["alpha_spa"], 1e-3), lw=2, label=r"$\alpha_{SPA}$",     color="steelblue")
    ax.semilogy(t, np.maximum(d["alpha_fca"], 1e-3), lw=2, label=r"$\alpha_{FCA}$",     color="tomato")
    ax.semilogy(t, np.maximum(d["beta_I"],    1e-3), lw=2, label=r"$\beta \times I$",   color="seagreen")
    _add_vline(ax, t_peak)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel(r"Effective absorption coeff. [cm$^{-1}$]")
    ax.set_title(r"[CP2a] $\alpha_{ext}$ breakdown — dominant mechanism (log)")
    ax.legend(fontsize=8)


def _plot_cp2b_tau_e(ax: plt.Axes, t: np.ndarray, d: dict, t_peak: float) -> None:
    """CP2b: electron collision time tau_e time series (log)."""
    ax.semilogy(t, np.maximum(d["tau_e_fs"], 1e-4), lw=2, color="darkorchid", label=r"$\tau_e$")
    ax.axhline(0.5, color="red",  ls=":",  lw=1.5, label="Ioffe-Regel limit (0.5 fs)")
    ax.axhline(200, color="gray", ls="--", lw=1.0, label="tau_0 = 200 fs (room temp)")
    _add_vline(ax, t_peak)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel(r"$\tau_e$ [fs]")
    ax.set_title(r"[CP2b] Electron collision time $\tau_e$ (log)")
    ax.legend(fontsize=8)


def _plot_cp2c_nu(ax: plt.Axes, t: np.ndarray, d: dict, t_peak: float) -> None:
    """CP2c: scattering frequency nu breakdown — dominant mechanism check (log)."""
    nu_total = d["nu_ph"] + d["nu_ei"] + d["nu_ee"]
    ax.semilogy(t, np.maximum(d["nu_ph"],  1.0), lw=2,   label=r"$\nu_{phonon}$",        color="steelblue")
    ax.semilogy(t, np.maximum(d["nu_ei"],  1.0), lw=2,   label=r"$\nu_{ei}$ (Spitzer)",  color="tomato")
    ax.semilogy(t, np.maximum(d["nu_ee"],  1.0), lw=2,   label=r"$\nu_{ee}$ (Yoffa/Chen)", color="seagreen")
    ax.semilogy(t, np.maximum(nu_total,    1.0), lw=1.5, ls="--", label=r"$\nu_{total}$", color="black")
    _add_vline(ax, t_peak)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel(r"$\nu$ [s$^{-1}$]")
    ax.set_title(r"[CP2c] Scattering freq. $\nu$ breakdown — rise of $\nu_{ee}$ (log)")
    ax.legend(fontsize=8)


def _plot_cp3_ce(ax: plt.Axes, t: np.ndarray, d: dict, t_peak: float) -> None:
    """CP3: Ce time series (log) — division-by-zero risk detection."""
    ax.semilogy(t, np.maximum(d["Ce"], 1e-40), lw=2, color="darkorange", label=r"$C_e$(z=0)")
    ax.axhline(1e-10, color="red", ls=":", lw=1.5, label="Warning zone < 1e-10 J/(cm3 K)")
    _add_vline(ax, t_peak)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel(r"$C_e$ [J/(cm$^3$ K)]")
    ax.set_title(r"[CP3] Electron heat capacity $C_e$ — anomaly / div-by-zero risk (log)")
    ax.legend(fontsize=8)


def _plot_cp4_energy(ax: plt.Axes, t: np.ndarray, d: dict, t_peak: float) -> None:
    """CP4: energy conservation check — int S dz vs I(0)-I(L)."""
    ax.plot(t, d["integrated_S"], lw=2,   label=r"$\int S\,dz$",   color="darkred")
    ax.plot(t, d["I_diff"],       lw=1.5, ls="--", label=r"$I(0) - I(L)$", color="navy")
    _add_vline(ax, t_peak)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("[W/cm$^2$]")
    ax.set_title(r"[CP4] Energy conservation: $\int S\,dz$ vs $I(0) - I(L)$")
    ax.legend(fontsize=8)


if __name__ == "__main__":
    print(f"PULSE_DURATION = {PULSE_DURATION * 1e15:.0f} fs")
    print(f"T_PEAK_PS      = {T_PEAK_PS:.3f} ps")
    print(f"FLUENCE        = {FLUENCE} J/cm²")
    print(f"N_Z = {N_Z},  DZ = {DZ:.1e} cm,  RECORD_EVERY = {RECORD_EVERY}")
    print()

    t0 = _time.time()
    d  = run_debug_simulation(FLUENCE)
    print(f"実行時間: {_time.time() - t0:.1f}s\n")

    _print_diagnostics(d)

    out_path = os.path.join(os.path.dirname(__file__), "fig6_debug_output.png")
    plot_debug(d, out_path)
