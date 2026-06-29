"""reproduce/test_beta_sensitivity.py — TPA 係数 β の感度テスト。

β を 1/8 〜 2x に変えて ne の立ち上がりタイミングがどう変わるかを確認する。
目的: baseline 0.85 ps → 論文 1.1 ps に必要な β の縮小量を特定する。

実行:
  docker run --rm -v "$(pwd):/app" -w /app fslaser-sim python reproduce/test_beta_sensitivity.py
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
from modules.euler_fdm.solver import initialize_state_vectors
from modules.optics.config import OpticsConfig
from modules.carrier.config import CarrierConfig
from modules.ttm.config import TTMConfig
from modules.ablation.config import AblationConfig
from modules.optics.public import compute_laser_field
from modules.carrier.public import advance_carrier_density
from modules.ttm.public import advance_temperatures
from modules.material_properties.constants import SILICON

# ─────────────────────────────────────────
# 設定
# ─────────────────────────────────────────
N_Z          = 50
DZ           = 5e-7
T_END        = 2.737e-12
DT_MAX       = 1e-15       # 1 fs
RECORD_EVERY = 50
FLUENCE      = 3.06

ONSET_THRESHOLD = 0.05     # ne > 0.05×10²¹ で立ち上がりと判定

BETA_BASE = SILICON.beta_tpa   # 9.0 cm/GW
BETA_FACTORS = [1/8, 1/4, 1/2, 1, 2]


# ─────────────────────────────────────────
# シミュレーション
# ─────────────────────────────────────────

def run_with_beta(beta_cmGW: float) -> dict:
    """指定した β で F=3.06 J/cm² を シミュレーション。

    Args:
        beta_cmGW: TPA 係数 [cm/GW]
    """
    config = EulerFDMConfig(
        fluence=FLUENCE,
        grid=GridConfig(n_z=N_Z, dz=DZ),
        time=TimeConfig(t_end=T_END, dt_max=DT_MAX, snapshot_interval=RECORD_EVERY),
        initial=InitialCondition(),
    )

    # β を両 Config に明示注入
    optics_cfg  = OpticsConfig(dz=DZ, fluence=FLUENCE, beta_tpa=beta_cmGW)
    carrier_cfg = CarrierConfig(dz=DZ, beta_tpa=beta_cmGW)
    ttm_cfg     = TTMConfig(dz=DZ)

    ne, Te, Tl, phase_state, latent_acc, _, _ = initialize_state_vectors(
        config.grid, config.initial
    )

    t    = config.time.t_start
    step = 0
    times_list = []
    ne_list    = []

    while t < T_END:
        opt = compute_laser_field(
            ne=ne, Tl=Tl, Te=Te, phase_state=phase_state, t=t, config=optics_cfg
        )

        if step % RECORD_EVERY == 0:
            paper_t = (t - config.time.t_start) * 1e12
            times_list.append(paper_t)
            ne_list.append(ne[0])

        car = advance_carrier_density(
            ne=ne, intensity=opt.intensity, Te=Te, Tl=Tl,
            phase_state=phase_state, dt=DT_MAX, config=carrier_cfg,
        )
        ttm = advance_temperatures(
            Te=Te, Tl=Tl, ne=ne, dne_dt=car.dne_dt,
            source_term=opt.source_term, phase_state=phase_state,
            latent_heat_accumulated=latent_acc, dt=DT_MAX, config=ttm_cfg,
        )
        ne, Te, Tl = car.ne, ttm.Te, ttm.Tl
        phase_state, latent_acc = ttm.phase_state, ttm.latent_heat_accumulated
        t    += DT_MAX
        step += 1

    return {
        "t":   np.array(times_list),
        "ne":  np.array(ne_list) / 1e21,
        "beta": beta_cmGW,
    }


# ─────────────────────────────────────────
# 解析補助
# ─────────────────────────────────────────

def onset_time(d: dict, threshold: float = ONSET_THRESHOLD) -> float:
    idx = np.where(d["ne"] >= threshold)[0]
    return float(d["t"][idx[0]]) if len(idx) > 0 else float("nan")


# ─────────────────────────────────────────
# メイン
# ─────────────────────────────────────────

if __name__ == "__main__":
    print(f"F={FLUENCE} J/cm²  β_base={BETA_BASE} cm/GW  DT_MAX={DT_MAX*1e15:.0f}fs")
    print()

    results = {}
    for factor in BETA_FACTORS:
        beta = BETA_BASE * factor
        label = f"β × {factor:.4g}  ({beta:.3g} cm/GW)"
        t0 = time.time()
        d = run_with_beta(beta)
        elapsed = time.time() - t0
        t_on = onset_time(d)
        ne_max = d["ne"].max()
        t_max  = d["t"][np.argmax(d["ne"])]
        results[factor] = d
        print(f"{label}")
        print(f"  実行時間: {elapsed:.1f}s")
        print(f"  立ち上がり: {t_on:.3f} ps  |  ne_max = {ne_max:.3f}×10²¹ @ {t_max:.3f} ps")
        print()

    print("── 立ち上がりまとめ ────────────────────────────────")
    print(f"  {'β [cm/GW]':>14}  {'倍率':>6}  {'onset [ps]':>12}  {'ne_max':>8}")
    for factor in BETA_FACTORS:
        d = results[factor]
        t_on = onset_time(d)
        print(f"  {d['beta']:>14.3g}  {factor:>6.4g}  {t_on:>12.3f}  {d['ne'].max():>8.3f}")
    print(f"  論文観測値                                ~1.100")

    # ─── グラフ ───────────────────────────────────────────
    cmap = plt.cm.plasma
    colors = [cmap(i / (len(BETA_FACTORS) - 1)) for i in range(len(BETA_FACTORS))]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) 全体
    ax = axes[0]
    for factor, color in zip(BETA_FACTORS, colors):
        d = results[factor]
        lw = 2.5 if factor == 1 else 1.5
        ax.plot(d["t"], d["ne"], color=color, lw=lw,
                label=f"β×{factor:.4g} = {d['beta']:.2g} cm/GW")
    ax.axhline(ONSET_THRESHOLD, color="gray", ls=":", lw=1, alpha=0.6,
               label=f"onset 閾値 {ONSET_THRESHOLD}×10²¹")
    ax.set_xlabel("Time (ps)", fontsize=12)
    ax.set_ylabel("Carrier density (10²¹ cm⁻³)", fontsize=12)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 6)
    ax.set_title(f"(a) ne — beta sensitivity (F={FLUENCE} J/cm²)", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")

    # (b) 立ち上がり拡大
    ax2 = axes[1]
    for factor, color in zip(BETA_FACTORS, colors):
        d = results[factor]
        lw = 2.5 if factor == 1 else 1.5
        ax2.plot(d["t"], d["ne"], color=color, lw=lw,
                 label=f"β×{factor:.4g}")
    ax2.axhline(ONSET_THRESHOLD, color="gray", ls=":", lw=1, alpha=0.6)
    ax2.axvline(1.1, color="red", ls="--", lw=1.5, alpha=0.7, label="論文 ~1.1 ps")
    for factor, color in zip(BETA_FACTORS, colors):
        t_on = onset_time(results[factor])
        if not np.isnan(t_on):
            ax2.axvline(t_on, color=color, ls=":", lw=1, alpha=0.5)
    ax2.set_xlabel("Time (ps)", fontsize=12)
    ax2.set_ylabel("Carrier density (10²¹ cm⁻³)", fontsize=12)
    ax2.set_xlim(0.5, 1.5)
    ax2.set_ylim(0, 0.5)
    ax2.set_title("(b) onset 拡大 (0.5〜1.5 ps)", fontsize=12)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "test_beta_sensitivity.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n保存: {out}")
