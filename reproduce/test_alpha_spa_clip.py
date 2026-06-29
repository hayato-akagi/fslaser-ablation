"""reproduce/test_alpha_spa_clip.py — α_SPA 低温クリップの影響テスト。

コードを修正せず monkey-patch で α_SPA(Tl < clip_temp) = 0 を適用し、
ne の立ち上がりタイミングが変わるかを確認する。

目的:
  baseline (~0.8ps) → clipped (~1.1ps?) に遅れれば、立ち上がり早期化の
  原因が「室温での α_SPA ≠ 0」であることが確定する。

実行:
  docker run --rm -v "$(pwd):/app" -w /app fslaser-sim python reproduce/test_alpha_spa_clip.py
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import modules.material_properties as _mat_props
from modules.euler_fdm.config import EulerFDMConfig, GridConfig, TimeConfig, InitialCondition
from modules.euler_fdm.solver import initialize_state_vectors, create_domain_configs
from modules.optics.public import compute_laser_field
from modules.carrier.public import advance_carrier_density
from modules.ttm.public import advance_temperatures

# ─────────────────────────────────────────
# シミュレーション設定
# ─────────────────────────────────────────
N_Z          = 50
DZ           = 5e-7
T_END        = 2.737e-12   # paper t=4ps
DT_MAX       = 1e-15       # 1 fs（速度優先、定性的比較に十分）
RECORD_EVERY = 50
FLUENCE      = 3.06        # F=3.06 のみ（立ち上がり差異が大きいため）

ONSET_THRESHOLD = 0.05     # ne > 0.05×10²¹ で「立ち上がり」と判定


# ─────────────────────────────────────────
# シミュレーションループ
# ─────────────────────────────────────────

def _run_loop() -> dict:
    """シミュレーションループ本体（α_SPA は呼び出し元が差し替え済み）。"""
    config = EulerFDMConfig(
        fluence=FLUENCE,
        grid=GridConfig(n_z=N_Z, dz=DZ),
        time=TimeConfig(t_end=T_END, dt_max=DT_MAX, snapshot_interval=RECORD_EVERY),
        initial=InitialCondition(),
    )
    optics_cfg, carrier_cfg, ttm_cfg, _ = create_domain_configs(config)
    ne, Te, Tl, phase_state, latent_acc, _, _ = initialize_state_vectors(
        config.grid, config.initial
    )

    t    = config.time.t_start
    step = 0
    times_list = []
    ne_list    = []
    tl_list    = []

    while t < T_END:
        opt = compute_laser_field(
            ne=ne, Tl=Tl, Te=Te, phase_state=phase_state, t=t, config=optics_cfg
        )

        if step % RECORD_EVERY == 0:
            paper_t = (t - config.time.t_start) * 1e12
            times_list.append(paper_t)
            ne_list.append(ne[0])
            tl_list.append(Tl[0])

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
        "t":  np.array(times_list),
        "ne": np.array(ne_list) / 1e21,
        "Tl": np.array(tl_list),
    }


def run_baseline() -> dict:
    """標準 α_SPA（温度依存あり）でシミュレーション。"""
    return _run_loop()


def run_clipped(clip_temp: float) -> dict:
    """α_SPA(Tl < clip_temp) = 0 を monkey-patch して実行。"""
    original = _mat_props.compute_alpha_spa

    def _clipped(Tl, phase_state):
        alpha = original(Tl, phase_state)
        alpha[Tl < clip_temp] = 0.0
        return alpha

    _mat_props.compute_alpha_spa = _clipped
    try:
        result = _run_loop()
    finally:
        _mat_props.compute_alpha_spa = original   # 必ず元に戻す

    return result


# ─────────────────────────────────────────
# 解析補助
# ─────────────────────────────────────────

def onset_time(d: dict, threshold: float = ONSET_THRESHOLD) -> float:
    """ne が threshold (×10²¹) を初めて超える時刻 [ps]。"""
    idx = np.where(d["ne"] >= threshold)[0]
    return float(d["t"][idx[0]]) if len(idx) > 0 else float("nan")


def print_summary(label: str, d: dict) -> None:
    t_on  = onset_time(d)
    t_max = d["t"][np.argmax(d["ne"])]
    print(f"  {label}")
    print(f"    立ち上がり (ne>{ONSET_THRESHOLD}×10²¹) : {t_on:.3f} ps")
    print(f"    ne_max = {d['ne'].max():.3f} ×10²¹  @ t = {t_max:.3f} ps")


# ─────────────────────────────────────────
# メイン
# ─────────────────────────────────────────

if __name__ == "__main__":
    print(f"F={FLUENCE} J/cm²  N_Z={N_Z}  DT_MAX={DT_MAX*1e15:.0f}fs")
    print()

    print("=== baseline (α_SPA 温度依存あり) ===")
    t0 = time.time()
    d_base = run_baseline()
    print(f"  実行時間: {time.time()-t0:.1f}s")
    print_summary("baseline", d_base)

    print()
    print("=== clip: α_SPA = 0 (Tl < 500 K) ===")
    t0 = time.time()
    d_500 = run_clipped(500.0)
    print(f"  実行時間: {time.time()-t0:.1f}s")
    print_summary("clip 500K", d_500)

    print()
    print("=== clip: α_SPA = 0 (Tl < 1000 K) ===")
    t0 = time.time()
    d_1000 = run_clipped(1000.0)
    print(f"  実行時間: {time.time()-t0:.1f}s")
    print_summary("clip 1000K", d_1000)

    delay_500  = onset_time(d_500)  - onset_time(d_base)
    delay_1000 = onset_time(d_1000) - onset_time(d_base)
    print()
    print("── 立ち上がり遅延まとめ ────────────────────────")
    print(f"  baseline     : {onset_time(d_base):.3f} ps")
    print(f"  clip 500K    : {onset_time(d_500):.3f} ps  (遅延 {delay_500:+.3f} ps)")
    print(f"  clip 1000K   : {onset_time(d_1000):.3f} ps  (遅延 {delay_1000:+.3f} ps)")
    print(f"  論文の観測値 : ~1.1 ps")

    # ─── グラフ ────────────────────────────────────────────
    ORANGE = "#E86B00"
    BLUE   = "#0072BD"
    GREEN  = "#2E7D32"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) ne 比較
    ax = axes[0]
    ax.plot(d_base["t"], d_base["ne"], color=ORANGE, lw=2,      label="baseline")
    ax.plot(d_500["t"],  d_500["ne"],  color=BLUE,   lw=2, ls="--", label="clip Tl<500K")
    ax.plot(d_1000["t"], d_1000["ne"], color=GREEN,  lw=2, ls=":",  label="clip Tl<1000K")
    for t_on, col in [
        (onset_time(d_base), ORANGE),
        (onset_time(d_500),  BLUE),
        (onset_time(d_1000), GREEN),
    ]:
        if not np.isnan(t_on):
            ax.axvline(t_on, color=col, ls=":", lw=1, alpha=0.7)
    ax.axhline(ONSET_THRESHOLD, color="gray", ls=":", lw=1, alpha=0.5,
               label=f"onset 閾値 {ONSET_THRESHOLD}×10²¹")
    ax.set_xlabel("Time (ps)", fontsize=12)
    ax.set_ylabel("Carrier density (10²¹ cm⁻³)", fontsize=12)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 6)
    ax.set_title("(a) ne — α_SPA clip 比較 (F=3.06 J/cm²)", fontsize=12)
    ax.legend(fontsize=9)

    # (b) 立ち上がり付近の拡大
    ax2 = axes[1]
    ax2.plot(d_base["t"], d_base["ne"], color=ORANGE, lw=2,      label="baseline")
    ax2.plot(d_500["t"],  d_500["ne"],  color=BLUE,   lw=2, ls="--", label="clip Tl<500K")
    ax2.plot(d_1000["t"], d_1000["ne"], color=GREEN,  lw=2, ls=":",  label="clip Tl<1000K")
    ax2.axhline(ONSET_THRESHOLD, color="gray", ls=":", lw=1, alpha=0.5)
    ax2.set_xlabel("Time (ps)", fontsize=12)
    ax2.set_ylabel("Carrier density (10²¹ cm⁻³)", fontsize=12)
    ax2.set_xlim(0.5, 1.5)
    ax2.set_ylim(0, 0.5)
    ax2.set_title("(b) 立ち上がり拡大 (0.5〜1.5 ps)", fontsize=12)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "test_alpha_spa_clip.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n保存: {out}")
