"""シミュレーション実行スクリプト。

Usage:
    docker compose run --rm sim python run.py
"""

from pathlib import Path

from modules.euler_fdm.config import EulerFDMConfig, GridConfig, TimeConfig
from modules.euler_fdm.public import run_simulation
from views.io import save_result
from views.plotting import plot_single_run


def main() -> None:
    config = EulerFDMConfig(
        fluence=1.5,  # J/cm² — 論文条件
        grid=GridConfig(
            n_z=1000,
            dz=5.0e-7,  # 5 nm in cm
        ),
        time=TimeConfig(
            t_end=20e-12,            # 20 ps
            dt_max=1e-15,            # 1 fs
            snapshot_interval=100,   # 100ステップ毎 ≈ 0.1 ps 間隔
        ),
    )

    print(f"=== Simulation Config ===")
    print(f"  Fluence     : {config.fluence} J/cm²")
    print(f"  Grid        : {config.grid.n_z} points, dz = {config.grid.dz*1e7:.1f} nm")
    print(f"  t_start     : {config.time.t_start*1e12:.3f} ps")
    print(f"  t_end       : {config.time.t_end*1e12:.1f} ps")
    print(f"  dt_max      : {config.time.dt_max*1e15:.1f} fs")
    print(f"  snapshot_int: every {config.time.snapshot_interval} steps")
    print()

    print("Running simulation...")
    result = run_simulation(config)

    print(f"\n=== Result ===")
    print(f"  Total steps       : {result.total_steps}")
    print(f"  Ablation depth    : {result.ablation_depth_nm:.1f} nm")
    print(f"  Final Te(surface) : {result.Te_final[0]:.1f} K")
    print(f"  Final Tl(surface) : {result.Tl_final[0]:.1f} K")
    print(f"  Final ne(surface) : {result.ne_final[0]:.3e} cm⁻³")
    print(f"  Snapshots         : {len(result.time_points)}")
    print()

    print("Saving results...")
    run_dir = save_result(result, config)
    print(f"  Saved to: {run_dir}")

    print("Generating plots...")
    plot_single_run(run_dir)
    print(f"  Plots in: {run_dir / 'plots'}")

    print("\nDone!")


if __name__ == "__main__":
    main()
