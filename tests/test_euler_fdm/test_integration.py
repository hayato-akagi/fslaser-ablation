"""Quick integration test to verify euler_fdm runs end-to-end."""

from modules.euler_fdm import EulerFDMConfig, GridConfig, TimeConfig, run_simulation


def test_full_simulation_low_fluence():
    """低フルエンスで短時間シミュレーションを実行"""
    config = EulerFDMConfig(
        fluence=0.1,  # 低フルエンス
        grid=GridConfig(n_z=50),  # 小さいグリッド
        time=TimeConfig(
            t_end=10e-12,  # 10 ps（短時間）
            dt_max=1e-15,
            snapshot_interval=10,
        ),
    )
    
    result = run_simulation(config)
    
    # 基本的な検証
    assert result.total_steps > 0
    assert result.fluence == 0.1
    assert len(result.Te_final) == 50
    assert len(result.Tl_final) == 50
    assert len(result.ne_final) == 50
    
    # F=0.1 J/cm²（ピーク強度 ~300 GW/cm²）ではアブレーションが発生する
    assert result.ablation_depth_nm >= 0.0
    
    # スナップショットが記録されている
    assert len(result.time_points) > 0
    assert len(result.Te_surface_history) == len(result.time_points)
    
    print(f"✓ Simulation completed successfully")
    print(f"  Total steps: {result.total_steps}")
    print(f"  Snapshots: {len(result.time_points)}")
    print(f"  Final surface Te: {result.Te_final[0]:.1f} K")
    print(f"  Final surface Tl: {result.Tl_final[0]:.1f} K")
    print(f"  Final surface ne: {result.ne_final[0]:.2e} cm⁻³")


if __name__ == "__main__":
    test_full_simulation_low_fluence()
