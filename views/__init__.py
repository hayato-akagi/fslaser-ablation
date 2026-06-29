"""views パッケージ — データ保存・可視化

このパッケージは CLAUDE.md の規約に従い、modules/ の外側に配置された
最上位プレゼンテーション層。物理計算は一切含まず、
SimulationResult の永続化とグラフ生成のみを担当する。

使用方法:
    from modules.euler_fdm import run_simulation, EulerFDMConfig
    from views.io import save_result
    from views.plotting import plot_single_run
    
    # シミュレーション実行
    config = EulerFDMConfig(fluence=1.5)
    result = run_simulation(config)
    
    # 結果保存
    run_dir = save_result(result, config)
    
    # グラフ生成
    plot_single_run(run_dir)
"""

from pathlib import Path

__all__ = [
    "save_result",
    "load_result",
    "plot_single_run",
    "plot_fluence_comparison",
]


# 遅延インポート（循環依存回避）
def __getattr__(name: str):
    if name == "save_result":
        from views.io import save_result
        return save_result
    elif name == "load_result":
        from views.io import load_result
        return load_result
    elif name == "plot_single_run":
        from views.plotting import plot_single_run
        return plot_single_run
    elif name == "plot_fluence_comparison":
        from views.plotting import plot_fluence_comparison
        return plot_fluence_comparison
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
