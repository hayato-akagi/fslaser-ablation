"""modules/phase_transition — シリコンの相転移状態管理。

このモジュールは相転移の有限状態機械（FSM）を提供する。Layer構造は持たない。

主な責務:
- 相状態の遷移判定（SOLID ⟷ MELTING ⟷ LIQUID ⟷ VAPORIZING ⟷ VAPOR）
- 潜熱の蓄積・消費管理
- 温度範囲の制約適用

使用例:
    from modules.phase_transition import public as phase_trans
    
    Tl_new, state_new, latent_new = phase_trans.apply_phase_transitions(
        Tl, rhs_l, Cl, phase_state, latent_heat_accumulated, dt, config
    )
"""

from modules.phase_transition.public import apply_phase_transitions

__all__ = ["apply_phase_transitions"]
