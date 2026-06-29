"""modules/phase_transition/public — 外部公開API。

fsm.py の関数を外部に公開する。
他モジュールはこのpublic.pyを通じてのみ相転移にアクセスする。
"""

from modules.phase_transition.fsm import apply_phase_transitions

__all__ = ["apply_phase_transitions"]
