"""modules/optics — 動的光学モジュール（Drude + レーザー伝播）

外部からのアクセスは必ず public.py 経由で行う。
"""

from modules.optics.public import OpticsResult, compute_laser_field
from modules.optics.config import OpticsConfig

__all__ = [
    "OpticsResult",
    "compute_laser_field",
    "OpticsConfig",
]
