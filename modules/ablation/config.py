"""modules/ablation/config.py — パラメータ定義"""

from pydantic import BaseModel

from modules.material_properties.constants import SILICON


class AblationConfig(BaseModel):
    """アブレーション判定のパラメータ。
    
    Attributes:
        T_cr: 臨界温度 [K]
        threshold_fraction: 閾値係数（デフォルト 0.9）
    """
    
    T_cr: float = SILICON.T_cr          # 7925.0 K
    threshold_fraction: float = 0.9     # 0.9 × T_cr で判定
    
    @property
    def threshold_temperature(self) -> float:
        """アブレーション閾値温度 [K]。
        
        Returns:
            0.9 × T_cr = 7132.5 K (デフォルト)
        """
        return self.threshold_fraction * self.T_cr
