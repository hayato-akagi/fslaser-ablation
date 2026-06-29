"""modules/carrier — Configuration and parameters.

キャリア密度モデルの全パラメータを Pydantic モデルで型安全に管理する。
デフォルト値は modules.constants から取得する。
"""

import numpy as np

from pydantic import BaseModel, ConfigDict

from modules.material_properties.constants import LASER_1030NM, PHYSICAL, SILICON
from modules.material_properties import convert_J_to_erg
from modules.material_properties import convert_nm_to_cm

class CarrierConfig(BaseModel):
    """キャリア密度モデルのパラメータ。
    
    Attributes:
        gamma: Auger再結合係数 [cm⁶/s]
        beta_tpa: TPA係数 [cm/GW]（内部で cm/W に変換される）
        photon_energy_eV: 光子エネルギー hω [eV]
        T_room: 室温 [K]
        k_B: ボルツマン定数 [J/K]
        dz: グリッド間隔 [cm]（euler_fdm から注入される）
        
    Notes:
        単位系: I [W/cm²], hω [J], β [cm/W] で統一。
        beta_tpa の単位変換:
        - 入力: 9.0 cm/GW
        - 内部: 9.0e-9 cm/W (1 GW = 10⁹ W)
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # 定数（constants.py から初期化）
    gamma: float = SILICON.gamma_auger
    beta_tpa: float = SILICON.beta_tpa
    photon_energy_eV: float = LASER_1030NM.photon_energy_eV
    T_room: float = SILICON.T_room
    k_B: float = PHYSICAL.k_B
    k_B_eV: float = PHYSICAL.k_B_eV
    omega: float = 2.0 * np.pi * PHYSICAL.c_light / convert_nm_to_cm(LASER_1030NM.wavelength_nm)
    
    # euler_fdm から注入される値
    dz: float
    
    @property
    def beta_cgs(self) -> float:
        """TPA係数を内部単位 [cm/W] に変換する。
        
        Returns:
            β [cm/W]（= [cm·s/J]）
        
        Notes:
            β [cm/GW] → β [cm/W]
            9.0 cm/GW = 9.0 / 10⁹ cm/W = 9.0×10⁻⁹ cm/W
            
            単位検証 (TPA carrier generation):
            β [cm/W] × I² [W²/cm⁴] / hω [J]
            = [W/cm³] / [J] = [W/cm³] / [W·s] = [cm⁻³/s] ✓
            
            単位検証 (optics propagation):
            β [cm/W] × I² [W²/cm⁴] = [W/cm³] = dI/dz ✓
        """
        return self.beta_tpa * 1e-9
    
    @property
    def photon_energy_J(self) -> float:
        """光子エネルギーを J 単位で取得する。
        
        I [W/cm²] = [J/(s·cm²)] と整合する単位系。
        
        Returns:
            hω [J]
        """
        return self.photon_energy_eV * PHYSICAL.e_charge
    
    @property
    def photon_energy_erg(self) -> float:
        """光子エネルギーを erg 単位で取得する（後方互換用）。
        
        Returns:
            hω [erg]
        
        Notes:
            1 J = 1e7 erg なので、1 eV = e_charge [J] = e_charge × 1e7 [erg]
        """
        return convert_J_to_erg(self.photon_energy_eV * PHYSICAL.e_charge)
