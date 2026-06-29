"""modules/material_properties — シリコンの物性パラメータ計算。

このモジュールは純粋な物性計算を提供する。Layer構造は持たず、
全ての関数は純粋関数（副作用なし）として実装される。

主な責務:
- 温度依存パラメータの計算（Eg, α_SPA, Ce, Cl, Ke, Kl, τ_e）
- 固相・液相での物性切り替え
- 物理量の偏微分（∂Eg/∂ne）
- 物理定数の一元管理（PHYSICAL, SILICON, LASER_1030NM）

使用例:
    from modules.material_properties import public as mat_props
    from modules.material_properties.constants import PHYSICAL, SILICON, LASER_1030NM
    
    Eg = mat_props.compute_bandgap(Tl, ne, phase_state)
    alpha = mat_props.compute_alpha_spa(Tl, phase_state)
    k_B = PHYSICAL.k_B
"""

from modules.material_properties.public import (
    compute_alpha_spa,
    compute_bandgap,
    compute_bandgap_derivative,
    compute_bandgap_derivative_tl,
    compute_diffusion_coefficient,
    compute_electron_lattice_coupling,
    compute_impact_ionization_rate,
    compute_thermal_capacity_electron,
    compute_thermal_capacity_lattice,
    compute_thermal_conductivity_electron,
    compute_thermal_conductivity_lattice,
    compute_tau_e,
)

from modules.material_properties.constants import (
    PHYSICAL,
    SILICON,
    LASER_1030NM,
)

from modules.material_properties.unit_conversions import (
    convert_cm_to_nm,
    convert_nm_to_cm,
    convert_J_to_erg,
    convert_erg_to_J,
    convert_cm3_to_m3,
    convert_m3_to_cm3,
)

__all__ = [
    "compute_alpha_spa",
    "compute_bandgap",
    "compute_bandgap_derivative",
    "compute_bandgap_derivative_tl",
    "compute_diffusion_coefficient",
    "compute_electron_lattice_coupling",
    "compute_impact_ionization_rate",
    "compute_thermal_capacity_electron",
    "compute_thermal_capacity_lattice",
    "compute_thermal_conductivity_electron",
    "compute_thermal_conductivity_lattice",
    "compute_tau_e",
    "PHYSICAL",
    "SILICON",
    "LASER_1030NM",
    "convert_cm_to_nm",
    "convert_nm_to_cm",
    "convert_J_to_erg",
    "convert_erg_to_J",
    "convert_cm3_to_m3",
    "convert_m3_to_cm3",
]
