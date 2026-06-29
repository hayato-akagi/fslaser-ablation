"""modules/material_properties/unit_conversions.py — 単位変換ユーティリティ

物理シミュレーションで使用する単位変換関数を集約する。
内部計算は CGS 単位系（cm, s, K, erg など）で統一し、
外部出力時に SI 単位系（nm, J など）へ変換する。
"""


def convert_cm_to_nm(depth_cm: float) -> float:
    """深さを cm から nm に変換する。
    
    Args:
        depth_cm: 深さ [cm]
    
    Returns:
        深さ [nm]
    
    Examples:
        >>> convert_cm_to_nm(1e-7)  # 1 nm in cm
        1.0
        >>> convert_cm_to_nm(1e-5)  # 10 μm in cm
        10000.0
    """
    return depth_cm * 1e7


def convert_nm_to_cm(depth_nm: float) -> float:
    """深さを nm から cm に変換する。
    
    Args:
        depth_nm: 深さ [nm]
    
    Returns:
        深さ [cm]
    
    Examples:
        >>> convert_nm_to_cm(1.0)
        1e-07
        >>> convert_nm_to_cm(10000.0)  # 10 μm
        1e-05
    """
    return depth_nm * 1e-7


def convert_J_to_erg(energy_J: float) -> float:
    """エネルギーを J から erg に変換する。
    
    Args:
        energy_J: エネルギー [J]
    
    Returns:
        エネルギー [erg]
    
    Examples:
        >>> convert_J_to_erg(1.0)
        1e7
    """
    return energy_J * 1e7


def convert_erg_to_J(energy_erg: float) -> float:
    """エネルギーを erg から J に変換する。
    
    Args:
        energy_erg: エネルギー [erg]
    
    Returns:
        エネルギー [J]
    
    Examples:
        >>> convert_erg_to_J(1e7)
        1.0
    """
    return energy_erg * 1e-7


def convert_cm3_to_m3(density_cm3: float) -> float:
    """密度を cm⁻³ から m⁻³ に変換する。
    
    Args:
        density_cm3: 密度 [cm⁻³]
    
    Returns:
        密度 [m⁻³]
    
    Examples:
        >>> convert_cm3_to_m3(1e18)
        1e24
    """
    return density_cm3 * 1e6


def convert_m3_to_cm3(density_m3: float) -> float:
    """密度を m⁻³ から cm⁻³ に変換する。
    
    Args:
        density_m3: 密度 [m⁻³]
    
    Returns:
        密度 [cm⁻³]
    
    Examples:
        >>> convert_m3_to_cm3(1e24)
        1e18
    """
    return density_m3 * 1e-6
