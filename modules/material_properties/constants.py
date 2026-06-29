# modules/material_properties/constants.py
"""全シミュレーション共通の物理定数。

このファイルは不変の物理定数のみを定義する。
シミュレーション固有のパラメータ（フルエンス、グリッド数等）はここに含めない。

使用方法:
    from modules.material_properties.constants import PHYSICAL, SILICON, LASER_1030NM
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class _PhysicalConstants:
    """基本物理定数（CODATA準拠）"""

    k_B: float = 1.381e-23          # ボルツマン定数 [J/K]
    k_B_eV: float = 8.617333e-5    # ボルツマン定数 [eV/K]
    e_charge: float = 1.602e-19     # 電子電荷 [C]
    epsilon_0: float = 8.854e-12    # 真空誘電率 [F/m]
    m_e: float = 9.11e-31          # 電子質量 [kg]
    c_light: float = 3.0e10        # 光速 [cm/s] (CGS)
    h_bar_eV_s: float = 6.582e-16  # ディラック定数 [eV·s]
    h_plank: float = 6.626e-34  # プランク定数 [J·s]


@dataclass(frozen=True)
class _SiliconProperties:
    """シリコンの物性定数"""

    # 温度
    T_room: float = 300.0           # 室温 [K]
    T_m: float = 1687.0             # 融点 [K]
    T_b: float = 3583.0             # 沸点 [K]
    T_cr: float = 7925.0            # 臨界温度 [K]

    # 潜熱
    L_m: float = 4206.0             # 融解潜熱 [J/cm³]
    L_v: float = 32020.0            # 気化潜熱 [J/cm³]

    # 密度
    rho: float = 2.54               # 密度 [g/cm³]

    # 誘電率 (1030 nm, 300 K)
    epsilon_r_real: float = 12.709
    epsilon_r_imag: float = 0.0017149

    # 電子
    n0_valence: float = 5.0e22      # 価電子帯の電子密度 [cm⁻³]

    # 電子衝突時間（旧 Yoffa モデル用・compute_tau_el で継続使用）
    tau_e_base: float = 240e-15     # 固相基底衝突時間 [s]
    tau_e_ne_ref: float = 6.0e20    # 衝突時間の ne 参照密度 [cm⁻³]
    tau_e_liquid: float = 1e-12     # 液相での衝突時間 [s]

    # Drude-Plasma モデル用定数 (compute_tau_e で使用)
    m_eff_optical: float = 2.369e-31   # 光学有効質量 [kg] = 0.26 × m_e
    epsilon_s: float = 1.036e-10       # Si 静電誘電率 [F/m] = 11.7 × ε_0
    tau_e_phonon_base: float = 200e-15 # 室温基準衝突時間 [s] = μ_e(300K) × m_eff_optical / e
    n0_valence_m3: float = 5.0e28      # 価電子帯の電子密度 [m⁻³] (= 5e22 cm⁻³)
    c_ee_base: float = 1.0e16          # 電子-電子散乱基底係数 [s⁻¹] (Yoffa/Chen モデル)

    # キャリア
    gamma_auger: float = 3.8e-31    # Auger再結合係数 [cm⁶/s]
    beta_tpa: float = 9.0           # TPA係数 [cm/GW]

    @property
    def epsilon_r(self) -> complex:
        """複素誘電率"""
        return complex(self.epsilon_r_real, self.epsilon_r_imag)


@dataclass(frozen=True)
class _Laser1030nm:
    """1030 nm レーザー固有の定数"""

    wavelength_nm: float = 1030.0     # 波長 [nm]
    pulse_duration: float = 421e-15   # パルス幅 FWHM [s] (論文条件)

    @property
    def wavelength_cm(self) -> float:
        """波長 [cm]（内部計算用）"""
        return self.wavelength_nm * 1e-7
    
    @property
    def photon_energy_eV(self) -> float:
        """光子エネルギー [eV]（プランク定数と波長から計算）
        
        E = h × ν = h × (c / λ)
        
        Notes:
            - h: プランク定数 [J·s]
            - c: 光速 [cm/s]
            - λ: 波長 [cm]
            - 1 eV = e_charge [J] (電子電荷と同じ値)
        
        Returns:
            hω [eV]
        """
        # h [J·s] × (c [cm/s] / λ [cm]) = E [J]
        E_J = PHYSICAL.h_plank * (PHYSICAL.c_light / self.wavelength_cm)
        # J → eV (1 eV = e_charge J)
        return E_J / PHYSICAL.e_charge


# シングルトンインスタンス（全モジュールがこれを参照）
PHYSICAL = _PhysicalConstants()
SILICON = _SiliconProperties()
LASER_1030NM = _Laser1030nm()
