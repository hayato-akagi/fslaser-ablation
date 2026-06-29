"""modules/optics/config.py — 光学計算パラメータの型安全な定義

Pydantic BaseModel により、物理パラメータをバリデーション付きで管理する。
YAML等の外部ファイルは使用しない（Python Config 原則）。
"""

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from modules.material_properties.constants import PHYSICAL, SILICON, LASER_1030NM
from modules.material_properties import convert_nm_to_cm


class OpticsConfig(BaseModel):
    """Drude動的光学 + レーザー伝播計算のパラメータ。

    デフォルト値は modules/constants.py から取得する。
    dz と fluence は euler_fdm から注入される（必須フィールド）。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ===== 誘電率関連 =====
    epsilon_r: complex = Field(
        default=SILICON.epsilon_r,
        description="静的複素誘電率（1030nm, 300K）",
    )
    n0_valence: float = Field(
        default=SILICON.n0_valence,
        description="価電子帯の電子密度 [cm⁻³]",
    )

    # ===== 電子・真空の物理定数 =====
    m_e: float = Field(
        default=PHYSICAL.m_e,
        description="自由電子質量 [kg]（参照用）",
    )
    m_eff_drude: float = Field(
        default=0.26 * PHYSICAL.m_e,
        description="Drude モデルの光学有効質量 m* [kg]（Si 伝導有効質量 ≈ 0.26 me）",
    )
    e_charge: float = Field(
        default=PHYSICAL.e_charge,
        description="電子電荷 [C]",
    )
    epsilon_0: float = Field(
        default=PHYSICAL.epsilon_0,
        description="真空誘電率 [F/m]",
    )

    # ===== レーザーパラメータ =====
    wavelength_nm: float = Field(
        default=LASER_1030NM.wavelength_nm,
        description="レーザー波長 [nm]",
    )
    c_light: float = Field(
        default=PHYSICAL.c_light,
        description="光速 [cm/s]",
    )
    fluence: float = Field(
        ...,
        description="レーザーフルエンス [J/cm²] ← euler_fdm から注入",
        gt=0.0,
    )
    pulse_duration: float = Field(
        default=LASER_1030NM.pulse_duration,
        description="パルス幅 FWHM [s]",
    )

    # ===== 吸収係数 =====
    beta_tpa: float = Field(
        default=SILICON.beta_tpa,
        description="2光子吸収係数 [cm/GW]",
    )

    # ===== 電子衝突時間（相依存） =====
    tau_e_base: float = Field(
        default=SILICON.tau_e_base,
        description="固相基底衝突時間 [s]",
    )
    tau_e_ne_ref: float = Field(
        default=SILICON.tau_e_ne_ref,
        description="衝突時間の ne 参照密度 [cm⁻³]",
    )
    tau_e_liquid: float = Field(
        default=SILICON.tau_e_liquid,
        description="液相での衝突時間 [s]",
    )

    # ===== グリッドパラメータ =====
    dz: float = Field(
        ...,
        description="グリッド間隔 [cm] ← euler_fdm から注入",
        gt=0.0,
    )

    # ===== 計算用プロパティ =====
    @property
    def omega(self) -> float:
        """レーザー角振動数 [rad/s]。

        ω = 2πc/λ
        """
        wavelength_cm = convert_nm_to_cm(self.wavelength_nm)
        return 2.0 * np.pi * self.c_light / wavelength_cm

    @property
    def beta_cgs(self) -> float:
        """TPA係数の内部単位（CGS）への変換 [cm·s/erg]。

        β_TPA = 9.0 cm/GW → 9.0×10⁻⁹ cm·s/erg
        """
        return self.beta_tpa * 1e-9
