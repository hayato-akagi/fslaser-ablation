"""modules/optics/solver — レーザー場の計算。

layer1/2/3 を統合し、レーザー強度分布と熱源項を計算する。
物性計算（α_SPA, τ_e）は material_properties モジュールに委譲する。
"""

import numpy as np
from numpy.typing import NDArray

from modules.optics.config import OpticsConfig
from modules.optics.public import OpticsResult
from modules import material_properties
from modules.material_properties import convert_cm3_to_m3, convert_nm_to_cm


def compute_laser_field_sequence(
    ne: NDArray[np.float64],
    Tl: NDArray[np.float64],
    Te: NDArray[np.float64],
    phase_state: NDArray[np.int32],
    t: float,
    config: OpticsConfig,
) -> OpticsResult:
    """9ステップの光学計算シーケンス。

    シーケンス:
    1. SPA吸収係数を取得（material_properties経由）
    2. τ_e を取得（material_properties経由、Drude-Plasma モデル）
    3. Drude 誘電率を計算
    4. 屈折率・消衰係数を計算
    5. 表面反射率を計算
    6. FCA 吸収係数を計算
    7. 表面レーザー強度を計算
    8. z方向強度伝播（ODE前進差分）
    9. 熱源項を計算

    Args:
        ne: キャリア密度分布 [cm⁻³], shape (n_z,)
        Tl: 格子温度分布 [K], shape (n_z,)
        Te: 電子温度分布 [K], shape (n_z,)
        phase_state: 相状態分布 (PhaseState), shape (n_z,)
        t: 現在時刻 [s]
        config: パラメータ

    Returns:
        OpticsResult: I(z), S(z), R, α_FCA
    """
    # === 1. SPA吸収係数を取得 ===
    alpha_spa = material_properties.compute_alpha_spa(Tl, phase_state)

    # === 2. τ_e を取得（Drude-Plasma モデル）===
    tau_e = material_properties.compute_tau_e(ne, Te, Tl, phase_state)
    
    # === 3. Drude 誘電率を計算 ===
    epsilon = _compute_drude_epsilon(ne, tau_e, config)
    
    # === 4. 屈折率・消衰係数を計算 ===
    n_ref = _compute_refractive_index(epsilon)
    k_ext = _compute_extinction_coefficient(epsilon)
    
    # === 5. 表面反射率を計算 ===
    reflectivity = _compute_surface_reflectivity(n_ref[0], k_ext[0])
    
    # === 6. FCA 吸収係数を計算 ===
    alpha_fca = _compute_alpha_fca(k_ext, config)
    
    # === 7. 表面レーザー強度を計算 ===
    I_surface = _compute_surface_intensity(reflectivity, t, config)
    
    # === 8. z方向強度伝播 ===
    intensity = _propagate_laser_intensity(I_surface, alpha_spa, alpha_fca, config)
    
    # === 9. 熱源項を計算 ===
    source_term = _compute_heat_source(intensity, alpha_spa, alpha_fca, config.beta_cgs)
    
    return OpticsResult(
        intensity=intensity,
        source_term=source_term,
        reflectivity=reflectivity,
        alpha_fca=alpha_fca,
    )


def _compute_drude_epsilon(
    ne: NDArray[np.float64],
    tau_e: NDArray[np.float64],
    config: OpticsConfig,
) -> NDArray[np.complex128]:
    """Drude 複素誘電率を計算する。
    
    ε = 1 + (εr - 1)(1 - ne/n0) - (ne e²)/(ε0 me ω²(1 + iν/ω))
    
    Args:
        ne: キャリア密度 [cm⁻³], shape: (n_z,)
        tau_e: 衝突時間 [s], shape: (n_z,)
        config: OpticsConfig
    
    Returns:
        ε: 複素誘電率, shape: (n_z,)
    """
    n_z = len(ne)
    epsilon = np.zeros(n_z, dtype=np.complex128)
    
    for i in range(n_z):
        epsilon[i] = _compute_epsilon_at_point(
            ne[i], tau_e[i], config
        )
    
    return epsilon


def _compute_epsilon_at_point(
    ne: float,
    tau_e: float,
    config: OpticsConfig,
) -> complex:
    """1点における Drude 複素誘電率を計算する。
    
    Args:
        ne: キャリア密度 [cm⁻³]
        tau_e: 衝突時間 [s]
        config: OpticsConfig
    
    Returns:
        ε: 複素誘電率
    """
    # 単位変換: cm⁻³ → m⁻³
    ne_m3 = convert_cm3_to_m3(ne)

    # 第1項: 背景誘電率（Si 光学誘電率 εr = 12.709 @ 1030nm、ne 非依存）
    epsilon_s_rel = config.epsilon_r.real

    # 第2項: 自由キャリアの Drude 寄与（光学有効質量 m* を使用）
    nu = 1.0 / tau_e
    omega = config.omega
    denominator = 1.0 + 1j * nu / omega
    drude_term = (ne_m3 * config.e_charge**2) / (config.epsilon_0 * config.m_eff_drude * omega**2 * denominator)

    return epsilon_s_rel - drude_term


def _compute_refractive_index(epsilon: NDArray[np.complex128]) -> NDArray[np.float64]:
    """複素誘電率から屈折率を計算する。
    
    n = sqrt[(Re(ε) + sqrt(Re(ε)² + Im(ε)²))/2]
    
    Args:
        epsilon: 複素誘電率, shape: (n_z,)
    
    Returns:
        n_ref: 屈折率（無次元）, shape: (n_z,)
    """
    re_eps = epsilon.real
    im_eps = epsilon.imag
    return np.sqrt((re_eps + np.sqrt(re_eps**2 + im_eps**2)) / 2.0)


def _compute_extinction_coefficient(epsilon: NDArray[np.complex128]) -> NDArray[np.float64]:
    """複素誘電率から消衰係数を計算する。
    
    k = sqrt[(-Re(ε) + sqrt(Re(ε)² + Im(ε)²))/2]
    
    Args:
        epsilon: 複素誘電率, shape: (n_z,)
    
    Returns:
        k_ext: 消衰係数（無次元）, shape: (n_z,)
    """
    re_eps = epsilon.real
    im_eps = epsilon.imag
    return np.sqrt((-re_eps + np.sqrt(re_eps**2 + im_eps**2)) / 2.0)


def _compute_surface_reflectivity(n0: float, k0: float) -> float:
    """表面（z=0）の反射率を計算する。
    
    R = [(n-1)² + k²] / [(n+1)² + k²]
    
    Args:
        n0: 表面の屈折率
        k0: 表面の消衰係数
    
    Returns:
        R: 表面反射率（無次元）
    """
    numerator = (n0 - 1.0) ** 2 + k0**2
    denominator = (n0 + 1.0) ** 2 + k0**2
    return numerator / denominator


def _compute_alpha_fca(
    k_ext: NDArray[np.float64],
    config: OpticsConfig,
) -> NDArray[np.float64]:
    """自由キャリア吸収係数を計算する。
    
    α_FCA = 2ω k / c = 4π k / λ
    
    Args:
        k_ext: 消衰係数, shape: (n_z,)
        config: OpticsConfig
    
    Returns:
        α_FCA [cm⁻¹], shape: (n_z,)
    """
    wavelength_cm = convert_nm_to_cm(config.wavelength_nm)
    alpha_fca = (4.0 * np.pi * k_ext) / wavelength_cm
    return alpha_fca


def _compute_surface_intensity(
    reflectivity: float,
    t: float,
    config: OpticsConfig,
) -> float:
    """表面レーザー強度を計算する（ガウシアンパルス）。
    
    I(0,t) = 0.94 × [(1-R)F/tp] × exp(-2.77(t/tp)²)
    
    Args:
        reflectivity: 表面反射率
        t: 現在時刻 [s]
        config: OpticsConfig
    
    Returns:
        I_surface: 表面強度 [W/cm²]
    """
    # 論文 Eq.(9): I(0,t) = 0.94 × (2/tp) × (1-R) × F × exp(-2.77(t/tp)²)
    factor = 0.94
    peak_intensity = (1.0 - reflectivity) * config.fluence / config.pulse_duration
    time_profile = np.exp(-2.77 * (t / config.pulse_duration) ** 2)
    return factor * peak_intensity * time_profile


def _propagate_laser_intensity(
    I_surface: float,
    alpha_spa: NDArray[np.float64],
    alpha_fca: NDArray[np.float64],
    config: OpticsConfig,
) -> NDArray[np.float64]:
    """z方向に強度ODEを前進差分で解く。
    
    dI/dz = -(α_SPA + α_FCA)I - β I²
    境界条件: I[0] = I_surface
    
    Args:
        I_surface: 表面強度 [W/cm²]
        alpha_spa: α_SPA(z) [cm⁻¹], shape (n_z,)
        alpha_fca: α_FCA(z) [cm⁻¹], shape (n_z,)
        config: OpticsConfig
    
    Returns:
        I(z): 強度分布 [W/cm²], shape (n_z,)
    """
    n_z = len(alpha_spa)
    intensity = np.zeros(n_z, dtype=np.float64)
    intensity[0] = I_surface
    
    for i in range(n_z - 1):
        I_current = intensity[i]
        alpha_total = alpha_spa[i] + alpha_fca[i]
        dI_dz = -alpha_total * I_current - config.beta_cgs * I_current**2
        intensity[i + 1] = I_current + config.dz * dI_dz
        intensity[i + 1] = max(intensity[i + 1], 0.0)
    
    return intensity


def _compute_heat_source(
    intensity: NDArray[np.float64],
    alpha_spa: NDArray[np.float64],
    alpha_fca: NDArray[np.float64],
    beta_cgs: float,
) -> NDArray[np.float64]:
    """熱源項を計算する。

    S(z) = (α_SPA + α_FCA + β×I) × I = (α_SPA + α_FCA)×I + β×I²

    β×I² は TPA の電子系への熱源項。TPA で生成されたキャリア1個あたり 2hν の
    エネルギーが系に入り、そのうち Eg は電子系を経由してキャリア生成エネルギーとして
    消費され（TTM キャリア項 -(Eg+3kB Te)×dne/dt で扱われる）、残りが電子加熱に寄与する。
    β×I² を S に含め、キャリア項でのエネルギー保存を成立させる。

    Args:
        intensity: レーザー強度分布 [W/cm²], shape: (n_z,)
        alpha_spa: α_SPA(z) [cm⁻¹], shape: (n_z,)
        alpha_fca: α_FCA(z) [cm⁻¹], shape: (n_z,) — Drude 由来（背景含む）
        beta_cgs: TPA 係数 [cm/W]

    Returns:
        S(z): 熱源項 [W/cm³], shape: (n_z,)
    """
    source_term = (alpha_spa + alpha_fca) * intensity + beta_cgs * intensity**2
    return source_term
