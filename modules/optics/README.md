# modules/optics — 動的光学モデル（Drude + レーザー伝播）

> **前提**: 本ドキュメントは `modules/CONVENTIONS.md` の全規約に準拠する。  
> 実装前に必ず CONVENTIONS.md を参照すること。

---

## 1. 責務

キャリア密度 $n_e(z)$ と格子温度 $T_l(z)$ に基づき、**Drudeモデル**で動的光学特性を計算し、
**レーザー強度 $I(z,t)$ のz軸方向分布**を求める。TTM向けの熱源項 $S(z,t)$ を算出して返す。

本モジュールは**タイムステップ内で最初に呼ばれる**（euler_fdm の Step 1）。

---

## 2. 支配方程式

### 2.1 Drudeモデル（動的複素誘電率）

$$
\varepsilon = 1 + (\varepsilon_r - 1)\left(1 - \frac{n_e}{n_0}\right) - \frac{n_e e^2}{\varepsilon_0 m_e \omega^2 (1 + i\nu/\omega)}
$$

ここで:
- $\varepsilon_r = 12.709 + 0.0017149i$（1030 nm, 300 K での誘電率）
- $n_0 = 5.0 \times 10^{22}$ cm⁻³（価電子帯の電子密度）
- $e = 1.602 \times 10^{-19}$ C（電子電荷）
- $\varepsilon_0 = 8.854 \times 10^{-12}$ F/m（真空の誘電率）
- $m_e = 9.11 \times 10^{-31}$ kg（有効電子質量）
- $\omega = 2\pi c / \lambda$（$\lambda = 1030$ nm でのレーザー角振動数）
- $\nu = 1/\tau_e$（電子-フォノン衝突振動数。$\tau_e$ は相依存）

**$\tau_e$（電子衝突時間）の相依存**:
| 相 | $\tau_e$ |
|---|---|
| 固相 | $240 \times (1 + n_e / 6.0 \times 10^{20})$ fs |
| 液相 | $10^{-12}$ s（=1 ps 固定） |

### 2.2 屈折率・消衰係数

$\varepsilon$ は一般に複素数。実部 $\text{Re}(\varepsilon)$, 虚部 $\text{Im}(\varepsilon)$ から:

$$
n_{ref} = \sqrt{\frac{\text{Re}(\varepsilon) + \sqrt{\text{Re}(\varepsilon)^2 + \text{Im}(\varepsilon)^2}}{2}}
$$

$$
k_{ext} = \sqrt{\frac{-\text{Re}(\varepsilon) + \sqrt{\text{Re}(\varepsilon)^2 + \text{Im}(\varepsilon)^2}}{2}}
$$

注意: 本書では光学的屈折率を $n_{ref}$、消衰係数を $k_{ext}$ と表記する（キャリア密度 $n_e$ との混同を避けるため）。

### 2.3 表面動的反射率

$$
R(0, t) = \frac{(n_{ref,0} - 1)^2 + k_{ext,0}^2}{(n_{ref,0} + 1)^2 + k_{ext,0}^2}
$$

ここで $n_{ref,0}$, $k_{ext,0}$ は表面（$z = 0$）での値。

### 2.4 自由キャリア吸収（FCA）係数

$$
\alpha_{FCA}(z) = \frac{2 \omega \, k_{ext}(z)}{c}
$$

$c = 3.0 \times 10^{10}$ cm/s（光速、CGS）。

### 2.5 SPA吸収係数（温度依存）

$$
\alpha_{SPA}(T_l) = -58.95 + 0.6226 T_l - 2.3 \times 10^{-3} T_l^2 + 3.186 \times 10^{-6} T_l^3 + 9.967 \times 10^{-10} T_l^4 - 1.409 \times 10^{-13} T_l^5 \quad [\text{cm}^{-1}]
$$

### 2.6 表面レーザー強度（ガウシアンパルスの時間プロファイル）

$$
I(0, t) = 0.94 \times \frac{[1 - R(0,t)] \times F}{t_p} \cdot \exp\left(-2.77 \left(\frac{t}{t_p}\right)^2\right)
$$

- $F$: レーザーフルエンス [J/cm²]
- $t_p$: パルス幅 (FWHM) [s]。デフォルト: $421 \times 10^{-15}$ s
- $t$: 時刻 [s]。**$t = 0$ がパルス中心**

### 2.7 z軸方向のレーザー強度減衰（ODE）

$$
\frac{\partial I}{\partial z} = -(\alpha_{SPA} + \alpha_{FCA}) \, I - \beta \, I^2
$$

- $\beta = 9.0$ cm/GW。**CGS内部単位への変換が必要**：$\beta_{\text{CGS}} = 9.0 \times 10^{-9}$ cm·s/erg
- 境界条件: $I(z=0) = I(0, t)$（上記 2.6 で計算した表面強度）
- z方向に**前進差分**で数値積分する（表面 $z[0]$ → 底面 $z[N-1]$）

離散化:
$$
I_{i+1} = I_i + \Delta z \cdot \left[-(\alpha_{SPA,i} + \alpha_{FCA,i}) I_i - \beta I_i^2\right]
$$

### 2.8 熱源項

$$
S(z, t) = (\alpha_{SPA}(z) + \alpha_{FCA}(z)) \times I(z, t) \quad [\text{W/cm}^3]
$$

---

## 3. 計算フロー（solver.py）

```
1. Drude誘電率計算:     ε(z) ← ne(z), phase_state(z)
2. 屈折率・消衰係数:    n_ref(z), k_ext(z) ← ε(z)
3. 表面反射率:          R ← n_ref[0], k_ext[0]
4. FCA吸収係数:         α_FCA(z) ← k_ext(z)
5. SPA吸収係数:         α_SPA(z) ← Tl(z)
6. 表面レーザー強度:    I[0] ← R, F, t, t_p
7. z方向強度伝播:       I[1..N-1] ← ODE前進差分
8. 熱源項:              S(z) ← (α_SPA + α_FCA) × I(z)
9. OpticsResult構築
```

---

## 4. public API シグネチャ

```python
def compute_laser_field(
    ne: NDArray[np.float64],        # (n_z,) キャリア密度 [cm⁻³]
    Tl: NDArray[np.float64],        # (n_z,) 格子温度 [K]
    phase_state: NDArray[np.int32], # (n_z,) 相状態 (PhaseState)
    t: float,                       # 現在時刻 [s]（t=0 がパルス中心）
    config: OpticsConfig,
) -> OpticsResult:
    """レーザー場の空間分布を計算。

    呼び出しタイミング: euler_fdm の Step 1（各タイムステップの最初）
    """
```

### 4.1 OpticsResult の定義

```python
class OpticsResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    intensity: NDArray[np.float64]    # I(z): shape (n_z,), W/cm²
    source_term: NDArray[np.float64]  # S(z): shape (n_z,), W/cm³
    reflectivity: float               # R(0,t): 表面反射率, 無次元
    alpha_fca: NDArray[np.float64]    # α_FCA(z): shape (n_z,), cm⁻¹
```

---

## 5. config.py のパラメータ一覧

```python
class OpticsConfig(BaseModel):
    # 誘電率
    epsilon_r: complex = 12.709 + 0.0017149j   # 1030nm, 300K
    n0_valence: float = 5.0e22                  # 価電子帯密度 [cm⁻³]
    
    # 電子
    m_e: float = 9.11e-31                       # 有効電子質量 [kg]
    e_charge: float = 1.602e-19                 # 電子電荷 [C]
    epsilon_0: float = 8.854e-12                # 真空誘電率 [F/m]
    
    # レーザー
    wavelength_nm: float = 1030.0               # 波長 [nm] (constants.py から)
    c_light: float = 3.0e10                     # 光速 [cm/s]
    fluence: float                              # F [J/cm²] ← euler_fdm から注入
    pulse_duration: float = 421e-15             # t_p (FWHM) [s]
    
    # 吸収
    beta_tpa: float = 9.0                       # TPA係数 [cm/GW]
    
    # 衝突時間（固相用パラメータ）
    tau_e_base: float = 240e-15                 # 基底衝突時間 [s]
    tau_e_ne_ref: float = 6.0e20                # 衝突時間の ne 参照密度 [cm⁻³]
    tau_e_liquid: float = 1e-12                 # 液相での衝突時間 [s]
    
    # グリッド
    dz: float                                   # グリッド間隔 [cm] ← euler_fdm から注入
    
    @property
    def omega(self) -> float:
        """レーザー角振動数 [rad/s]"""
        return 2.0 * np.pi * self.c_light / (self.wavelength_nm * 1e-7)
    
    @property
    def beta_cgs(self) -> float:
        """TPA係数を内部単位 [cm/W] に変換。9.0 cm/GW → 9.0e-9 cm/W"""
```

---

## 6. ファイル構成と責務

| ファイル | 責務 | 主要関数/クラス |
|---|---|---|
| `public.py` | 外部API + 型定義 | `OpticsResult`, `compute_laser_field()` |
| `config.py` | パラメータ定義 | `OpticsConfig` |
| `solver.py` | 計算シーケンス（層統合） | `compute_laser_field_sequence()`, Drude→(n,k)→(R,αFCA)→I(z)→S(z) |

---

## 7. テスト方針

```bash
docker compose run --rm sim pytest tests/test_optics/ -v
```

### 7.1 単体テストケース

| テスト名 | 内容 | 検証方法 |
|---|---|---|
| `test_drude_low_ne` | $n_e \approx 0$ → 静的誘電率に一致 | $\varepsilon \approx \varepsilon_r$ |
| `test_reflectivity_300K` | 300K, $n_e = 10^{12}$ → 静的反射率 | $R \approx 0.30$（Si の既知値） |
| `test_reflectivity_high_ne` | $n_e = 10^{22}$ → 金属的反射率上昇 | $R > 0.5$ |
| `test_beer_lambert` | $\beta = 0$ → $I(z) = I_0 e^{-\alpha z}$ | 指数減衰 |
| `test_pulse_energy` | ガウスパルスの時間積分 ≈ F | $\int I(0,t) dt \approx (1-R) F$ |
| `test_source_term` | $S = (\alpha_{SPA} + \alpha_{FCA}) \times I$ | 数値一致 |
| `test_tau_e_phase_switch` | 固相→液相 で τe が切り替わる | $\nu$ の値変化を確認 |
| `test_alpha_spa_polynomial` | 既知温度での αSPA 値 | 300K: $\alpha_{SPA} \approx 29$ cm⁻¹ |
