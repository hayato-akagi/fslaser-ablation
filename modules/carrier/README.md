# modules/carrier — キャリア密度モデル

> **前提**: 本ドキュメントは `modules/CONVENTIONS.md` の全規約に準拠する。  
> 実装前に必ず CONVENTIONS.md を参照すること。

---

## 1. 責務

フェムト秒レーザー照射時のシリコン伝導帯における**自由キャリア密度 $n_e(z,t)$** の時間発展を計算する。
1タイムステップ分の $n_e$ 更新と、TTMが必要とする $\partial n_e / \partial t$ の提供が本モジュールの責務。

---

## 2. 支配方程式

$$
\frac{\partial n_e}{\partial t} = \underbrace{\frac{\alpha_{SPA} I}{h\omega}}_{\text{SPA}} + \underbrace{\frac{\beta I^2}{2h\omega}}_{\text{TPA}} - \underbrace{\gamma n_e^3}_{\text{Auger}} + \underbrace{\theta n_e}_{\text{Impact}} - \underbrace{\nabla(D_0 \nabla n_e)}_{\text{Diffusion}}
$$

### 2.1 各項の定義

| # | 項名 | 式 | 物理的意味 |
|---|---|---|---|
| 1 | 単一光子吸収 (SPA) | $\frac{\alpha_{SPA}(T_l) \cdot I(z)}{h\omega}$ | 線形吸収によるキャリア生成 |
| 2 | 二光子吸収 (TPA) | $\frac{\beta \cdot I(z)^2}{2 h\omega}$ | 非線形吸収によるキャリア生成 |
| 3 | オージェ再結合 | $-\gamma \cdot n_e(z)^3$ | 3体衝突によるキャリア消滅 |
| 4 | 衝突電離 | $+\theta(T_e, E_g) \cdot n_e(z)$ | キャリア増殖 |
| 5 | 両極性拡散 | $-\nabla(D_0(T_l) \cdot \nabla n_e)$ | z方向のキャリア拡散 |

### 2.2 温度依存パラメータの計算式

**衝突電離係数** $\theta$:
$$
\theta = 3.6 \times 10^{20} \cdot \exp\left(\frac{-1.5 \, E_g[\text{eV}]}{k_B[\text{J/K}] \cdot T_e[\text{K}]}\right) \quad [\text{s}^{-1}]
$$
注意: $k_B = 1.381 \times 10^{-23}$ J/K、$E_g$ は eV 単位のまま渡す（混合単位系）。指数は `np.clip(exponent, -100, 100)` でクリップする。

**両極性拡散係数** $D_0$:
$$
D_0 = 18 \cdot \frac{T_{rm}}{T_l} \quad [\text{cm}^2/\text{s}]
$$
ここで $T_{rm} = 300$ K（室温）。

**バンドギャップエネルギー** $E_g$（固相のみ。液相では $E_g = 0$）:
$$
E_g = 1.16 - 7.02 \times 10^{-4} \frac{T_l^2}{T_l + 1108} - 1.5 \times 10^{-8} \cdot n_e^{1/3} \quad [\text{eV}]
$$

**SPA吸収係数** $\alpha_{SPA}$:
$$
\alpha_{SPA}(T_l) = -58.95 + 0.6226 T_l - 2.3 \times 10^{-3} T_l^2 + 3.186 \times 10^{-6} T_l^3 + 9.967 \times 10^{-10} T_l^4 - 1.409 \times 10^{-13} T_l^5 \quad [\text{cm}^{-1}]
$$
注意: `carrier` は `optics` の出力 `I(z)` を受け取るが、SPA項のキャリア生成計算には `material_properties.compute_alpha_spa(Tl)` を独立に呼び出して $\alpha_{SPA}$ を計算する。`optics` での光伝播計算と同一の物性値を使用するが、計算は各モジュールが独立して行う。

### 2.3 拡散項のFDM離散化（中心差分）

$$
\nabla(D_0 \nabla n_e)\bigg|_i = \frac{1}{\Delta z^2}\left[D_{0,i+1/2}(n_{e,i+1} - n_{e,i}) - D_{0,i-1/2}(n_{e,i} - n_{e,i-1})\right]
$$

半整数点の拡散係数:
$$
D_{0,i+1/2} = \frac{D_{0,i} + D_{0,i+1}}{2}
$$

境界条件（断熱）: ゴースト点 $n_{e,-1} = n_{e,0}$, $n_{e,N} = n_{e,N-1}$

### 2.4 時間積分（前進オイラー法）

$$
n_e^{n+1}(z_i) = n_e^n(z_i) + \Delta t \cdot \text{RHS}_i
$$

ここで $\text{RHS}_i$ は上記5項の合計。

`dne_dt` の出力:
$$
\text{dne\_dt}(z_i) = \text{RHS}_i = \frac{n_e^{n+1}(z_i) - n_e^n(z_i)}{\Delta t}
$$

---

## 3. public API シグネチャ

```python
def advance_carrier_density(
    ne: NDArray[np.float64],        # (n_z,) 現在のキャリア密度 [cm⁻³]
    intensity: NDArray[np.float64], # (n_z,) レーザー強度 I(z) [W/cm²]
    Te: NDArray[np.float64],        # (n_z,) 電子温度 [K]
    Tl: NDArray[np.float64],        # (n_z,) 格子温度 [K]
    phase_state: NDArray[np.int32], # (n_z,) 相状態 (PhaseState)
    dt: float,                      # 時間刻み [s]
    config: CarrierConfig,
) -> CarrierResult:
    """1タイムステップ分のキャリア密度更新。

    呼び出しタイミング: euler_fdm の Step 2（optics の後、ttm の前）
    """
```

### 3.1 CarrierResult の定義

```python
class CarrierResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    ne: NDArray[np.float64]      # (n_z,) 更新後キャリア密度 [cm⁻³]
    dne_dt: NDArray[np.float64]  # (n_z,) 時間変化率 [cm⁻³/s]
```

- `ne`: 更新後の値（$n_e^{n+1}$）
- `dne_dt`: RHS値。TTMの電子温度方程式（Eq.2）の第4-5項で使用される

---

## 4. phase_state の利用方法

`phase_state` は以下の目的で使用する:
- **液相** (`PhaseState.LIQUID`, `VAPORIZING`, `VAPOR`) のグリッドでは $E_g = 0$ とする
- **固相** (`PhaseState.SOLID`, `MELTING`) のグリッドでは $E_g$ を温度依存式で計算する
- $\theta$ と $D_0$ の計算に $E_g$ が関与するため、相状態が間接的に影響する

---

## 5. config.py のパラメータ一覧

```python
class CarrierConfig(BaseModel):
    # 定数（material_properties.constants から取得）
    gamma: float = SILICON.gamma_auger   # Auger再結合係数 [cm⁶/s] = 3.8e-31
    beta_tpa: float = SILICON.beta_tpa   # TPA係数 [cm/GW] = 9.0
    photon_energy_eV: float = LASER_1030NM.photon_energy_eV  # hω [eV]（波長から計算）
    T_room: float = SILICON.T_room       # 室温 [K] = 300.0
    k_B: float = PHYSICAL.k_B           # ボルツマン定数 [J/K] = 1.381e-23
    omega: float = ...                   # レーザー角振動数 [rad/s]（波長・光速から計算）
    
    # 拡散項に必要
    dz: float                            # グリッド間隔 [cm] ← euler_fdm から注入
    
    @property
    def beta_cgs(self) -> float:
        """TPA係数を内部単位 [cm/W] に変換。9.0 cm/GW → 9.0e-9 cm/W"""
    
    @property
    def photon_energy_J(self) -> float:
        """光子エネルギー [J]（I[W/cm²] と整合する単位系）"""
```

### 5.1 `dz` の注入

`dz` は `euler_fdm` の `GridConfig.dz` から取得し、`CarrierConfig` 生成時に渡す。

### 5.2 単位系

本モジュールの単位系は W-J（SI ベース）を採用する。
- 強度 I: `[W/cm²]`
- 光子エネルギー hω: `[J]`（`photon_energy_J` プロパティ使用）
- β: `[cm/W]`（`beta_cgs` プロパティ = `beta_tpa * 1e-9`）

---

## 6. ファイル構成と責務

| ファイル | 責務 | 主要関数/クラス |
|---|---|---|
| `public.py` | 外部API + 型定義 | `CarrierResult`, `advance_carrier_density()` |
| `config.py` | パラメータ定義 | `CarrierConfig` |
| `solver.py` | 更新シーケンス（層統合） | `advance_carrier_density_impl()`, 物性取得 → RHS(5項) → オイラー更新 |

---

## 7. テスト方針

```bash
docker compose run --rm sim pytest tests/test_carrier/ -v
```

### 7.1 単体テストケース

| テスト名 | 内容 | 検証方法 |
|---|---|---|
| `test_spa_term` | SPA項のみ（他項=0）でキャリア増加 | $\Delta n_e = \alpha_{SPA} I \Delta t / h\omega$ と一致 |
| `test_tpa_term` | TPA項のみでキャリア増加 | $\Delta n_e = \beta I^2 \Delta t / (2 h\omega)$ と一致 |
| `test_auger_term` | Auger項のみでキャリア減少 | $\Delta n_e = -\gamma n_e^3 \Delta t$ と一致 |
| `test_impact_term` | Impact項のみでキャリア増加 | $\Delta n_e = \theta n_e \Delta t$ と一致 |
| `test_diffusion_conservation` | 拡散項のみ → 総キャリア数保存 | $\sum n_e^{n+1} \approx \sum n_e^n$ |
| `test_diffusion_gaussian` | 初期ガウス分布の拡散 → 広がり検証 | 分散の増加が $2 D_0 \Delta t$ に一致 |
| `test_eg_phase_switch` | 固相→液相で $E_g = 0$ に切り替わる | $\theta$, 吸収項が適切に変化 |
| `test_dne_dt_consistency` | `dne_dt` が `(ne_new - ne_old) / dt` と一致 | 数値比較 |
