# modules/ttm — 二温度モデル (Two-Temperature Model)

> **前提**: 本ドキュメントは `modules/CONVENTIONS.md` の全規約に準拠する。  
> 実装前に必ず CONVENTIONS.md を参照すること。

---

## 1. 責務

電子温度 $T_e(z,t)$ と格子温度 $T_l(z,t)$ の時間発展を計算する。
相転移（固→液→気）における**潜熱管理**と**熱物性パラメータの相切替**も本モジュールの責務。

本モジュールは euler_fdm の **Step 3**（carrier の後、ablation の前）で呼ばれる。

---

## 2. 支配方程式

### 2.1 電子温度方程式

$$
C_e \frac{\partial T_e}{\partial t} = \nabla(K_e \nabla T_e) - G(T_e - T_l) + S(z,t) - \left(E_g + 3k_B T_e - n_e \frac{\partial E_g}{\partial n_e}\right)\frac{\partial n_e}{\partial t} - n_e \frac{\partial E_g}{\partial T_l}\frac{\partial T_l}{\partial t}
$$

右辺の各項:
| # | 項 | 物理的意味 |
|---|---|---|
| 1 | $\nabla(K_e \nabla T_e)$ | 電子熱拡散 |
| 2 | $-G(T_e - T_l)$ | 電子-格子結合（エネルギー移動） |
| 3 | $S(z,t)$ | レーザー熱源 |
| 4 | $-(E_g + 3k_B T_e - n_e \frac{\partial E_g}{\partial n_e})\frac{\partial n_e}{\partial t}$ | キャリア生成に伴うエネルギー消費 |
| 5 | $-n_e \frac{\partial E_g}{\partial T_l}\frac{\partial T_l}{\partial t}$ | バンドギャップの温度変化によるエネルギー変化 |

**第5項の取り扱い**: $\partial T_l / \partial t$ は前ステップの値（陽的評価）を使用する。初回ステップでは 0 とする。

### 2.2 格子温度方程式

$$
C_l \frac{\partial T_l}{\partial t} = \nabla(K_l \nabla T_l) + G(T_e - T_l)
$$

### 2.3 FDM離散化（中心差分 + 前進オイラー）

熱拡散項の離散化（$T_e$ の例。$T_l$ も同様）:
$$
\nabla(K_e \nabla T_e)\bigg|_i = \frac{1}{\Delta z^2}\left[K_{e,i+1/2}(T_{e,i+1} - T_{e,i}) - K_{e,i-1/2}(T_{e,i} - T_{e,i-1})\right]
$$

半整数点: $K_{e,i+1/2} = (K_{e,i} + K_{e,i+1}) / 2$

境界条件（断熱 Neumann）:
- 表面: ゴースト点 $T_{e,-1} = T_{e,0}$（$i=0$ での拡散項フラックス = 0）
- 底面: ゴースト点 $T_{e,N} = T_{e,N-1}$

---

## 3. 熱物性パラメータ（相依存）

### 3.1 パラメータ一覧

| パラメータ | 固相 (SOLID / MELTING) | 液相 (LIQUID / VAPORIZING / VAPOR) | 単位 |
|---|---|---|---|
| $C_e$ | $3 n_e k_B$ | $10^{-4} \times T_e$ | J/(cm³·K) |
| $C_l$ | $1.978 + 3.54 \times 10^{-4} T_l - 3.68 T_l^{-2}$ | $1.06 \times \rho = 2.69$ J/(cm³·K)（$\rho = 2.54$ g/cm³） | J/(cm³·K) |
| $K_e$ | $1.6 \times 10^{11} \times (-3.47 \times 10^8 + 4.45 \times 10^6 T_e)$ | $67 \times 10^{-2}$ | W/(cm·K) |
| $K_l$ | $1585 \, T_l^{-1.23}$ | $0.5 + 2.9 \times 10^{-4} (T_l - T_m)$ | W/(cm·K) |
| $G$ | — | — | W/(cm³·K) |

### 3.2 結合因子 $G$ の計算

$$
G = \frac{C_e}{\tau_e}
$$

$\tau_e$ は相依存（optics と同一の定義）:
| 相 | $\tau_e$ |
|---|---|
| 固相 | $240 \times (1 + n_e / 6.0 \times 10^{20})$ fs |
| 液相 | $10^{-12}$ s |

### 3.3 $k_B$ の値

- 熱容量計算での $k_B$: $1.381 \times 10^{-23}$ J/K = $1.381 \times 10^{-16}$ erg/K
- CGS → J/cm³·K への変換: $k_B = 1.381 \times 10^{-23}$ J/K（$n_e$ が cm⁻³ なので $3 n_e k_B$ は J/(cm³·K)）

### 3.4 バンドギャップ関連の補助量

$E_g$ のキャリア密度偏微分:
$$
\frac{\partial E_g}{\partial n_e} = -\frac{1.5 \times 10^{-8}}{3} n_e^{-2/3} \quad [\text{eV·cm}]
$$

$E_g$ の格子温度偏微分:
$$
\frac{\partial E_g}{\partial T_l} = -7.02 \times 10^{-4} \frac{T_l(T_l + 2 \times 1108)}{(T_l + 1108)^2} \quad [\text{eV/K}]
$$

注意: 液相（$E_g = 0$）では両偏微分とも 0 とする。

---

## 4. 相転移ロジック

### 4.1 状態遷移図

```
SOLID ──[Tl >= Tm]──► MELTING ──[Lm 消化完了]──► LIQUID ──[Tl >= Tb]──► VAPORIZING ──[Lv 消化完了]──► VAPOR
```

### 4.2 潜熱処理アルゴリズム（各グリッド点 i について）

```python
# 格子温度方程式の RHS を計算
rhs_l = diffusion_term + G * (Te[i] - Tl[i])

if phase_state[i] == SOLID and Tl[i] + dt * rhs_l / Cl[i] >= Tm:
    # 融点到達 → MELTING 状態へ
    phase_state[i] = MELTING
    Tl[i] = Tm  # 温度を融点に固定
    # 余剰エネルギーを潜熱カウンタに蓄積
    excess_energy = Cl[i] * (Tl[i] + dt * rhs_l / Cl[i] - Tm)
    latent_heat_accumulated[i] += excess_energy

elif phase_state[i] == MELTING:
    # Tl を Tm に固定し、エネルギーを潜熱カウンタに蓄積
    Tl[i] = Tm
    latent_heat_accumulated[i] += dt * rhs_l  # rhs_l > 0 ならエネルギー蓄積
    if latent_heat_accumulated[i] >= Lm:
        # 融解完了 → LIQUID
        phase_state[i] = LIQUID
        # 残余エネルギーで温度上昇
        remaining = latent_heat_accumulated[i] - Lm
        Tl[i] = Tm + remaining / Cl_liquid[i]
        latent_heat_accumulated[i] = 0.0

elif phase_state[i] == LIQUID and Tl[i] + dt * rhs_l / Cl[i] >= Tb:
    # 沸点到達 → VAPORIZING
    phase_state[i] = VAPORIZING
    Tl[i] = Tb
    excess_energy = Cl[i] * (Tl[i] + dt * rhs_l / Cl[i] - Tb)
    latent_heat_accumulated[i] += excess_energy

elif phase_state[i] == VAPORIZING:
    Tl[i] = Tb
    latent_heat_accumulated[i] += dt * rhs_l
    if latent_heat_accumulated[i] >= Lv:
        phase_state[i] = VAPOR
        remaining = latent_heat_accumulated[i] - Lv
        Tl[i] = Tb + remaining / Cl[i]
        latent_heat_accumulated[i] = 0.0

else:
    # SOLID (Tl < Tm), LIQUID (Tl < Tb), VAPOR → 通常のオイラー更新
    Tl[i] = Tl[i] + dt * rhs_l / Cl[i]
```

### 4.3 相転移時の熱物性切替タイミング

- 相状態が変化したグリッド点は、**次のタイムステップから**新しい相の熱物性を使用する
- 同一ステップ内で相転移した場合、そのステップの $T_e$ 計算にはまだ旧相のパラメータを使用

---

## 5. public API シグネチャ

```python
def advance_temperatures(
    Te: NDArray[np.float64],                      # (n_z,) 電子温度 [K]
    Tl: NDArray[np.float64],                      # (n_z,) 格子温度 [K]
    ne: NDArray[np.float64],                      # (n_z,) キャリア密度 [cm⁻³]
    dne_dt: NDArray[np.float64],                  # (n_z,) キャリア時間変化率 [cm⁻³/s]
    source_term: NDArray[np.float64],             # (n_z,) 熱源項 S(z) [W/cm³]
    phase_state: NDArray[np.int32],               # (n_z,) 現在の相状態 (入力)
    latent_heat_accumulated: NDArray[np.float64], # (n_z,) 現在の潜熱蓄積量 [J/cm³] (入力)
    dt: float,                                    # 時間刻み [s]
    config: TTMConfig,
) -> TTMResult:
    """1タイムステップ分の温度更新。

    呼び出しタイミング: euler_fdm の Step 3（carrier の後、ablation の前）
    
    注意: phase_state, latent_heat_accumulated は入力を直接変更せず、
    新しい配列として TTMResult に格納して返す。
    """
```

### 5.1 TTMResult の定義

```python
class TTMResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    Te: NDArray[np.float64]                       # (n_z,) 更新後電子温度 [K]
    Tl: NDArray[np.float64]                       # (n_z,) 更新後格子温度 [K]
    phase_state: NDArray[np.int32]                # (n_z,) 更新後相状態
    latent_heat_accumulated: NDArray[np.float64]  # (n_z,) 更新後潜熱蓄積量 [J/cm³]
```

### 5.2 入力配列の不変性（重要）

- 入力として受け取る全ての `NDArray` は**読み取り専用として扱い、in-place 変更しない**
- 更新結果は新しい配列を生成して `TTMResult` に格納する
- これにより、euler_fdm が状態管理を一元的に制御できる

---

## 6. config.py のパラメータ一覧

```python
class TTMConfig(BaseModel):
    # 相転移温度
    T_m: float = 1687.0              # 融点 [K]
    T_b: float = 3583.0              # 沸点 [K]
    T_room: float = 300.0            # 室温 [K]
    
    # 潜熱
    L_m: float = 4206.0              # 融解潜熱 [J/cm³]
    L_v: float = 32020.0             # 気化潜熱 [J/cm³]
    
    # 密度
    rho: float = 2.54                # 密度 [g/cm³]
    
    # ボルツマン定数
    k_B: float = 1.381e-23           # [J/K]
    k_B_eV: float = 8.617333e-5     # [eV/K]
    
    # 衝突時間（G計算用。optics と同一値）
    tau_e_base: float = 240e-15      # [s]
    tau_e_ne_ref: float = 6.0e20     # [cm⁻³]
    tau_e_liquid: float = 1e-12      # [s]
    
    # グリッド
    dz: float                        # グリッド間隔 [cm] ← euler_fdm から注入
```

---

## 7. `dTl_dt_prev` の管理

電子温度方程式の第5項に $\partial T_l / \partial t$ が含まれる。これは **前ステップの格子温度変化率**を使用する。

- **euler_fdm** が前ステップの $T_l$ と現ステップの $T_l$ から $\partial T_l / \partial t$ を計算し、次ステップの `advance_temperatures` に渡す
- 初回ステップでは $\partial T_l / \partial t = 0$（全グリッド）

**実装方針**: 第5項の寄与は他項に比べて小さいため、初期実装では**第5項を省略し、第4項のみ実装**しても物理的に妥当な結果が得られる。完全実装は最適化フェーズで追加する。

---

## 8. ファイル構成と責務

| ファイル | 責務 | 主要関数/クラス |
|---|---|---|
| `public.py` | 外部API + 型定義 | `TTMResult`, `advance_temperatures()` |
| `config.py` | パラメータ定義 | `TTMConfig` |
| `solver.py` | 更新シーケンス（層統合） | `advance_temperatures_impl()`, 物性取得 → RHS計算 → 相転移判定 → オイラー更新 |

相転移処理は `phase_transition.apply_phase_transitions()` に委譲する。物性計算は `material_properties` に委譲する。

---

## 9. テスト方針

```bash
docker compose run --rm sim pytest tests/test_ttm/ -v
```

### 9.1 単体テストケース

| テスト名 | 内容 | 検証方法 |
|---|---|---|
| `test_thermal_equilibrium` | $S=0$, $G>0$, $T_e \neq T_l$ → 平衡 | $T_e \approx T_l$ に収束 |
| `test_independent_diffusion` | $G=0$ → Te, Tl が独立に拡散 | 解析解と比較 |
| `test_melting_latent_heat` | steady $S$ → $T_l$ が $T_m$ で停滞 | 潜熱期間 $\approx L_m / S$ |
| `test_vaporizing_latent_heat` | 融解完了後 → $T_l$ が $T_b$ で停滞 | 潜熱期間 $\approx L_v / S$ |
| `test_phase_state_transition` | SOLID→MELTING→LIQUID→VAPORIZING→VAPOR | 各状態の遷移条件 |
| `test_Ce_phase_switch` | 固相→液相で $C_e$ 計算式が切り替わる | 値の不連続を確認 |
| `test_Kl_phase_switch` | 固相→液相で $K_l$ 計算式が切り替わる | 値の不連続を確認 |
| `test_immutability` | 入力配列が変更されないこと | `np.array_equal(input_before, input_after)` |
| `test_energy_conservation` | $S=0$, 断熱 → 総エネルギー保存 | $\sum(C_e T_e + C_l T_l) \approx$ const |
