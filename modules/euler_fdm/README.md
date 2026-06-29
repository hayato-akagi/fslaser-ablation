# modules/euler_fdm — オイラー法・有限差分オーケストレーター

> **前提**: 本ドキュメントは `modules/CONVENTIONS.md` の全規約に準拠する。  
> 実装前に必ず CONVENTIONS.md を参照すること。

---

## 1. 責務

**前進オイラー法 (Explicit Euler)** と **1D有限差分法 (FDM)** を用いて、
全ドメインモジュール（carrier, optics, ttm, ablation）を時間方向に連成し、
フェムト秒レーザーアブレーション・シミュレーション全体を実行する。

本モジュールの責務:
- **状態ベクトルの一元管理**（$n_e$, $T_e$, $T_l$, `phase_state`, `latent_heat`）
- **タイムループの制御**（時間刻み決定、開始・終了判定）
- **各ドメインモジュールの呼び出し順序制御**
- **アブレーション深さの時間累積管理**
- **シミュレーション結果の収集・出力**

---

## 2. 状態ベクトル

euler_fdm が所有・管理する全状態変数:

| 変数名 | 型 | shape | 初期値 | 単位 | 更新元 |
|---|---|---|---|---|---|
| `ne` | `NDArray[np.float64]` | `(n_z,)` | $1.0 \times 10^{12}$ | cm⁻³ | carrier |
| `Te` | `NDArray[np.float64]` | `(n_z,)` | 300.0 | K | ttm |
| `Tl` | `NDArray[np.float64]` | `(n_z,)` | 300.0 | K | ttm |
| `phase_state` | `NDArray[np.int32]` | `(n_z,)` | `PhaseState.SOLID` (=0) | — | ttm |
| `latent_heat_acc` | `NDArray[np.float64]` | `(n_z,)` | 0.0 | J/cm³ | ttm |
| `max_ablation_depth` | `float` | — | 0.0 | cm | ablation (累積) |
| `cumulative_ablated_mask` | `NDArray[np.bool_]` | `(n_z,)` | all False | — | ablation (累積) |
| `t` | `float` | — | `t_start` | s | self |
| `dTl_dt_prev` | `NDArray[np.float64]` | `(n_z,)` | 0.0 | K/s | self (計算) |

---

## 3. メインタイムループ

### 3.1 1タイムステップの処理順序

```python
def _execute_one_iteration(state, config, history):
    """solver.py の概念的な擬似コード（1イテレーション）"""
    
    # dt 決定のために optics を先行計算（Step 1a）
    optics_result_pre = compute_laser_field(
        ne=state["ne"], Tl=state["Tl"],
        phase_state=state["phase_state"], t=state["t"],
        config=state["optics_config"],
    )
    
    # CFL + レーザー加熱率による安全な dt を計算
    dt = compute_safe_dt(Te, Tl, ne, dz, dt_max,
                         source_term=optics_result_pre.source_term)
    
    # Step 1: レーザー場計算（再計算）
    optics_result = compute_laser_field(...)
    
    # Step 2: キャリア密度更新
    carrier_result = advance_carrier_density(
        ne=state["ne"], intensity=optics_result.intensity,
        Te=state["Te"], Tl=state["Tl"],
        phase_state=state["phase_state"], dt=dt,
        config=state["carrier_config"],
    )
    
    # Step 3: 温度更新（TTM）。TTM には dTl_dt_prev は渡さない（第5項は省略）
    Tl_before = state["Tl"].copy()
    ttm_result = advance_temperatures(
        Te=state["Te"], Tl=state["Tl"],
        ne=carrier_result.ne,       # ← 更新後の ne を使用
        dne_dt=carrier_result.dne_dt,
        source_term=optics_result.source_term,
        phase_state=state["phase_state"],
        latent_heat_accumulated=state["latent_heat_acc"],
        dt=dt, config=state["ttm_config"],
    )
    
    # Step 4: アブレーション判定
    ablation_result = evaluate_ablation(
        Tl=ttm_result.Tl, dz=config.grid.dz,
        config=state["ablation_config"],
    )
    
    # Step 5: 状態更新
    state["ne"] = carrier_result.ne
    state["Te"] = ttm_result.Te
    state["Tl"] = ttm_result.Tl
    state["phase_state"] = ttm_result.phase_state
    state["latent_heat_acc"] = ttm_result.latent_heat_accumulated
    state["max_ablation_depth"] = max(state["max_ablation_depth"],
                                      ablation_result.ablation_depth)
    state["cumulative_ablated_mask"] |= ablation_result.ablated_mask
    state["dTl_dt_prev"] = (ttm_result.Tl - Tl_before) / dt  # 計算済みだが未使用
    state["t"] += dt
```

注意: `optics` は1イテレーション内で **2回**呼ばれる（dt計算用の先行計算 + Step 1 本計算）。

### 3.2 タイムループ全体

```python
while state["t"] < config.time.t_end:
    _execute_one_iteration(state, config, history)
```

### 3.3 時間パラメータ

| パラメータ | 意味 | デフォルト値 | 単位 |
|---|---|---|---|
| `t_start` | シミュレーション開始時刻 | 0 | s |
| `t_end` | シミュレーション終了時刻 | $500 \times 10^{-12}$ (=500 ps) | s |
| `t_p` | パルス幅 (FWHM) | $421 \times 10^{-15}$ | s |

- **$t = 0$ はパルス中心**
- 初期条件は既に平衡状態として定義されているため、t = 0 から開始

---

## 4. CFL安定性条件

明示的オイラー法の安定性を保証するために:

$$
\Delta t \leq \frac{(\Delta z)^2}{2 \max_i(D_i)}
$$

ここで $D_i$ は各グリッド点での最大拡散率。考慮すべき拡散係数:
1. **キャリア拡散**: $D_0 = 18 \cdot T_{rm} / \max(T_l, T_{rm})$
2. **格子熱拡散**: $K_l / C_l$（動的計算）
3. **電子熱拡散**: $K_e / C_e$（$n_e < 10^{15}$ cm⁻³ 時は `dt_max` から逆算した上限で制限）

さらに**レーザー加熱率**による制限も追加する:
- $S_{\max} > 0$ の場合、1ステップあたりの温度上昇が 1000 K を超えないよう dt を制限

```python
def compute_safe_dt(Te, Tl, ne, dz, dt_max,
                    source_term=None) -> float:
    """CFL条件 + レーザー加熱率に基づく安全な時間刻みを計算"""
    # キャリア拡散
    Tl_safe = np.maximum(Tl, T_room)
    D_carrier = 18.0 * T_room / Tl_safe
    
    # 格子熱拡散（動的計算）
    Cl = 1.978 + 3.54e-4 * Tl - 3.68 * Tl**(-2)
    Kl = np.where(Tl < T_m, 1585.0 * Tl**(-1.23), 0.5 + 2.9e-4 * (Tl - T_m))
    D_lattice = np.maximum(Kl, 0.0) / np.maximum(Cl, 1e-10)
    
    # 電子熱拡散（ne が低い場合は上限制限）
    Ce = material_properties.compute_thermal_capacity_electron(Te, ne, phase_state)
    Ke = material_properties.compute_thermal_conductivity_electron(Te, phase_state)
    D_electron = Ke / Ce
    if ne.max() < 1e15:
        D_electron_max = 0.5 * dz**2 / (2.0 * dt_max)
        D_electron = np.minimum(D_electron, D_electron_max)
    
    D_max = max(D_carrier.max(), D_lattice.max(), D_electron.max(), 1e-30)
    dt_cfl = 0.5 * dz**2 / (2.0 * D_max)  # 安全係数 0.5 × CFL係数 0.5
    
    # レーザー加熱率制限
    if source_term is not None and source_term.max() > 1e-30:
        dt_source = Ce.min() * 1000.0 / source_term.max()
        return min(dt_cfl, dt_source, dt_max)
    
    return min(dt_cfl, dt_max)
```

---

## 5. グリッド設定の配布

euler_fdm が所有する `GridConfig` からの `dz` の配布方法:

```python
# euler_fdm が各ドメイン Config を生成する際に dz を注入
carrier_config = CarrierConfig(dz=grid_config.dz)
optics_config = OpticsConfig(dz=grid_config.dz, fluence=sim_config.fluence)
ttm_config = TTMConfig(dz=grid_config.dz)
# ablation は dz を関数引数で受け取る（Config に含めない）
```

---

## 6. public API シグネチャ

```python
def run_simulation(config: EulerFDMConfig) -> SimulationResult:
    """シミュレーション全体を実行。

    1. 各ドメイン Config を生成・注入
    2. 状態ベクトルを初期化
    3. タイムループを実行
    4. 結果を収集して返す
    """
```

### 6.1 SimulationResult の定義

```python
class SimulationResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # 最終状態
    Te_final: NDArray[np.float64]          # (n_z,) 最終電子温度 [K]
    Tl_final: NDArray[np.float64]          # (n_z,) 最終格子温度 [K]
    ne_final: NDArray[np.float64]          # (n_z,) 最終キャリア密度 [cm⁻³]
    
    # アブレーション結果
    ablation_depth_cm: float               # 最終アブレーション深さ [cm]
    ablation_depth_nm: float               # 最終アブレーション深さ [nm]（SI変換済み）
    ablated_mask: NDArray[np.bool_]        # (n_z,) 累積アブレーションマスク
    
    # 時間履歴（スナップショット）
    time_points: NDArray[np.float64]       # (n_snapshots,) 記録時刻 [s]
    Te_surface_history: NDArray[np.float64] # (n_snapshots,) z=0 の Te 履歴
    Tl_surface_history: NDArray[np.float64] # (n_snapshots,) z=0 の Tl 履歴
    ne_surface_history: NDArray[np.float64] # (n_snapshots,) z=0 の ne 履歴
    reflectivity_history: NDArray[np.float64] # (n_snapshots,) 表面反射率履歴
    alpha_fca_surface_history: NDArray[np.float64] # (n_snapshots,) z=0 の α_FCA 履歴 [cm⁻¹]
    auger_term_surface_history: NDArray[np.float64] # (n_snapshots,) z=0 の γn_e³ 履歴 [cm⁻³/s]
    ablation_depth_history: NDArray[np.float64] # (n_snapshots,) 各ステップの深さ [nm]
    
    # メタデータ
    total_steps: int                       # 総ステップ数
    fluence: float                         # 入力フルエンス [J/cm²]
```

---

## 7. config.py のパラメータ一覧

```python
class GridConfig(BaseModel):
    """グリッド設定（全モジュールの唯一の真実源）"""
    n_z: int = 1000               # グリッド数
    dz: float = 5e-7              # グリッド間隔 [cm] (= 5 nm)
    
    @property
    def L_z(self) -> float:
        """計算領域長 [cm]"""
        return self.n_z * self.dz

class TimeConfig(BaseModel):
    """時間設定"""
    t_end: float = 500e-12         # 終了時刻 [s] (= 500 ps)
    dt_max: float = 1e-15          # 最大時間刻み [s] (= 1 fs)
    snapshot_interval: int = 100   # スナップショット記録間隔 [ステップ数]
    
    # t_start = 0（パルス中心から開始）
    pulse_duration: float = 421e-15  # [s]
    
    @property
    def t_start(self) -> float:
        return 0.0

class InitialCondition(BaseModel):
    """初期条件"""
    Te_init: float = 300.0          # [K]
    Tl_init: float = 300.0          # [K]
    ne_init: float = 1.0e12         # [cm⁻³]

class EulerFDMConfig(BaseModel):
    """最上位設定（全てを包含）"""
    grid: GridConfig = GridConfig()
    time: TimeConfig = TimeConfig()
    initial: InitialCondition = InitialCondition()
    fluence: float                  # 入力フルエンス [J/cm²]（必須）
```

---

## 8. スナップショット記録

### 8.1 記録対象

全タイムステップの全データを保存するとメモリが不足するため、`snapshot_interval` ステップごとに以下を記録:

- 表面（z=0）の $T_e$, $T_l$, $n_e$, $R$, $\alpha_{FCA}$, $\gamma n_e^3$
- アブレーション深さ

### 8.2 全空間プロファイルの記録

最終状態のみ全空間プロファイルを `SimulationResult` に含める。
中間ステップの全空間データが必要な場合は、将来的にコールバック機構で対応。

### 8.3 データ永続化

`run_simulation()` が返す `SimulationResult` は、呼び出し側で `views/io.py` の `save_result()` を使い `results/` ディレクトリに保存する。

- **保存先**: `results/{YYYYMMDD_HHMMSS}_F{fluence:.2f}/`
- **保存形式**: `metadata.json`（設定・メタ情報）+ `arrays.npz`（numpy 配列）
- **責務分離**: `euler_fdm` は `SimulationResult` を返すだけ。保存・可視化は `views/` 層の責務

詳細は `views/README.md` を参照。

---

## 9. ファイル構成と責務

| ファイル | 責務 | 主要関数/クラス |
|---|---|---|
| `public.py` | 外部API + 型定義 | `SimulationResult`, `run_simulation()` |
| `config.py` | 全体設定定義 | `GridConfig`, `TimeConfig`, `InitialCondition`, `EulerFDMConfig` |
| `solver.py` | メインループ全統合 | `run_simulation_impl()`, `_execute_time_loop()`, `_execute_one_iteration()`, `_execute_one_step()`, `compute_safe_dt()`, `create_domain_configs()`, `initialize_state_vectors()` |

---

## 10. テスト方針

```bash
docker compose run --rm sim pytest tests/test_euler_fdm/ -v
```

### 10.1 単体テストケース

| テスト名 | 内容 | 検証方法 |
|---|---|---|
| `test_cfl_condition` | 既知パラメータ → 安全な dt | $dt \leq dz^2 / (2 D_{max})$ |
| `test_initial_state` | 状態ベクトル初期化 | 全配列が初期値で埋まっている |
| `test_step_order` | Mock化した各moduleを呼び出し順序検証 | optics→carrier→ttm→ablation の順 |
| `test_state_update` | 1ステップ後の状態が Result の値で更新 | before != after |
| `test_ablation_cumulative` | 2ステップで深さが増加 | max_depth が更新される |
| `test_config_injection` | 各ドメイン Config に dz が正しく注入 | config.dz == grid_config.dz |
| `test_snapshot_interval` | N ステップ実行 → 記録数が正しい | len(history) == N / interval |

### 10.2 統合テスト（`tests/test_integration/`）

| テスト名 | 内容 | 検証方法 |
|---|---|---|
| `test_melting_threshold` | $F = 0.25$ J/cm² → $T_l \geq T_m$ | 融点到達を確認 |
| `test_vaporization_threshold` | $F = 0.80$ J/cm² → $T_l \geq T_b$ | 沸点到達を確認 |
| `test_ablation_threshold` | $F = 1.5$ J/cm² → ablation_depth > 0 | Phase explosion 発生 |
| `test_ablation_depth_trend` | $F = 1.5 \sim 3.06$ J/cm² → 深さが増加 | 単調増加 |
| `test_ablation_depth_values` | 論文 Figure 9 の値と比較 | 相対誤差 < 30% |
