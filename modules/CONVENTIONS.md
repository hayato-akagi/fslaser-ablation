# modules/CONVENTIONS.md — 全モジュール共通規約

本ドキュメントは、全ドメインモジュール（carrier, optics, ttm, ablation, euler_fdm）の
**実装者が必ず参照すべき共通規約**を定義する。各モジュールの README.md はこの規約を前提とする。

---

## 0. 物理定数の共有（`modules/constants.py`）

### 0.1 目的

複数モジュールで使用される**不変の物理定数**を一箇所に集約し、値の重複・不整合を防ぐ。

### 0.2 構成

```python
from modules.constants import PHYSICAL, SILICON, LASER_1030NM
```

| インスタンス | 内容 | 例 |
|---|---|---|
| `PHYSICAL` | 基本物理定数（k_B, e, ε₀, m_e, c） | `PHYSICAL.k_B` → 1.381e-23 J/K |
| `SILICON` | シリコン物性（Tm, Tb, Tcr, ρ, εr, τe, γ, β） | `SILICON.T_m` → 1687.0 K |
| `LASER_1030NM` | 1030nmレーザー固有（λ, hω, tp） | `LASER_1030NM.photon_energy_eV` → 1.2036 |

### 0.3 ルール

- **`constants.py` にはロジックを書かない**（定数値と `@property` による単純変換のみ）
- `dataclass(frozen=True)` で不変性を保証
- 各モジュールの `config.py` はデフォルト値として `constants.py` の値を参照する
- シミュレーション固有パラメータ（フルエンス、グリッド数等）は `constants.py` に含めない
- `constants.py` の利用は `public.py` 経由ルールの**例外**（定数はドメインロジックではないため）

### 0.4 各モジュール config.py での参照例

```python
from modules.constants import PHYSICAL, SILICON

class CarrierConfig(BaseModel):
    gamma: float = SILICON.gamma_auger      # 定数からデフォルト取得
    k_B_eV: float = PHYSICAL.k_B_eV
    dz: float                               # euler_fdm から注入（定数ではない）
```

---

## 1. グリッド座標系

### 1.1 座標の定義

```
z[0]          z[1]          z[2]     ...   z[N-1]
  |             |             |               |
  ▼             ▼             ▼               ▼
  表面 -------> 深さ方向 -----------------> 底面 (5 µm)
```

- **z軸の正方向**: 材料表面から深さ方向（材料内部へ向かう）
- **`z[0]`**: 材料表面（深さ 0 nm）
- **`z[i]`**: 深さ `i × Δz` nm の位置
- **`z[N-1]`**: 材料底面（深さ `(N-1) × Δz` nm）
- レーザーは `z[0]`（表面）から入射する

### 1.2 デフォルトグリッドパラメータ

| パラメータ | 変数名 | デフォルト値 | 単位 |
|---|---|---|---|
| グリッド数 | `n_z` | 1000 | — |
| グリッド間隔 | `dz` | 5.0 | nm |
| 計算領域長 | `L_z` | 5000.0 (`= n_z × dz`) | nm |

### 1.3 グリッド情報の所有と注入

- **グリッドパラメータ（`n_z`, `dz`）は `euler_fdm/config.py` の `GridConfig` が唯一の所有者**
- 各ドメインモジュールはグリッド数をハードコードしない
- 各ドメインの関数は**配列の `len()` からグリッド数を実行時に取得**する
- `dz` が必要な場合は、各ドメインの `config` に注入するか、引数として渡す

---

## 2. 配列規約

### 2.1 データ型

全ての空間分布量は **`numpy.ndarray`（dtype=`np.float64`）** で扱う。

```python
import numpy as np
from numpy.typing import NDArray

# 型エイリアス（全モジュール共通）
SpatialArray = NDArray[np.float64]  # shape: (n_z,)
```

### 2.2 Pydantic モデルでの numpy 配列

Pydantic v2 はデフォルトで `numpy.ndarray` をサポートしない。以下の方法で対処する：

```python
from pydantic import BaseModel, ConfigDict

class SomeResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    Te: NDArray[np.float64]   # shape: (n_z,)
    Tl: NDArray[np.float64]   # shape: (n_z,)
```

- `model_config = ConfigDict(arbitrary_types_allowed=True)` を **全ての Result / Config モデルに付与**
- ドキュメントコメントで `shape` を明記する

### 2.3 スカラー vs 配列

| 量 | 型 | 備考 |
|---|---|---|
| 空間分布量（`Te(z)`, `Tl(z)`, `ne(z)`, `I(z)` 等） | `NDArray[np.float64]` shape `(n_z,)` | グリッド点ごとの値 |
| 時刻 `t` | `float` | 単位: s |
| 時間刻み `dt` | `float` | 単位: s |
| 表面反射率 `R` | `float` | スカラー（表面 z=0 の値のみ） |
| アブレーション深さ | `float` | 単位: m（SI） |
| フルエンス `F` | `float` | 単位: J/cm² |

---

## 3. 単位系

### 3.1 内部計算の単位（CGS寄り — 論文準拠）

論文の全パラメータは **CGS系** で記述されているため、内部計算も**論文と同一の単位系**を使用する。
SI変換は最終出力（`SimulationResult`）でのみ行い、内部モジュール間のデータ受け渡しは以下に統一する。

| 物理量 | 内部単位 | 備考 |
|---|---|---|
| 長さ | cm | `dz = 5e-7 cm`（=5 nm） |
| 時間 | s | `dt` は秒。`tp = 421e-15 s`（=421 fs） |
| 温度 | K | — |
| キャリア密度 `ne` | cm⁻³ | — |
| レーザー強度 `I` | W/cm² | — |
| 熱源項 `S` | W/cm³ | — |
| 熱容量 `C` | J/(cm³·K) | — |
| 熱伝導率 `K` | W/(cm·K) | — |
| 結合因子 `G` | W/(cm³·K) | — |
| フルエンス `F` | J/cm² | — |
| 潜熱 | J/cm³ | — |
| 吸収係数 `α` | cm⁻¹ | — |
| TPA係数 `β` | cm/GW → 内部: cm·s/erg | 変換注意 |
| 拡散係数 `D0` | cm²/s | — |

### 3.2 単位変換の責務

- **各ドメインの `config.py`** がパラメータの単位変換を担当する
- `public.py` 経由で外部に公開する値は上記内部単位系のまま渡す
- **最終出力（`SimulationResult`）のみ**、必要に応じてSI変換する

---

## 4. 相状態の定義

```python
from enum import IntEnum

class PhaseState(IntEnum):
    """各グリッド点の相状態"""
    SOLID = 0
    MELTING = 1    # Tl == Tm かつ潜熱蓄積中
    LIQUID = 2
    VAPORIZING = 3 # Tl == Tb かつ潜熱蓄積中
    VAPOR = 4
```

- `PhaseState` は **`modules/__init__.py`** で定義し、全モジュールが参照する共有型
- 空間配列としては `NDArray[np.int32]` shape `(n_z,)` で保持し、値は `PhaseState` の整数値

---

## 5. 境界条件

### 5.1 空間境界（z方向）

| 境界 | 位置 | 条件 | 適用対象 |
|---|---|---|---|
| 表面 (z=0) | `z[0]` | **断熱** (Neumann: $\partial T/\partial z = 0$) | Te, Tl, ne |
| 底面 (z=L) | `z[N-1]` | **断熱** (Neumann: $\partial T/\partial z = 0$) | Te, Tl, ne |

FDMでの実装: ゴーストセル法またはミラー法（`T[-1] = T[0]`, `T[N] = T[N-1]`）

### 5.2 時間境界（初期条件）

| 量 | 初期値 | 単位 |
|---|---|---|
| $T_e$ | 300.0 | K |
| $T_l$ | 300.0 | K |
| $n_e$ | $1.0 \times 10^{12}$ | cm⁻³ |
| `phase_state` | `PhaseState.SOLID` (全グリッド) | — |
| `latent_heat_accumulated` | 0.0 (全グリッド) | J/cm³ |

---

## 6. モジュール間インターフェース（public.py の型契約）

### 6.1 呼び出し順序（1タイムステップ内）

```
euler_fdm.layer1_sequence:
    ┌──────────────────────────────────────────────────────────┐
    │  Step 1: optics.public.compute_laser_field(...)          │
    │          → OpticsResult                                  │
    │                                                          │
    │  Step 2: carrier.public.advance_carrier_density(...)     │
    │          → CarrierResult                                 │
    │                                                          │
    │  Step 3: ttm.public.advance_temperatures(...)            │
    │          → TTMResult                                     │
    │                                                          │
    │  Step 4: ablation.public.evaluate_ablation(...)          │
    │          → AblationResult                                │
    │                                                          │
    │  Step 5: 状態ベクトル更新                                  │
    └──────────────────────────────────────────────────────────┘
```

### 6.2 各 Result 型の正式定義

```python
# --- optics/public.py ---
class OpticsResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    intensity: NDArray[np.float64]    # I(z): shape (n_z,), 単位 W/cm²
    source_term: NDArray[np.float64]  # S(z) = α_total * I(z): shape (n_z,), 単位 W/cm³
    reflectivity: float               # R(0,t): 表面反射率, 無次元
    alpha_fca: NDArray[np.float64]    # α_FCA(z): shape (n_z,), 単位 cm⁻¹

# --- carrier/public.py ---
class CarrierResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    ne: NDArray[np.float64]           # 更新後キャリア密度: shape (n_z,), 単位 cm⁻³
    dne_dt: NDArray[np.float64]       # キャリア密度の時間変化率: shape (n_z,), 単位 cm⁻³/s

# --- ttm/public.py ---
class TTMResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    Te: NDArray[np.float64]                      # 更新後電子温度: shape (n_z,), 単位 K
    Tl: NDArray[np.float64]                      # 更新後格子温度: shape (n_z,), 単位 K
    phase_state: NDArray[np.int32]               # 相状態: shape (n_z,), PhaseState の整数値
    latent_heat_accumulated: NDArray[np.float64]  # 潜熱蓄積量: shape (n_z,), 単位 J/cm³

# --- ablation/public.py ---
class AblationResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    ablation_depth: float                   # アブレーション深さ: 単位 cm（内部単位系）
    ablated_mask: NDArray[np.bool_]         # shape (n_z,), True=アブレーション済
```

### 6.3 各 public 関数のシグネチャ

```python
# --- optics/public.py ---
def compute_laser_field(
    ne: NDArray[np.float64],        # (n_z,) キャリア密度 [cm⁻³]
    Tl: NDArray[np.float64],        # (n_z,) 格子温度 [K]
    phase_state: NDArray[np.int32], # (n_z,) 相状態
    t: float,                       # 現在時刻 [s]
    config: "OpticsConfig",
) -> OpticsResult: ...

# --- carrier/public.py ---
def advance_carrier_density(
    ne: NDArray[np.float64],        # (n_z,) 現在のキャリア密度 [cm⁻³]
    intensity: NDArray[np.float64], # (n_z,) レーザー強度 [W/cm²]
    Te: NDArray[np.float64],        # (n_z,) 電子温度 [K]
    Tl: NDArray[np.float64],        # (n_z,) 格子温度 [K]
    phase_state: NDArray[np.int32], # (n_z,) 相状態
    dt: float,                      # 時間刻み [s]
    config: "CarrierConfig",
) -> CarrierResult: ...

# --- ttm/public.py ---
def advance_temperatures(
    Te: NDArray[np.float64],                      # (n_z,) 電子温度 [K]
    Tl: NDArray[np.float64],                      # (n_z,) 格子温度 [K]
    ne: NDArray[np.float64],                      # (n_z,) キャリア密度 [cm⁻³]
    dne_dt: NDArray[np.float64],                  # (n_z,) キャリア時間変化率 [cm⁻³/s]
    source_term: NDArray[np.float64],             # (n_z,) 熱源項 S(z) [W/cm³]
    phase_state: NDArray[np.int32],               # (n_z,) 現在の相状態
    latent_heat_accumulated: NDArray[np.float64], # (n_z,) 現在の潜熱蓄積量 [J/cm³]
    dt: float,                                    # 時間刻み [s]
    config: "TTMConfig",
) -> TTMResult: ...

# --- ablation/public.py ---
def evaluate_ablation(
    Tl: NDArray[np.float64],        # (n_z,) 格子温度 [K]
    dz: float,                      # グリッド間隔 [cm]
    config: "AblationConfig",
) -> AblationResult: ...
```

---

## 7. アブレーション判定の詳細規約

### 7.1 判定条件

温度のみで判定する。`phase_state` はアブレーション判定には**使用しない**。

```
アブレーション対象: Tl[i] >= 0.9 * Tcr のグリッド点
```

### 7.2 連続性ルール

表面（`z[0]`）から**連続して**閾値を超えるグリッド点のみをアブレーション領域とする。
途中に閾値未満のグリッドがある場合、その深さ以降は無視する。

```
例: Tl = [8000, 8000, 8000, 5000, 8000, ...]
     mask = [True, True, True, False, False, ...]  ← z[4]は不連続のため除外
     ablation_depth = 3 × dz
```

### 7.3 累積ルール

- `evaluate_ablation()` は**呼び出し時点の Tl から毎回独立に**アブレーション深さを計算する
- 時間方向の累積管理（前ステップとの比較・最大値保持）は **`euler_fdm` が担当する**
- `ablation` モジュールは状態を持たない（ステートレス）

---

## 8. ファイル構成テンプレート（CLAUDE.md 準拠）

### 8.1 modules/ 内のドメインモジュール

```
modules/<domain>/
├── __init__.py
├── README.md             # 本仕様書
├── public.py             # Step 1: 外部窓口（Result型 + API関数）
├── protocols.py          # Step 2: 内部Protocol定義
├── config.py             # Step 4: Pydantic パラメータクラス
├── layer1_sequence.py    # Step 3: 最上位シーケンス
├── layer2_logic.py       # Step 5: 中間ロジック
└── layer3_detail.py      # Step 5: 数値計算詳細
```

### 8.2 プロジェクト全体構成

```
fslaser-ablation-sim/
├── modules/              # ドメインロジック（物理計算）
│   ├── constants.py
│   ├── CONVENTIONS.md
│   ├── carrier/
│   ├── optics/
│   ├── ttm/
│   ├── ablation/
│   └── euler_fdm/
├── views/                # プレゼンテーション層（保存・可視化）
│   ├── io.py             # SimulationResult の保存・読込
│   └── plotting.py       # matplotlib グラフ生成
├── results/              # シミュレーション出力（git管理外）
│   └── {run_id}/
│       ├── metadata.json
│       ├── arrays.npz
│       └── plots/
└── tests/
```

### 8.3 views/ 層のルール

- `views/` は `modules/` の外側に配置する（CLAUDE.md 準拠）
- `views/` から `modules/` を参照する方向のみ許可（逆方向の参照禁止）
- `views/io.py`: `SimulationResult` を `results/` に保存・読込
- `views/plotting.py`: `results/` からデータを読み込みグラフを生成・保存
- 物理計算や判定ロジックを `views/` に記述することは禁止

---

## 9. テスト規約

### 9.1 テストディレクトリ構成

```
tests/
├── conftest.py               # 共通フィクスチャ
├── test_carrier/             # carrier モジュール単体テスト
├── test_optics/              # optics モジュール単体テスト  
├── test_ttm/                 # ttm モジュール単体テスト
├── test_ablation/            # ablation モジュール単体テスト
├── test_euler_fdm/           # euler_fdm モジュール単体テスト
└── test_integration/         # 全連成テスト
```

### 9.2 実行方法

```bash
# 全テスト
docker compose run --rm sim pytest tests/ -v

# モジュール単体
docker compose run --rm sim pytest tests/test_ablation/ -v

# 統合テストのみ
docker compose run --rm sim pytest tests/test_integration/ -v
```

### 9.3 Mock 戦略

- 各モジュールの単体テストでは、他モジュールの Result 型を直接生成してテストデータとする
- Protocol に対する Mock を注入して依存を遮断
- `config` はテスト用の値を指定して生成する（デフォルト値に依存しない）
