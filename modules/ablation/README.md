# modules/ablation — Phase Explosion アブレーション判定

> **前提**: 本ドキュメントは `modules/CONVENTIONS.md` の全規約に準拠する。  
> 実装前に必ず CONVENTIONS.md を参照すること。

---

## 1. 責務

格子温度 $T_l(z)$ のプロファイルから、**Phase Explosion 条件**に基づいてアブレーション深さを算出する。
本モジュールは**ステートレス**（状態を持たない純粋な判定器）であり、呼び出しごとに独立に深さを計算する。

本モジュールは euler_fdm の **Step 4**（ttm の後、状態更新の前）で呼ばれる。

---

## 2. アブレーション判定

### 2.1 判定条件

$$
T_l(z_i) \geq 0.9 \times T_{cr} = 0.9 \times 7925 = 7132.5 \text{ K}
$$

- 判定は **温度のみ** で行う
- `phase_state` は判定に**使用しない**（`evaluate_ablation` の引数にも含まない）
- 閾値 $0.9 \times T_{cr}$ は `AblationConfig` から取得する

### 2.2 グリッド座標の方向

```
z[0]   z[1]   z[2]   ...   z[N-1]
 ▼      ▼      ▼             ▼
表面                         底面 (5 µm)
```

- **`z[0]` = 材料表面**（レーザー入射面）
- **`z[N-1]` = 材料底面**（深さ 5 µm）
- アブレーションは表面から始まり、深さ方向に進行する

### 2.3 連続性ルール

**表面 `z[0]` から連続して**閾値を超えるグリッド点のみをアブレーション対象とする。

```
ケース1: 連続
  Tl   = [8000, 8000, 8000, 5000, 3000, ...]
  mask = [True, True, True, False, False, ...]
  ablation_depth = 3 × dz

ケース2: 不連続（中間に閾値未満がある）
  Tl   = [8000, 8000, 5000, 8000, 3000, ...]
  mask = [True, True, False, False, False, ...]  ← z[3] は不連続のため除外
  ablation_depth = 2 × dz

ケース3: 表面が閾値未満
  Tl   = [5000, 8000, 8000, 3000, ...]
  mask = [False, False, False, False, ...]        ← 表面が未達なら深さ = 0
  ablation_depth = 0

ケース4: 全グリッドが閾値未満
  ablation_depth = 0
  mask = all False

ケース5: ちょうど閾値
  Tl[i] == 7132.5 → True（閾値以上：>= で判定）
```

### 2.4 ablation_depth の計算

```python
# 表面から連続して閾値を超えるグリッド数をカウント
count = 0
for i in range(n_z):
    if Tl[i] >= threshold:
        count += 1
    else:
        break

ablation_depth = count * dz  # [cm]（内部単位系）
```

---

## 3. public API シグネチャ

```python
def evaluate_ablation(
    Tl: NDArray[np.float64],   # (n_z,) 格子温度 [K]
    dz: float,                 # グリッド間隔 [cm]
    config: AblationConfig,
) -> AblationResult:
    """Phase Explosion に基づくアブレーション深さを算出。

    呼び出しタイミング: euler_fdm の Step 4（ttm の後）

    ステートレス: 前ステップの結果に依存しない。
    時間方向のアブレーション深さ最大値の追跡は euler_fdm が行う。
    """
```

### 3.1 引数の詳細

| 引数 | 型 | shape | 単位 | 説明 |
|---|---|---|---|---|
| `Tl` | `NDArray[np.float64]` | `(n_z,)` | K | 格子温度。`z[0]`=表面, `z[N-1]`=底面 |
| `dz` | `float` | — | cm | グリッド間隔。`euler_fdm` の `GridConfig` から取得 |
| `config` | `AblationConfig` | — | — | 閾値パラメータ |

### 3.2 AblationResult の定義

```python
class AblationResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    ablation_depth: float              # アブレーション深さ [cm]（内部単位系）
    ablated_mask: NDArray[np.bool_]    # shape (n_z,), True = アブレーション済
```

### 3.3 ablation_depth の単位

- **内部単位系: cm**（CONVENTIONS.md §3 準拠）
- 最終出力（`SimulationResult`）でのみ nm への変換を行う（euler_fdm の責務）

### 3.4 ablated_mask の用途

- `ablated_mask` は **euler_fdm が状態管理に使用する** ための出力
- euler_fdm は各タイムステップの `ablated_mask` を累積 OR して、全体のアブレーションマスクを管理する
- **ablation モジュール自体はマスクの累積管理をしない**（ステートレス）
- ablated_mask の True/False は連続性ルール（§2.3）に基づく

---

## 4. 時間方向の累積管理（euler_fdm の責務 — 参考情報）

本セクションは euler_fdm 側の仕様だが、ablation 実装者の理解のために記載する。

```python
# euler_fdm/solver.py（概念）
max_ablation_depth = 0.0
cumulative_mask = np.zeros(n_z, dtype=np.bool_)

for step in time_steps:
    # ... optics, carrier, ttm ...
    result = ablation.evaluate_ablation(Tl, dz, ablation_config)
    
    max_ablation_depth = max(max_ablation_depth, result.ablation_depth)
    cumulative_mask |= result.ablated_mask
```

- アブレーション深さは**全タイムステップの最大値**が最終結果
- 一度アブレーション済みと判定されたグリッドは、温度が下がっても取り消されない

---

## 5. config.py のパラメータ一覧

```python
class AblationConfig(BaseModel):
    T_cr: float = 7925.0            # 臨界温度 [K]
    threshold_fraction: float = 0.9  # 閾値係数（0.9 Tcr）
    
    @property
    def threshold_temperature(self) -> float:
        """アブレーション閾値温度 [K]"""
        return self.threshold_fraction * self.T_cr  # = 7132.5 K
```

### 5.1 グリッドサイズはハードコードしない

- `dz` は `AblationConfig` に含めない
- `evaluate_ablation()` の引数として受け取る
- `n_z` は `Tl` 配列の `len()` から実行時に取得する

---

## 6. ファイル構成と責務

| ファイル | 責務 | 主要関数/クラス |
|---|---|---|
| `public.py` | 外部API + 型定義 | `AblationResult`, `evaluate_ablation()` |
| `config.py` | パラメータ定義 | `AblationConfig` |
| `solver.py` | 評価シーケンス（層統合） | `evaluate_ablation_sequence()`, 閾値取得 → マスク生成 → 連続カウント → 深さ計算 |

---

## 7. テスト方針

```bash
docker compose run --rm sim pytest tests/test_ablation/ -v
```

### 7.1 単体テストケース

| テスト名 | 入力 Tl | 期待結果 | 検証内容 |
|---|---|---|---|
| `test_no_ablation` | 全グリッド 5000 K | `depth=0`, `mask=all False` | 閾値未満 |
| `test_surface_ablation` | `z[0..2]=8000`, 他=3000 | `depth=3*dz`, `mask[0..2]=True` | 表面から3グリッド |
| `test_discontinuous` | `z[0..1]=8000, z[2]=5000, z[3]=8000` | `depth=2*dz`, `mask[0..1]=True, mask[3]=False` | 連続性ルール |
| `test_surface_below` | `z[0]=5000, z[1..3]=8000` | `depth=0`, `mask=all False` | 表面未達 |
| `test_exact_threshold` | `z[0]=7132.5` (=0.9×7925) | `depth=1*dz`, `mask[0]=True` | >= 判定 |
| `test_just_below` | `z[0]=7132.4` | `depth=0`, `mask=all False` | < 閾値 |
| `test_all_ablated` | 全グリッド 8000 K | `depth=n_z*dz`, `mask=all True` | 全グリッド除去 |
| `test_different_grid` | n_z=500, dz=10nm | 正しい depth | グリッドサイズ非依存性 |
| `test_stateless` | 同一 Tl で2回呼び出し | 同一結果 | 状態を持たない |
| `test_immutability` | 呼び出し前後で Tl 不変 | 入力配列が変更されない | 安全性 |

### 7.2 テスト用フィクスチャ例

```python
import numpy as np
from modules.ablation.config import AblationConfig

def make_Tl(n_z: int, hot_count: int, hot_temp: float = 8000.0, cold_temp: float = 3000.0):
    """先頭 hot_count グリッドを hot_temp、残りを cold_temp にした配列を生成"""
    Tl = np.full(n_z, cold_temp)
    Tl[:hot_count] = hot_temp
    return Tl

# 使用例
config = AblationConfig()  # デフォルト値を使用
dz = 5e-7  # 5 nm in cm
Tl = make_Tl(n_z=1000, hot_count=20)
```
