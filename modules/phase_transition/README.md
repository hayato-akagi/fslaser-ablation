# modules/phase_transition — シリコンの相転移状態管理モジュール

## 概要

このモジュールは、シリコンの固相・液相・気相間の相転移を管理する有限状態機械（FSM）を提供する。

**設計方針:**
- Layer1/2/3 構造は持たない（相転移FSMは単一の状態機械として実装）
- 潜熱の蓄積・消費を追跡
- 相境界での温度固定を保証

## 相状態の遷移図

```
SOLID → MELTING → LIQUID → VAPORIZING → VAPOR
  ↓        ↓          ↓          ↓          ↓
T_m到達  L_m蓄積  T_b到達   L_v蓄積     自由温度
```

| 相状態 | 温度範囲 | 潜熱蓄積 | 遷移条件 |
|---|---|---|---|
| SOLID | T < T_m | - | T ≥ T_m → MELTING |
| MELTING | T = T_m | 0 → L_m | 潜熱 ≥ L_m → LIQUID |
| LIQUID | T_m < T < T_b | - | T ≥ T_b → VAPORIZING |
| VAPORIZING | T = T_b | 0 → L_v | 潜熱 ≥ L_v → VAPOR |
| VAPOR | T > T_b | - | — |

## 提供する機能

| 関数名 | 説明 |
|---|---|
| `apply_phase_transitions` | 相転移判定と潜熱処理を適用 |

## 使用例

```python
from modules.phase_transition import public as phase_trans
from modules.phase_transition.config import PhaseTransitionConfig

config = PhaseTransitionConfig()

# 相転移処理を適用
Tl_new, phase_state_new, latent_new = phase_trans.apply_phase_transitions(
    Tl=Tl,
    rhs_l=rhs_l,
    Cl=Cl,
    phase_state=phase_state,
    latent_heat_accumulated=latent_heat_accumulated,
    dt=dt,
    config=config,
)
```

## ファイル構成

```
phase_transition/
├── __init__.py       # モジュール初期化
├── config.py         # 相転移パラメータ（T_m, T_b, L_m, L_v）
├── fsm.py            # 相転移FSMの実装
├── public.py         # 外部公開API
└── README.md         # このファイル
```

## 依存関係

- `modules/__init__.py` (PhaseState enum)
- `modules.material_properties.constants` (SILICON)
- `numpy`

他のドメインモジュールへの依存は**ゼロ**。

## テスト

```bash
docker compose run --rm sim pytest tests/test_phase_transition/ -v
```

## 実装の詳細

### 潜熱の蓄積

相転移境界（T_m, T_b）に達すると、温度を固定し、入力エネルギーを潜熱として蓄積する：

```python
energy_input = dt * rhs_l  # [J/cm³]
latent_heat_accumulated += energy_input
```

### 相転移の完了

潜熱が閾値に達すると、次の相に遷移し、余剰エネルギーで温度を上昇させる：

```python
# MELTING → LIQUID
if latent_heat_accumulated >= L_m:
    remaining_energy = latent_heat_accumulated - L_m
    Tl_new = T_m + remaining_energy / Cl  # Cl は呼び出し元から渡される
    phase_state = LIQUID
    latent_heat_accumulated = 0

# VAPORIZING → VAPOR（Cl は沸点での固相式から内部計算）
if latent_heat_accumulated >= L_v:
    Cl_at_boiling = 1.978 + 3.54e-4 * T_b - 3.68 / T_b**2
    remaining_energy = latent_heat_accumulated - L_v
    Tl_new = T_b + remaining_energy / Cl_at_boiling
    phase_state = VAPOR
    latent_heat_accumulated = 0
```

### 数値安定性

- NaN/Inf の自動除去
- 温度の範囲制限（T_room 〜 1e6 K）
- ゼロ除算回避
