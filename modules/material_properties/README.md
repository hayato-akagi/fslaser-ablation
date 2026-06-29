# modules/material_properties — シリコンの物性パラメータ計算モジュール

## 概要

このモジュールは、シリコンの温度依存・相依存の物性パラメータを計算する純粋関数群を提供する。

**設計方針:**
- Layer1/2/3 構造は持たない（物性計算は純粋な数学関数のため）
- 全ての関数はステートレス（副作用なし）
- 固相・液相の分岐ロジックを一元化

## 提供する物性パラメータ

| パラメータ | 関数名 | 単位 | 固液依存 |
|---|---|---|---|
| バンドギャップ | `compute_bandgap` | eV | ✓ |
| SPA吸収係数 | `compute_alpha_spa` | cm⁻¹ | — |
| 衝突電離係数 | `compute_impact_ionization_rate` | s⁻¹ | — |
| 拡散係数 | `compute_diffusion_coefficient` | cm²/s | — |
| 電子衝突時間 | `compute_tau_e` | s | ✓ |
| 電子熱容量 | `compute_thermal_capacity_electron` | J/(cm³·K) | ✓ |
| 格子熱容量 | `compute_thermal_capacity_lattice` | J/(cm³·K) | ✓ |
| 電子熱伝導率 | `compute_thermal_conductivity_electron` | W/(cm·K) | ✓ |
| 格子熱伝導率 | `compute_thermal_conductivity_lattice` | W/(cm·K) | ✓ |
| 電子格子結合 | `compute_electron_lattice_coupling` | W/(cm³·K) | — |
| Eg偏微分(∂Eg/∂ne) | `compute_bandgap_derivative` | eV·cm³ | ✓ |

## 使用例

```python
from modules.material_properties import public as mat_props

# バンドギャップエネルギーを計算
Eg = mat_props.compute_bandgap(Tl, ne, phase_state)

# SPA吸収係数を計算
alpha_spa = mat_props.compute_alpha_spa(Tl)

# 熱容量を計算（固液自動切り替え）
Ce = mat_props.compute_thermal_capacity_electron(Te, ne, phase_state)
Cl = mat_props.compute_thermal_capacity_lattice(Tl, phase_state)
```

## ファイル構成

```
material_properties/
├── __init__.py             # モジュール初期化（public.py の関数を再エクスポート）
├── constants.py            # 全シミュレーション共通の物理定数・物性定数
│                           #   (PHYSICAL, SILICON, LASER_1030NM シングルトン)
├── silicon.py              # 全物性計算の実装（純粋関数群）
├── unit_conversions.py     # 単位変換ユーティリティ（cm↔nm, J↔erg 等）
├── public.py               # 外部公開API（silicon.py の関数を再エクスポート）
└── README.md               # このファイル
```

## 依存関係

- `modules/__init__.py` (PhaseState enum)
- `numpy`

他のドメインモジュールへの依存は**ゼロ**。

## 定数ファイル（constants.py）

全シミュレーション共通の物理定数・物性定数は `constants.py` に集約されている：

```python
from modules.material_properties.constants import PHYSICAL, SILICON, LASER_1030NM

PHYSICAL.k_B          # 1.381e-23 J/K
SILICON.T_m           # 1687.0 K（融点）
LASER_1030NM.photon_energy_eV  # プランク定数・波長から計算
```

## テスト

```bash
docker compose run --rm sim pytest tests/test_material_properties/ -v
```

## 固液分岐の実装

全ての固液依存パラメータは、内部で `_is_solid_phase(phase_state)` を使用して判定する。

```python
solid_mask = (phase_state == PhaseState.SOLID) | (phase_state == PhaseState.MELTING)
```

この判定ロジックは `silicon.py` 内で一元管理される。

## ∂Eg/∂ne の単位

`compute_bandgap_derivative` が返す値の単位は **eV·cm³**（`ne` が cm⁻³ なので、`ne × ∂Eg/∂ne` が eV となる）。
