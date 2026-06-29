# views/ — 可視化・グラフ出力

> **前提**: 本ディレクトリは CLAUDE.md の規約に従い、`modules/` の外側に配置された  
> **最上位プレゼンテーション層**である。ビジネス/物理ロジックは一切含まない。

---

## 1. 責務

- `results/` に保存されたシミュレーション結果を読み込み、グラフを生成・保存する
- `euler_fdm.public.SimulationResult` の構造に依存するが、物理計算は行わない
- 生成されたグラフは `results/<run_id>/plots/` に保存する

---

## 2. results/ ディレクトリ構造

各シミュレーション実行ごとにサブディレクトリが作成される:

```
results/
├── .gitkeep
├── 20260528_120000_F1.50/          # {日時}_{フルエンス}
│   ├── metadata.json               # 設定・メタ情報
│   ├── arrays.npz                  # numpy 配列データ
│   └── plots/                      # 生成されたグラフ
│       ├── temperature_history.png
│       ├── carrier_density_history.png
│       ├── reflectivity_history.png
│       ├── alpha_fca_history.png
│       ├── Te_ne_dual.png
│       ├── auger_ne_history.png
│       ├── spatial_profiles.png
│       └── summary.png
├── 20260528_120100_F2.00/
│   ├── ...
└── fluence_scan/                   # 複数フルエンスの比較
    ├── Tl_surface_compare.png
    ├── Te_surface_compare.png
    ├── reflectivity_compare.png
    ├── alpha_fca_compare.png
    ├── auger_ne_compare.png
    └── ablation_depth_vs_fluence.png
```

### 2.1 metadata.json の構造

```json
{
  "created_at": "2026-05-28T12:00:00",
  "fluence_J_cm2": 1.50,
  "grid": {
    "n_z": 1000,
    "dz_cm": 5e-7
  },
  "time": {
    "t_start_s": 0.0,
    "t_end_s": 5e-10,
    "dt_max_s": 1e-15,
    "total_steps": 42000
  },
  "ablation_depth_nm": 123.4
}
```

### 2.2 arrays.npz の内容

`np.savez_compressed()` で保存。キー名は `SimulationResult` のフィールド名と一致させる:

| キー | shape | 単位 | 説明 |
|---|---|---|---|
| `Te_final` | `(n_z,)` | K | 最終電子温度 |
| `Tl_final` | `(n_z,)` | K | 最終格子温度 |
| `ne_final` | `(n_z,)` | cm⁻³ | 最終キャリア密度 |
| `ablated_mask` | `(n_z,)` | bool | 累積アブレーションマスク |
| `time_points` | `(n_snapshots,)` | s | 記録時刻 |
| `Te_surface_history` | `(n_snapshots,)` | K | z=0 の Te 履歴 |
| `Tl_surface_history` | `(n_snapshots,)` | K | z=0 の Tl 履歴 |
| `ne_surface_history` | `(n_snapshots,)` | cm⁻³ | z=0 の ne 履歴 |
| `reflectivity_history` | `(n_snapshots,)` | — | 表面反射率履歴 |
| `alpha_fca_surface_history` | `(n_snapshots,)` | cm⁻¹ | z=0 の α_FCA 履歴 |
| `auger_term_surface_history` | `(n_snapshots,)` | cm⁻³/s | z=0 の γn_e³ 履歴 |
| `ablation_depth_history` | `(n_snapshots,)` | nm | アブレーション深さ履歴 |

---

## 3. 生成するグラフ一覧

グラフは **論文 (jmmp-07-00068) の全シミュレーション図** を再現できるよう分類する。
実験画像（Fig.1〜3）はシミュレーション対象外のため含まない。

### 3.1 単一フルエンス実行（per-run グラフ）

各 run ディレクトリの `plots/` に生成される。

| # | ファイル名 | 論文対応 | 内容 | X軸 | Y軸 |
|---|---|---|---|---|---|
| 1 | `temperature_history.png` | Fig.4 (単一F分) | 表面温度 Te(0,t), Tl(0,t) の時間変化 | 時間 [ps] | 温度 [K] |
| 2 | `carrier_density_history.png` | Fig.7 (単一F分) | 表面キャリア密度 ne(0,t) の時間変化 | 時間 [ps] | $n_e$ [cm⁻³] (対数) |
| 3 | `reflectivity_history.png` | Fig.5a (単一F分) | 表面反射率 R(t) の時間変化 | 時間 [ps] | R (0~1) |
| 4 | `alpha_fca_history.png` | Fig.5b (単一F分) | 表面 FCA 吸収係数 α_FCA(t) の時間変化 | 時間 [ps] | α_FCA [cm⁻¹] |
| 5 | `Te_ne_dual.png` | Fig.6, 7 | Te(0,t) と ne(0,t) をデュアルY軸で重ね描き | 時間 [ps] | 左: Te [K] / 右: $n_e$ [cm⁻³] |
| 6 | `auger_ne_history.png` | Fig.8 (単一F分) | γn_e³(t) と ne(0,t) のデュアルY軸 | 時間 [ps] | 左: γn_e³ [cm⁻³/s] / 右: $n_e$ [cm⁻³] |
| 7 | `spatial_profiles.png` | — | 最終状態の空間分布 Te(z), Tl(z), ne(z) | 深さ [nm] | 各物理量 |
| 8 | `summary.png` | — | #1, #2, #3, #7 を 2×2 サブプロットにまとめた概要図 | — | — |

### 3.2 フルエンス比較（multi-run グラフ）

複数フルエンスの結果を重ね描きする。保存先: `results/fluence_scan/`

| # | ファイル名 | 論文対応 | 内容 | X軸 | Y軸 |
|---|---|---|---|---|---|
| 9 | `Tl_surface_compare.png` | **Fig.4a** | 複数Fでの Tl(0,t) 重ね描き | 時間 [ps] | Tl [K] |
| 10 | `Te_surface_compare.png` | **Fig.4b** | 複数Fでの Te(0,t) 重ね描き | 時間 [ps] | Te [K] |
| 11 | `reflectivity_compare.png` | **Fig.5a** | 複数Fでの R(t) 重ね描き | 時間 [ps] | R (0~1) |
| 12 | `alpha_fca_compare.png` | **Fig.5b** | 複数Fでの α_FCA(t) 重ね描き | 時間 [ps] | α_FCA [cm⁻¹] |
| 13 | `auger_ne_compare.png` | **Fig.8** | 複数Fでの γn_e³(t) と ne(t) 重ね描き | 時間 [ps] | γn_e³ / $n_e$ |
| 14 | `ablation_depth_vs_fluence.png` | **Fig.9** | アブレーション深さ vs フルエンス | フルエンス [J/cm²] | 深さ [nm] |

- Fig.9 グラフは論文の実験データ点および参考文献 [24] の赤線も重ね描画可能

### 3.3 論文図との対応まとめ

| 論文 | 内容 | 再現方法 |
|---|---|---|
| Fig.1 | 実験装置の概略図 | ❌ 対象外（実験画像） |
| Fig.2 | AFM画像 | ❌ 対象外（実験画像） |
| Fig.3 | SEM画像 | ❌ 対象外（実験画像） |
| Fig.4a | Tl(0,t)（複数F） | `Tl_surface_compare.png` (#9) |
| Fig.4b | Te(0,t)（複数F） | `Te_surface_compare.png` (#10) |
| Fig.5a | R(t)（複数F） | `reflectivity_compare.png` (#11) |
| Fig.5b | α_FCA(t)（複数F） | `alpha_fca_compare.png` (#12) |
| Fig.6 | Te + ne（F=0.25, デュアルY軸） | `Te_ne_dual.png` (#5) で F=0.25 実行 |
| Fig.7 | Te + Tl + ne（F=3.06, 拡大図付き） | `Te_ne_dual.png` (#5) + `temperature_history.png` (#1) で F=3.06 実行 |
| Fig.8 | γn_e³ + ne（複数F） | `auger_ne_compare.png` (#13) |
| Fig.9 | Ablation depth vs fluence | `ablation_depth_vs_fluence.png` (#14) |

### 3.4 グラフ共通仕様

- **フォーマット**: PNG (dpi=150)
- **フォントサイズ**: タイトル 14pt, 軸ラベル 12pt, 凡例 10pt
- **カラー**: matplotlib デフォルトカラーサイクル、Te=赤, Tl=青 を基本
- **グリッド表示**: ON
- **時間軸の単位変換**: 内部 [s] → 表示 [ps]（$\times 10^{12}$）
- **深さ軸の単位変換**: 内部 [cm] → 表示 [nm]（$\times 10^{7}$）
- **凡例ラベル**: 複数F比較時は `F = {fluence:.2f} J/cm²` 形式

---

## 4. public API

### 4.1 データ保存（euler_fdm から呼ぶ）

```python
# views/io.py
def save_result(
    result: "SimulationResult",
    config: "EulerFDMConfig",
    output_dir: Path = Path("results"),
) -> Path:
    """SimulationResult を results/ に保存し、保存先パスを返す。

    1. run_id ディレクトリを作成 ({YYYYMMDD_HHMMSS}_F{fluence:.2f})
    2. metadata.json を書き出し
    3. arrays.npz を書き出し
    
    Returns:
        作成された run ディレクトリの Path
    """
```

### 4.2 データ読み込み

```python
# views/io.py
def load_result(run_dir: Path) -> tuple["SimulationResult", dict]:
    """保存済み結果を読み込む。

    Returns:
        (SimulationResult, metadata_dict) のタプル
    """
```

### 4.3 グラフ生成（単一実行）

```python
# views/plotting.py
def plot_single_run(run_dir: Path) -> None:
    """1回のシミュレーション結果からグラフ8枚を生成・保存。

    生成グラフ (#1~#8):
      temperature_history, carrier_density_history,
      reflectivity_history, alpha_fca_history,
      Te_ne_dual, auger_ne_history,
      spatial_profiles, summary

    保存先: run_dir/plots/
    """
```

### 4.4 グラフ生成（フルエンス比較）

```python
# views/plotting.py
def plot_fluence_comparison(
    run_dirs: list[Path],
    output_dir: Path = Path("results/fluence_scan"),
    experimental_data: dict[float, float] | None = None,
) -> None:
    """複数 run の結果を重ね描きした比較グラフ6枚を生成・保存。

    生成グラフ (#9~#14):
      Tl_surface_compare, Te_surface_compare,
      reflectivity_compare, alpha_fca_compare,
      auger_ne_compare, ablation_depth_vs_fluence

    Args:
        run_dirs: 各フルエンスの run ディレクトリリスト
        output_dir: 比較グラフの保存先
        experimental_data: {fluence: depth_nm} の辞書（論文実験値。Noneなら省略）
    """
```

---

## 5. ファイル構成

```
views/
├── __init__.py
├── README.md          # 本仕様書
├── io.py              # save_result(), load_result()
└── plotting.py        # plot_single_run(), plot_fluence_comparison()
```

- `io.py`: データの永続化（保存・読み込み）。numpy / json 操作のみ
- `plotting.py`: matplotlib によるグラフ生成。`io.py` の `load_result()` を使う

---

## 6. 使い方（想定ワークフロー）

```python
from modules.euler_fdm.public import run_simulation
from modules.euler_fdm.config import EulerFDMConfig
from views.io import save_result
from views.plotting import plot_single_run, plot_fluence_comparison

# 1. シミュレーション実行
config = EulerFDMConfig(fluence=1.5)
result = run_simulation(config)

# 2. 結果保存
run_dir = save_result(result, config)

# 3. グラフ生成（単一実行）
plot_single_run(run_dir)

# 4. フルエンススキャン比較（複数実行後）
all_dirs = [Path("results/run1"), Path("results/run2"), ...]
plot_fluence_comparison(all_dirs)
```

---

## 7. テスト方針

```bash
docker compose run --rm sim pytest tests/test_views/ -v
```

| テスト名 | 内容 | 検証方法 |
|---|---|---|
| `test_save_load_roundtrip` | 保存→読込でデータが一致 | `np.allclose` |
| `test_metadata_fields` | metadata.json に必須フィールドが存在 | JSON キー検証 |
| `test_plot_single_run` | グラフ8枚が生成される | ファイル存在確認 |
| `test_plot_fluence_comparison` | 比較グラフ6枚が生成される | ファイル存在確認 |
| `test_run_id_format` | ディレクトリ名のフォーマット | 正規表現マッチ |
