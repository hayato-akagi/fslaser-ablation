# fslaser-ablation-sim

フェムト秒レーザー照射によるシリコンアブレーションの1次元数値シミュレーション。  
論文 **jmmp-07-00068** の Figure 4〜9 を再現することを目的としている。

---

## 物理モデルの概要

| モデル | 説明 |
|---|---|
| 二温度モデル (TTM) | 電子温度 $T_e$ と格子温度 $T_l$ を別々に時間発展させる |
| Drude 動的光学モデル | キャリア密度に依存した複素誘電率・反射率・FCA吸収を計算 |
| キャリア密度方程式 | SPA / TPA / Auger / 衝突電離 / 両極性拡散の5項 |
| 相転移 FSM | SOLID → MELTING → LIQUID → VAPORIZING → VAPOR の潜熱管理 |
| Phase Explosion 判定 | $T_l \geq 0.9 \times T_{cr} = 7132.5$ K でアブレーション深さを算出 |

数値解法: 前進オイラー法 + 1D 有限差分法 (中心差分、CFL 適応刻み)

---

## ディレクトリ構成

```
fslaser-ablation-sim/
├── modules/                # ドメインモジュール群
│   ├── material_properties/  # シリコン物性・物理定数・単位変換
│   ├── optics/               # Drudeモデル + レーザー伝播
│   ├── carrier/              # キャリア密度方程式
│   ├── ttm/                  # 二温度モデル
│   ├── phase_transition/     # 相転移 FSM
│   ├── ablation/             # Phase Explosion アブレーション判定
│   └── euler_fdm/            # 連成ソルバー（タイムループ統括）
├── views/                  # 可視化・データ入出力
│   ├── io.py               # SimulationResult の保存・読み込み
│   └── plotting.py         # matplotlib グラフ生成（14種）
├── tests/                  # ユニット・統合テスト
├── results/                # シミュレーション出力（自動生成）
├── run.py                  # シミュレーション実行スクリプト
├── reproduce_figure4.py    # 論文 Figure 4 再現スクリプト
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 実行環境

Docker を使用する。ホストへのインストール不要。

```bash
# イメージをビルド（初回のみ）
docker compose build
```

---

## シミュレーションの実行

### 基本実行

```bash
docker compose run --rm sim python run.py
```

`run.py` はデフォルトで以下のパラメータを使用する:

| パラメータ | デフォルト値 | 説明 |
|---|---|---|
| `fluence` | `1.5` J/cm² | レーザーフルエンス（論文条件） |
| `n_z` | `1000` | グリッド数 |
| `dz` | `5.0e-7` cm (= 5 nm) | グリッド間隔 |
| `t_end` | `20e-12` s (= 20 ps) | シミュレーション終了時刻 |
| `dt_max` | `1e-15` s (= 1 fs) | 最大時間刻み |
| `snapshot_interval` | `100` | スナップショット記録間隔 [ステップ] |

### パラメータを変更して実行

`run.py` の `main()` 内の `EulerFDMConfig` を直接編集する:

```python
config = EulerFDMConfig(
    fluence=3.06,           # フルエンスを変更 [J/cm²]
    grid=GridConfig(
        n_z=1000,
        dz=5.0e-7,          # 5 nm
    ),
    time=TimeConfig(
        t_end=500e-12,      # 500 ps まで計算
        dt_max=1e-15,
        snapshot_interval=100,
    ),
)
```

編集後に再度 `docker compose run --rm sim python run.py` を実行する。

### 実行結果

```
=== Simulation Config ===
  Fluence     : 1.5 J/cm²
  Grid        : 1000 points, dz = 5.0 nm
  t_start     : 0.000 ps
  t_end       : 20.0 ps
  ...

=== Result ===
  Total steps       : ...
  Ablation depth    : 123.4 nm
  Final Te(surface) : ...
  ...

Saved to: results/20260528_120000_F1.50/
Plots in: results/20260528_120000_F1.50/plots/
```

結果は `results/{YYYYMMDD_HHMMSS}_F{fluence:.2f}/` に保存される:

```
results/20260528_120000_F1.50/
├── metadata.json       # 設定・メタ情報
├── arrays.npz          # NumPy 配列データ
└── plots/              # 生成グラフ（8枚）
    ├── temperature_history.png     # Te(0,t), Tl(0,t) ← 論文 Fig.4 相当
    ├── carrier_density_history.png # ne(0,t)           ← 論文 Fig.7 相当
    ├── reflectivity_history.png    # R(t)              ← 論文 Fig.5a 相当
    ├── alpha_fca_history.png       # α_FCA(t)          ← 論文 Fig.5b 相当
    ├── Te_ne_dual.png              # Te + ne デュアル軸 ← 論文 Fig.6 相当
    ├── auger_ne_history.png        # γne³ + ne         ← 論文 Fig.8 相当
    ├── spatial_profiles.png        # 最終空間プロファイル
    └── summary.png                 # 2×2 概要図
```

---

## 論文 Figure の再現

### Figure 4（複数フルエンスの温度時間発展）

```bash
docker compose run --rm sim python reproduce_figure4.py
```

F = 0.25, 0.8, 1.5 J/cm² の3条件で計算し、論文 Fig.4 相当のグラフを出力する。  
計算時間の目安: 各フルエンスで数分〜数十分。

---

## テスト

### 全テスト実行

```bash
docker compose run --rm sim
```

`docker-compose.yml` のデフォルトコマンドは `pytest tests/ -v`。

### 特定モジュールのテストのみ実行

```bash
# ablation モジュール
docker compose run --rm sim pytest tests/test_ablation/ -v

# carrier モジュール
docker compose run --rm sim pytest tests/test_carrier/ -v

# TTM モジュール
docker compose run --rm sim pytest tests/test_ttm/ -v

# optics モジュール
docker compose run --rm sim pytest tests/test_optics/ -v

# euler_fdm モジュール（統合テスト含む）
docker compose run --rm sim pytest tests/test_euler_fdm/ -v

# views モジュール（I/O + プロット）
docker compose run --rm sim pytest tests/test_views/ -v
```

### カバレッジ計測

```bash
docker compose run --rm sim pytest tests/ --cov=modules --cov=views --cov-report=term-missing
```

---

## モジュール構成と依存関係

```
euler_fdm  ←── optics
           ←── carrier  ←── material_properties
           ←── ttm      ←── material_properties
           │            ←── phase_transition ←── material_properties
           └── ablation ←── material_properties
```

各ドメインは `public.py` のみを外部インターフェースとして公開する。  
詳細は各モジュールの `README.md` および `modules/CONVENTIONS.md` を参照。

---

## 技術スタック

| 項目 | 内容 |
|---|---|
| 言語 | Python 3.12 |
| 数値計算 | NumPy 1.26 |
| データモデル | Pydantic 2.x |
| 可視化 | matplotlib 3.9 |
| テスト | pytest 8.x |
| コンテナ | Docker (python:3.12-slim) |
