"""modules/euler_fdm/config.py — 設定パラメータ定義

euler_fdm モジュールの全設定パラメータを型安全に管理する。
YAML等の外部ファイルは使用せず、Pydanticモデルとして定義。
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict

from modules.material_properties.constants import LASER_1030NM


class GridConfig(BaseModel):
    """グリッド設定（全モジュールの唯一の真実源）。
    
    Attributes:
        n_z: グリッド数
        dz: グリッド間隔 [cm]（= 5.0e-7 = 5 nm）
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    n_z: int = 1000
    dz: float = 5.0e-7  # 5 nm を cm で表現
    
    @property
    def L_z(self) -> float:
        """計算領域長 [cm]。
        
        Returns:
            n_z × dz の値
        """
        return self.n_z * self.dz


class TimeConfig(BaseModel):
    """時間設定。
    
    Attributes:
        t_end: 終了時刻 [s]（= 500e-12 = 500 ps）
        dt_max: 最大時間刻み [s]（= 1e-15 = 1 fs）
        snapshot_interval: スナップショット記録間隔 [ステップ数]
        pulse_duration: パルス幅 FWHM [s]（t_start の計算に使用）
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    t_end: float = 500e-12
    dt_max: float = 1e-15
    snapshot_interval: int = 100
    pulse_duration: float = LASER_1030NM.pulse_duration
    
    @property
    def t_start(self) -> float:
        """シミュレーション開始時刻 [s]。

        論文の時間軸で「レーザーピーク = 1.684 ps, ne_peak ≈ 1.8 ps」と一致させるため
        -4×pulse_duration から開始する（-4tp では強度 ≈ 10⁻²⁰ で物理的に無視可能）。

        Returns:
            -4.0 * pulse_duration [s]
        """
        return -4.0 * self.pulse_duration


class InitialCondition(BaseModel):
    """初期条件。
    
    Attributes:
        Te_init: 初期電子温度 [K]
        Tl_init: 初期格子温度 [K]
        ne_init: 初期キャリア密度 [cm⁻³]
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    Te_init: float = 300.0
    Tl_init: float = 300.0
    ne_init: float = 1.0e12


class EulerFDMConfig(BaseModel):
    """最上位設定（全てを包含）。

    Attributes:
        grid: グリッド設定
        time: 時間設定
        initial: 初期条件
        fluence: 入力フルエンス [J/cm²]（必須パラメータ）
        progress_callback: 進捗通知コールバック（オプション）
            シグネチャ: (step_index: int, t: float, t_end: float, history: dict) -> None
            snapshot_interval ごとに呼び出される
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    grid: GridConfig = GridConfig()
    time: TimeConfig = TimeConfig()
    initial: InitialCondition = InitialCondition()
    fluence: float  # 必須パラメータ（デフォルト値なし）
    # Te スキーム選択（TTMConfig に委譲）
    # "cn"    : Predictor-Corrector Crank-Nicolson（デフォルト）
    # "euler" : 前進オイラー（論文再現用。CFL条件を time.dt_max で管理すること）
    te_scheme: Literal["cn", "euler"] = "cn"
    progress_callback: Optional[Any] = None
