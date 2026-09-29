"""
数据加载与统一 —— 把 GPMF 里散落的传感器流拼成一份可分析的遥测对象
================================================================

对应关系：`gpmf.py` 相当于 open()，`telemetry.py` 相当于 read_csv() + 数据清洗，
之后所有模块拿到的都是一个 `Telemetry` 对象（可以理解成一个带元数据的 DataFrame）。

处理流程：
    1. 读流            MP4 → {GPS5, ACCL, GYRO, ...}
    2. 单位换算与校验   GoPro 存的是整数，要除以 SCAL 才是物理量；
                       万一 SCAL 方向搞反了，用数值范围自动识别
    3. 投影            经纬度 → 本地平面米制坐标
    4. 清洗            去 GPS 尖峰、剔除无效定位点
    5. G 值提取        用 GRAV 扣重力 + 投影到速度坐标系（见 imu.py）
    6. 交叉验证        用 GPS 独立推算 G，与 IMU 的对比
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import geo, gpmf, imu

# ==========================================================================
# 各传感器的分量含义（顺序与 GoPro 定义一致）
# ==========================================================================
_GPS_LAYOUTS: dict[str, tuple[list[str], list[str]]] = {
    # 流名: (分量名, 单位说明)
    "GPS5": (["lat", "lon", "alt", "speed2d", "speed3d"], ["deg", "deg", "m", "m/s", "m/s"]),
    "GPS9": (
        ["lat", "lon", "alt", "speed2d", "speed3d", "days", "secs", "dop", "fix"],
        ["deg", "deg", "m", "m/s", "m/s", "day", "ms", "-", "-"],
    ),
}

_ACCEL_NAMES = ["x", "y", "z"]
_GYRO_NAMES = ["x", "y", "z"]


def _pick_scaling(st: gpmf.Stream, validator) -> np.ndarray:
    """
    把原始整数换算成物理量，并检验结果是否合理。

    为什么要检验：不同固件版本里 SCAL 的含义（乘还是除、单值还是数对）出现过不一致。
    与其相信文档，不如直接看算出来的经纬度是不是落在合法范围内 —— 这是最可靠的判据。
    """
    cand = st.physical()
    if validator(cand):
        return cand

    # 反向试试（把 SCAL 当乘法系数）
    if st.scale is not None and st.scale.size:
        k = st.values.shape[1]
        inv = st.values.astype(np.float64) * np.resize(
            np.asarray(st.scale, dtype=np.float64), k
        )
        if validator(inv):
            return inv

    # 都不合理就返回正解，让上层的合理性检查去告警
    return cand


def _gps_validator(v: np.ndarray) -> bool:
    """检验解码结果是否像一份真实 GPS 数据。"""
    if v.size == 0 or v.ndim != 2:
        return False
    lat, lon = v[:, 0], v[:, 1]
    if not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)):
        return False
    if np.any(np.abs(lat) > 90.0) or np.any(np.abs(lon) > 180.0):
        return False
    # 还要能看出实际位移，否则可能是全 0 的流
    return float(np.ptp(lat)) > 1e-6 or float(np.ptp(lon)) > 1e-6


def _accel_validator(v: np.ndarray) -> bool:
    if v.size == 0:
        return False
    m = float(np.median(np.linalg.norm(v.reshape(-1, v.shape[-1]), axis=1)))
    return 3.0 < m < 40.0


# ==========================================================================
@dataclass
class Telemetry:
    """一份完整的、已标定的遥测数据。"""

    source: str
    """来源文件名（调试用）。"""

    # ---- GPS（时间基准） ----
    t: np.ndarray
    """GPS 时间轴，秒，从录制开始算。"""

    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray
    speed: np.ndarray
    """地面速度 m/s（GoPro 用多普勒测速，比"位置差分"平滑得多）。"""

    speed3d: np.ndarray
    x: np.ndarray
    """局部平面坐标（米，东为正）。"""

    y: np.ndarray
    """局部平面坐标（米，北为正）。"""

    dist: np.ndarray
    """累计行驶距离，米（基于平滑后的轨迹）。"""

    heading: np.ndarray
    """航向角，弧度，已解卷绕，逆时针（左转）为正。"""

    curvature: np.ndarray | None = None
    """轨迹曲率 1/米（左转为正）。由强平滑后的轨迹算出，用来判定转向方向。"""

    x_raw: np.ndarray | None = None
    y_raw: np.ndarray | None = None
    """未经平滑的原始平面坐标，保留给需要看 GPS 噪声的场景。"""

    fix: np.ndarray | None = None
    """定位质量标志（仅 GPS9 有）。"""

    dop: np.ndarray | None = None
    """精度因子，越小越好（仅 GPS9 有）。"""

    # ---- IMU ----
    t_imu: np.ndarray | None = None
    acc: np.ndarray | None = None
    """(N,3) 相机坐标系加速度 m/s²。"""

    gyro: np.ndarray | None = None
    grav: np.ndarray | None = None

    # ---- G 值提取结果 ----
    gfield: imu.GField | None = None
    """重力矢量（逐时刻）+ 轴变换 + 质量指标，见 `imu.GField`。"""

    a_long: np.ndarray | None = None
    """纵向 G，200Hz。正值 = 加速，负值 = 刹车。"""

    a_lat: np.ndarray | None = None
    """横向 G，200Hz。正值 = 左转。"""

    a_long_gps: np.ndarray | None = None
    a_lat_gps: np.ndarray | None = None
    """仅由 GPS 推算的 G（交叉验证用）。"""

    lat0: float = 0.0
    lon0: float = 0.0

    streams: dict[str, gpmf.Stream] = field(default_factory=dict)
    """所有原始流，供高级用户或调试使用。"""

    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if self.t.size else 0.0

    @property
    def gps_rate(self) -> float:
        if self.t.size < 2:
            return 0.0
        span = float(self.t[-1] - self.t[0])
        return (self.t.size - 1) / span if span > 0 else 0.0

    def imu_g(
        self, t_axis: np.ndarray, *, smooth_seconds: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """把 200Hz 的 G 值插值到任意时间轴（通常是 GPS 时间轴或距离网格）。

        smooth_seconds > 0 时**先按时间平滑再插值**。算"峰值"这类极值指标
        必须这么做，否则拿到的是短尖峰而不是驾驶动作 —— 理由见
        `imu.PEAK_SMOOTH_S`。
        """
        if self.a_long is None or self.t_imu is None:
            raise RuntimeError("G 值提取未完成，没有可用的 G 值")
        a_long, a_lat = self.a_long, self.a_lat
        if smooth_seconds > 0.0:
            a_long = imu.peak_g(a_long, self.t_imu, smooth_seconds)
            a_lat = imu.peak_g(a_lat, self.t_imu, smooth_seconds)
        return (
            geo.interp_to(self.t_imu, a_long, t_axis),
            geo.interp_to(self.t_imu, a_lat, t_axis),
        )

    def speed_stats(self) -> dict[str, float]:
        return {
            "max": float(np.max(self.speed)),
            "mean": float(np.mean(self.speed)),
            "p95": float(np.percentile(self.speed, 95)),
        }

    def summary(self) -> str:
        lines = [
            f"来源           : {self.source}",
            f"时长           : {self.duration:.1f} s",
            f"GPS 采样率     : {self.gps_rate:.1f} Hz  ({self.t.size} 个定位点)",
        ]
        if self.t_imu is not None and self.t_imu.size > 1:
            imu_rate = (self.t_imu.size - 1) / (self.t_imu[-1] - self.t_imu[0])
            lines.append(f"IMU 采样率     : {imu_rate:.1f} Hz  ({self.t_imu.size} 个采样点)")
        lines.append(f"行驶里程       : {float(self.dist[-1]):.0f} m")
        s = self.speed_stats()
        lines.append(f"速度           : 最高 {s['max'] * 3.6:.1f} km/h / 平均 {s['mean'] * 3.6:.1f} km/h")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Telemetry {self.source} {self.duration:.1f}s {self.t.size}pts>"


# ==========================================================================
def _decode_gps(streams: dict[str, gpmf.Stream], warnings_out: list[str]) -> tuple | None:
    """从 GPS5 / GPS9 里解出经纬度与速度，优先用精度更高的 GPS9。"""
    for name in ("GPS9", "GPS5"):
        st = streams.get(name)
        if st is None or st.times.size < 2:
            continue
        names, _units = _GPS_LAYOUTS[name]
        if st.values.shape[1] != len(names):
            warnings_out.append(f"{name} 的分量个数是 {st.values.shape[1]}，与预期的 {len(names)} 不符，已跳过。")
            continue

        phys = _pick_scaling(st, _gps_validator)
        if not _gps_validator(phys):
            warnings_out.append(
                f"{name} 解码后的经纬度超出合法范围，可能是该视频的 GPS 数据异常（常见于",
                )
            warnings_out[-1] += "停车时定位漂移或未搜到星），已跳过该流。"
            continue

        idx = {n: i for i, n in enumerate(names)}
        lat = phys[:, idx["lat"]]
        lon = phys[:, idx["lon"]]
        alt = phys[:, idx["alt"]]
        spd2 = phys[:, idx["speed2d"]]
        spd3 = phys[:, idx["speed3d"]]
        fix = phys[:, idx["fix"]] if "fix" in idx else None
        dop = phys[:, idx["dop"]] if "dop" in idx else None
        return name, st.times, lat, lon, alt, spd2, spd3, fix, dop
    return None


def _select_imu(streams: dict[str, gpmf.Stream], warnings_out: list[str]) -> tuple[np.ndarray, np.ndarray] | None:
    """取出加速度计流并换算成 m/s²。"""
    st = streams.get("ACCL")
    if st is None or st.times.size < 10:
        warnings_out.append(
            "没有找到加速度计（ACCL）数据，将只能用 GPS 推算 G 值 —— "
            "曲线会明显更粗糙，且无法捕捉瞬时峰值。"
        )
        return None
    phys = _pick_scaling(st, _accel_validator)
    if not _accel_validator(phys):
        warnings_out.append("加速度计解码结果量级异常，G 值可能不可信。")
    return st.times, phys


def _select_gyro(streams: dict[str, gpmf.Stream]) -> tuple[np.ndarray, np.ndarray] | None:
    st = streams.get("GYRO")
    if st is None or st.times.size < 10:
        return None
    units = (st.units or "").lower()
    phys = _pick_scaling(st, lambda v: 0.0 < float(np.percentile(np.abs(v), 50)) < 5000.0)
    # 统一成 rad/s
    if "deg" in units:
        phys = np.radians(phys)
    elif "rad" not in units:
        # 没有单位信息时按数值量级猜：静止时 |ω| 很小，2000 deg/s ≈ 35 rad/s
        if float(np.percentile(np.abs(phys), 99)) > 60.0:
            phys = np.radians(phys)
    return st.times, phys


def derive_track(
    t: np.ndarray, lat: np.ndarray, lon: np.ndarray, speed: np.ndarray | None = None
) -> dict[str, np.ndarray | float]:
    """
    经纬度 → 可直接做几何运算的平面轨迹。

    这里做了三件事，每一件都不是可选项：

    1. **投影到本地平面米制坐标**。度数不能直接求距离 —— 1° 纬度 ≈ 111 km，
       但 1° 经度在纬度 40° 处只有 ≈ 85 km。

    2. **轻平滑**。民用 GPS 有 1~3 m 定位噪声，需要压一压；但**不能过度**：
       卡丁车弯道半径只有十几米，0.9 秒的平滑窗口在 15 m/s 下相当于 ±7 米，
       会把弯道直接抹圆 —— 实测赛道形状失真、官方图上 16 个弯只识出 5 个。
       0.35 秒（约 ±3 米）既能压噪声又能保住弯形。

    3. **里程用速度积分而不是位置差分**。位置差分会把定位噪声当成位移累加，
       实测能让 1 km 的赛道算出 1.8 km（真实卡丁车练习中里程虚高近一倍）。
       GoPro 的多普勒速度本身就平缓且准，用它积分可靠得多。
    """
    x_raw, y_raw, lat0, lon0 = geo.to_local_xy(lat, lon)
    x = geo.savgol(t, x_raw, 0.35, poly=2)
    y = geo.savgol(t, y_raw, 0.35, poly=2)

    if speed is not None:
        dist = geo.cumulative_distance_from_speed(t, speed)
        # 曲率用 ω/v 反推。直接对位置二次求导算曲率在 10Hz 下会被噪声吃掉：
        # 实测曲率半径能算出 0 米、角速度 187 rad/s 这种无意义的值。
        omega = geo.robust_yaw_rate(t, x, y)
        v = np.asarray(speed, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            kappa = np.where(v > 3.0, omega / np.maximum(v, 1e-6), 0.0)
    else:
        dist = geo.cumulative_distance(x, y)
        kappa = geo.curvature(x, y)

    return {
        "x": x,
        "y": y,
        "x_raw": x_raw,
        "y_raw": y_raw,
        "lat0": lat0,
        "lon0": lon0,
        "dist": dist,
        "heading": geo.heading(x, y),
        "curvature": kappa,
    }


def attach_imu(
    tel: Telemetry,
    t_imu: np.ndarray | None,
    acc: np.ndarray | None,
    grav: np.ndarray | None = None,
    t_grav: np.ndarray | None = None,
) -> Telemetry:
    """
    把 IMU 数据挂到 Telemetry 上：扣重力 → 生成纵向 / 横向 G 值。

    只有两步：用 `GRAV` 流**逐时刻**扣掉重力，再投影到速度坐标系。
    没有"标定安装角"这一步 —— GRAV 是相机自己融合的重力方向，会跟着相机转，
    所以相机中途被碰、支架有弹性、温漂都能自动跟上。

    没有加速度计（或数据不可用）时自动退回"只用 GPS 求导"的方案，
    此时 G 值分辨率会掉到 GPS 的采样率，但至少不会整个流程跑不下去。
    """
    warns = tel.warnings

    if acc is not None and t_imu is not None and t_imu.size >= 10:
        gl, gt = imu.gps_derived_g(tel.t, tel.speed, tel.heading, kappa=tel.curvature)
        has_ref = float(np.std(gl)) > 1e-3 and float(np.std(gt)) > 1e-3

        if not has_ref:
            # 没有 GPS 参考量就既没有重力方向判据、也没有投影方向，无从下手。
            gl, gt = imu.gps_derived_g(tel.t, tel.speed, tel.heading, kappa=tel.curvature)
            tel.a_long_gps = gl
            tel.a_lat_gps = gt
            tel.t_imu = tel.t
            tel.a_long = gl
            tel.a_lat = gt
            warns.append(
                "没有可用的 GPS 参考量（速度或航向没有变化），无法定位加速度方向，"
                "已退回仅用 GPS 求导的 G 值 —— 曲线会明显更粗糙。"
            )
            return tel

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # GPS 推算的平面加速度模长 sqrt((dv/dt)² + (v·ω)²) —— 与坐标系无关，
            # 用来识别 GRAV 的轴约定、并作为扣重力是否干净的判据。
            # 注意乘 G0 换成 m/s²：gravity_from_grav() 要拿它和加速度计的模长比
            # 较，单位不一致的话"模长比"这个判据就失效了（gps_derived_g 输出的是 g）。
            ref_mag = imu.G0 * np.hypot(
                geo.savgol(tel.t, gl, 0.5, 2), geo.savgol(tel.t, gt, 0.5, 2)
            )
            est = imu.gravity_from_grav(
                acc, t_imu, grav, t_grav,
                speed_at_acc=geo.interp_to(tel.t, tel.speed, t_imu),
                ref_mag=geo.interp_to(tel.t, ref_mag, t_imu),
            )
            a_long, a_lat = imu.velocity_frame_g(
                acc, t_imu, est.vector, tel.t, gl, gt
            )

        tel.t_imu = t_imu
        tel.acc = acc
        tel.grav = grav
        tel.gfield = imu.GField(gravity=est)
        tel.a_long = a_long
        tel.a_lat = a_lat
        tel.a_long_gps = gl
        tel.a_lat_gps = gt
        warns.extend(tel.gfield.notes)

        # 用**最终输出**重新算一致性指标。
        # 只在"真正在行驶"的样本上算：静止/极低速段的加速度基本是噪声底，
        # 混进来会把相关性整体拉低，反映不出实际可用性。
        def _r(a: np.ndarray, b: np.ndarray) -> float:
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 100 or float(np.std(a[m])) < 1e-9 or float(np.std(b[m])) < 1e-9:
                return float("nan")
            return float(np.corrcoef(a[m], b[m])[0, 1])

        moving = geo.interp_to(tel.t, tel.speed, t_imu) > 3.0
        # 两边都先做 0.5 秒平滑再比，否则 GPS 求导噪声会把相关性压得极低
        a_lat_s = geo.savgol(t_imu, a_lat, 0.5, 2)
        a_lon_s = geo.savgol(t_imu, a_long, 0.5, 2)
        gt_s = geo.savgol(t_imu, geo.interp_to(tel.t, gt, t_imu), 0.5, 2)
        gl_s = geo.savgol(t_imu, geo.interp_to(tel.t, gl, t_imu), 0.5, 2)
        r_lat = _r(a_lat_s[moving], gt_s[moving])
        r_lon = _r(a_lon_s[moving], gl_s[moving])
        tel.gfield.lateral_r = r_lat
        tel.gfield.longitudinal_r = r_lon
        tel.gfield.quality = float(np.nanmean([abs(r_lat), abs(r_lon)]))

        for nm, r, hint in (
            ("横向", r_lat, "相机可能随头部晃动，或支架松动"),
            ("纵向", r_lon, "速度/轨迹数据可能不完整"),
        ):
            if np.isfinite(r) and abs(r) < 0.6:
                warns.append(f"输出 G 与 GPS 参考的{nm}一致性偏低（r={r:.2f}）：{hint}。")
    else:
        gl, gt = imu.gps_derived_g(tel.t, tel.speed, tel.heading, kappa=tel.curvature)
        tel.a_long_gps = gl
        tel.a_lat_gps = gt
        tel.t_imu = tel.t
        tel.a_long = gl
        tel.a_lat = gt

    return tel


def load(mp4_path: str | Path, *, verbose: bool = True) -> Telemetry:
    """
    从 GoPro MP4 加载完整遥测。

    这是整个工具链的入口，对应 `read_streams()` + 清洗 + G 值提取的封装。
    """
    mp4_path = Path(mp4_path)
    warns: list[str] = []

    streams = gpmf.read_streams(mp4_path)
    if not streams:
        raise RuntimeError(
            f"{mp4_path.name} 里没有解析出任何遥测流。请确认录制时开启了 GPS，"
            "且视频没有经过剪辑/转码。"
        )

    gps = _decode_gps(streams, warns)
    if gps is None:
        raise RuntimeError(
            f"{mp4_path.name} 里没有可用的 GPS 数据（需要 GPS5 或 GPS9 流）。"
            "GoPro 只能在「开启 GPS」的模式下记录，且必须在室外搜到星。"
        )
    gps_name, t, lat, lon, alt, spd2, spd3, fix, dop = gps

    # ---- 清洗：时间必须单调，定位点必须有效 ----
    order = np.argsort(t)
    t, lat, lon, alt, spd2, spd3 = (a[order] for a in (t, lat, lon, alt, spd2, spd3))
    if fix is not None:
        fix = fix[order]
        # fix=0 表示无效定位，这些点的经纬度是垃圾值
        if np.any(fix <= 0):
            n_bad = int((fix <= 0).sum())
            warns.append(f"剔除了 {n_bad} 个无效定位点（fix=0）。")
    if dop is not None:
        dop = dop[order]

    if t.size < 30:
        raise RuntimeError("有效 GPS 点太少，无法分析（至少需要约 30 个点）。")

    # 用中值滤波去掉异常的定位跳点，避免污染轨迹与航向
    lat = geo.despike(t, lat, window_seconds=2.0, n_sigma=5.0)
    lon = geo.despike(t, lon, window_seconds=2.0, n_sigma=5.0)

    # ---- 投影到平面坐标 + 平滑 ----
    trk = derive_track(t, lat, lon, speed=np.asarray(spd2, dtype=np.float64))

    # 速度：优先用 GoPro 的多普勒速度，异常时退回位置差分
    speed = np.asarray(spd2, dtype=np.float64).copy()
    d_from_pos = np.gradient(np.asarray(trk["dist"], dtype=np.float64), t)
    bad = (~np.isfinite(speed)) | (speed < 0) | (speed > 120.0)
    if bad.any():
        warns.append(f"有 {int(bad.sum())} 个速度样本异常（负数或超过 432 km/h），已用位置差分替换。")
        speed[bad] = d_from_pos[bad]
    speed = np.clip(speed, 0.0, 120.0)
    speed = geo.despike(t, speed, window_seconds=1.0, n_sigma=5.0)

    tel = Telemetry(
        source=mp4_path.name,
        t=t, lat=lat, lon=lon, alt=alt,
        speed=speed, speed3d=np.asarray(spd3, dtype=np.float64),
        x=trk["x"], y=trk["y"],
        dist=trk["dist"], heading=trk["heading"], curvature=trk["curvature"],
        x_raw=trk["x_raw"], y_raw=trk["y_raw"],
        fix=fix, dop=dop,
        lat0=float(trk["lat0"]), lon0=float(trk["lon0"]),
        streams=streams,
        warnings=warns,
    )
    if gps_name == "GPS5":
        warns.append("该视频只有 GPS5（10Hz）。HERO11 通常还会记录 18Hz 的 GPS9，"
                     "若本机固件未启用，圈速分辨率约为 0.1 s。")

    # ---- IMU：G 值提取 ----
    imu_data = _select_imu(streams, warns)
    t_acc, acc = imu_data if imu_data else (None, None)
    gyro_sel = _select_gyro(streams)
    tel.gyro = gyro_sel[1] if gyro_sel else None
    grav_stream = streams.get("GRAV")
    attach_imu(
        tel,
        t_acc,
        acc,
        grav=grav_stream.physical() if grav_stream is not None else None,
        t_grav=grav_stream.times if grav_stream is not None else None,
    )

    if verbose:
        print(tel.summary())
        if tel.gfield is not None:
            print("\n— G 值提取 —")
            print(tel.gfield.describe())
        for w in dict.fromkeys(warns):  # 去重但保持顺序
            print(f"  ⚠ {w}")

    return tel


__all__ = ["Telemetry", "attach_imu", "derive_track", "load"]
