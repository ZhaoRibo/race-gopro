"""
合成数据 —— 没有真实视频时，也能跑通并自检整条流水线
==============================================

它用一个简化但物理自洽的车辆模型生成一段"卡丁车练习"遥测：

    1. 造一条闭合赛道（用几个正弦谐波叠加出有快慢弯的圈）
    2. 按物理算速度曲线（过弯受横向抓地上限约束，出弯受加速度上限约束，
       进弯受刹车减速度约束）—— 这就是赛车仿真里最基本的 "速度剖面" 算法
    3. 每一圈让性能参数略有波动，于是圈速自然会产生 0.5~2 秒的差异
    4. 把速度、曲率换算成三轴加速度，再加上**真实的相机安装角**和振动噪声，
       并按 GoPro 的坐标约定（x 右 / y 下 / z 前）输出 ACCL 流
    5. 给 GPS 加上 1.5 m 量级的定位噪声和 18Hz 的采样率

它的第二个用途是**自检**：因为生成时加速度、重力方向都是已知的，所以可以反过来
验证轴识别、重力扣除与 G 值投影对不对（见 `selftest()`）。
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import geo, telemetry
from .geo import G0

# GoPro 的加速度计坐标约定：x = 画面右，y = 画面下，z = 镜头朝前
_GOPRO_TO_VEHICLE = np.diag([1.0, -1.0, 1.0])


def _rot(axis: str, deg: float) -> np.ndarray:
    """绕指定轴旋转的 3×3 矩阵（角度制）。"""
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _track(n: int = 3000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    生成闭合赛道中心线，返回 (x, y, 曲率, 总长)。

    用几组正弦谐波叠加半径，得到一条有直道、有发夹、有连续弯的闭合曲线 ——
    比单纯的椭圆更接近真实卡丁车场。
    """
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = (130.0
         + 40.0 * np.cos(3 * theta + 0.3)
         + 24.0 * np.sin(5 * theta)
         + 14.0 * np.cos(7 * theta + 1.1))
    x = r * np.cos(theta)
    y = r * np.sin(theta) * 0.82

    # 闭合后按弧长重采样，保证后面 1 米就是 1 米
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    d = geo.cumulative_distance(x, y)
    total = float(d[-1])
    s_new = np.linspace(0.0, total, n, endpoint=False)
    xi = np.interp(s_new, d, x)
    yi = np.interp(s_new, d, y)

    xi = np.append(xi, xi[0])
    yi = np.append(yi, yi[0])
    kappa = geo.curvature(xi, yi)[:-1]
    # 曲率再平滑一下，否则数值噪声会让速度曲线出现假的锯齿
    kappa = np.convolve(kappa, np.ones(9) / 9.0, mode="same")
    return xi[:-1], yi[:-1], kappa, total


def _speed_profile(
    kappa: np.ndarray,
    ds: float,
    a_lat_max: float,
    a_acc: float,
    a_brake: float,
    v_cap: float,
) -> np.ndarray:
    """
    求解满足抓地力/动力/刹车约束的速度剖面。

    做法是先算出"每个点的过弯速度上限" v = sqrt(a_lat_max / |κ|)，
    然后反复做两遍扫描：
        正向扫：受限于"上一处速度 + 加速度能力"
        反向扫：受限于"下一处速度 + 刹车能力"
    扫几轮就收敛到最快的那条速度曲线。这是赛道仿真里的经典算法。
    """
    n = kappa.size
    v = np.minimum(np.sqrt(a_lat_max / np.maximum(np.abs(kappa), 1e-5)), v_cap)

    start = int(np.argmin(v))  # 从最慢点出发，收敛最快
    idx = (start + np.arange(2 * n)) % n

    for _ in range(4):
        for k in range(1, 2 * n):
            i, j = idx[k - 1], idx[k]
            v[j] = min(v[j], np.sqrt(v[i] ** 2 + 2.0 * a_acc * ds))
        for k in range(2 * n - 1, 0, -1):
            i, j = idx[k], idx[k - 1]
            v[j] = min(v[j], np.sqrt(v[i] ** 2 + 2.0 * a_brake * ds))
    return v


def _smooth_random(n: int, rng: np.random.Generator, scale: float) -> np.ndarray:
    """生成一段平滑的随机扰动（用低通滤过的白噪声），模拟人开车的节奏起伏。"""
    w = rng.normal(0.0, 1.0, n)
    kernel = np.hanning(max(9, n // 25))
    kernel /= kernel.sum()
    w = np.convolve(np.concatenate([w[-len(kernel):], w, w[: len(kernel)]]), kernel, mode="same")
    w = w[len(kernel) : len(kernel) + n]
    w = w / (np.std(w) + 1e-9)
    return 1.0 + scale * w


@dataclass
class DemoSession:
    """合成数据的结果，附带真值以便自检。"""

    telemetry: telemetry.Telemetry
    truth: dict = field(default_factory=dict)


def make(
    n_laps: int = 8,
    *,
    seed: int = 7,
    gps_rate: float = 18.0,
    imu_rate: float = 200.0,
    mount_pitch: float = -18.0,
    mount_roll: float = 6.0,
    mount_yaw: float = 4.0,
    origin: tuple[float, float] = (31.2304, 121.4737),
    verbose: bool = False,
) -> DemoSession:
    """
    生成一段合成的卡丁车练习遥测。

    mount_pitch/roll/yaw 是相机相对车身的安装角（度）——
    故意设成非零的"歪装"，用来检验标定算法能不能把它算出来。
    """
    rng = np.random.default_rng(seed)
    xc, yc, kappa, total = _track(seed=seed)
    ds = total / xc.size

    # 卡丁车量级的物理参数
    a_lat_max = 1.55 * G0     # 横向抓地力上限 ≈ 1.55 g
    a_acc = 0.42 * G0         # 加速能力
    a_brake = 0.95 * G0       # 刹车减速度
    v_cap = 26.0              # 约 94 km/h

    # 相机相对车身的固定安装角（真值）
    M_mount = _rot("z", mount_roll) @ _rot("x", mount_pitch) @ _rot("y", mount_yaw)

    def lap_profile(lat_scale: float, cap_scale: float, jitter: float):
        """
        生成一圈，返回按**等时间间隔**采样的 (t, s, v, a_long, a_lat, κ)。

        注意采样方式：速度剖面本身是按"等弧长"给的（每步 ds 米），
        而传感器是按"等时间"采样的，所以中间必须做一次重采样，
        否则速度和位移就对不上了。
        """
        v_arc = _speed_profile(kappa, ds, a_lat_max * lat_scale, a_acc, a_brake, v_cap * cap_scale)
        v_arc = v_arc * _smooth_random(v_arc.size, rng, jitter)
        v_arc = np.clip(v_arc, 3.0, v_cap * 1.05)
        # 平滑一下，免得 dv/dt 出现不真实的大尖峰
        v_arc = np.convolve(
            np.concatenate([v_arc[-30:], v_arc, v_arc[:30]]), np.hanning(21) / 10.5, mode="same"
        )[30:-30]

        s_arc = np.arange(v_arc.size) * ds
        t_arc = np.concatenate([[0.0], np.cumsum(ds / np.maximum(v_arc[:-1], 1.0))])

        # 等时间重采样
        t_u = np.arange(0.0, float(t_arc[-1]), 1.0 / imu_rate)
        s_u = np.interp(t_u, t_arc, s_arc, left=0.0, right=total)
        v_u = np.interp(t_u, t_arc, v_arc)
        kappa_u = np.interp(t_u, t_arc, kappa)
        a_long_u = np.gradient(v_u, t_u)      # 纵向加速度（正 = 加速）
        a_lat_u = v_u * v_u * kappa_u         # 横向加速度（正 = 左转，向心加速度指向左）
        return t_u, s_u, v_u, a_long_u, a_lat_u, kappa_u

    # ---- 组装：出场圈（慢）+ N 个飞驰圈 ----
    laps_spec = [(0.72, 0.62, 0.03)]  # 出场圈：抓地和速度都打折
    for _ in range(n_laps):
        laps_spec.append((
            rng.uniform(0.93, 1.0),
            rng.uniform(0.96, 1.0),
            rng.uniform(0.015, 0.035),
        ))

    s_all, t_all, v_all, al_all, at_all, kp_all = [], [], [], [], [], []
    offset = 0.0
    n_path = xc.size
    for t_u, s_u, v_u, along_u, alat_u, kap_u in (
        lap_profile(ls_, cs_, jt) for ls_, cs_, jt in laps_spec
    ):
        s_all.append(s_u + offset)   # 各圈首尾相接，路径连续
        t_all.append(t_u)
        v_all.append(v_u)
        al_all.append(along_u)
        at_all.append(alat_u)
        kp_all.append(kap_u)
        offset += total

    # 把每一圈的时间轴错开，拼成一条连续时间线
    t_global = []
    base = 0.0
    for t_lap in t_all:
        t_global.append(t_lap + base)
        base = float(t_global[-1][-1]) + 1.0 / imu_rate
    t_truth = np.concatenate(t_global)

    # 去掉跨圈接缝处重复的时间点
    keep = np.concatenate([[True], np.diff(t_truth) > 0])
    t_truth = t_truth[keep]
    s_path = np.concatenate(s_all)[keep]
    v_truth = np.concatenate(v_all)[keep]
    a_long_truth = np.concatenate(al_all)[keep]
    a_lat_truth = np.concatenate(at_all)[keep]
    kappa_truth = np.concatenate(kp_all)[keep]

    # ---- 位置：沿路径取点，再加一点横向摆动（模拟走线宽窄差异）----
    s_norm = s_path / total * n_path
    i0 = np.floor(s_norm).astype(int) % n_path
    i1 = (i0 + 1) % n_path
    frac = s_norm - np.floor(s_norm)
    px = xc[i0] * (1 - frac) + xc[i1] * frac
    py = yc[i0] * (1 - frac) + yc[i1] * frac
    tan_x, tan_y = np.gradient(px), np.gradient(py)
    tn = np.hypot(tan_x, tan_y) + 1e-9
    nrm_x, nrm_y = -tan_y / tn, tan_x / tn
    wob = 0.9 * np.sin(2 * np.pi * t_truth / 11.0) + 0.55 * np.sin(2 * np.pi * t_truth / 3.7)
    px += nrm_x * wob
    py += nrm_y * wob

    t_imu = t_truth
    v_imu = v_truth
    a_long_imu = a_long_truth
    a_lat_imu = a_lat_truth

    # ---- 生成相机坐标系下的加速度计读数 ----
    # 车体系（x 右 / y 上 / z 前）下的比力：静止时读数指向"上"
    f_veh = np.column_stack([
        -a_lat_imu,          # 横向：左转时向心加速度朝左，所以 x（向右）为负
        np.full_like(v_imu, G0),
        a_long_imu,
    ])
    # 转到 GoPro 的坐标约定（x 右 / y 下 / z 前），再施加安装角
    f_nominal = f_veh @ _GOPRO_TO_VEHICLE.T
    f_cam = f_nominal @ M_mount.T

    # 卡丁车没有悬挂，振动很凶：用多频正弦叠加模拟发动机与路面振动
    t = t_imu
    vib = sum(
        amp * np.sin(2 * np.pi * f * t + ph)
        for f, amp, ph in ((23.0, 0.45, 0.3), (37.0, 0.30, 1.4), (61.0, 0.22, 2.7))
    )
    f_cam = f_cam + np.column_stack([vib, vib * 0.8, vib * 0.6])
    f_cam += rng.normal(0.0, 0.12, f_cam.shape)

    # ---- 陀螺仪：卡丁车转弯时主要是绕垂直轴的旋转，ω = v·κ ----
    yaw_rate_veh = v_imu * kappa_truth
    omega_veh = np.column_stack([
        np.zeros_like(yaw_rate_veh),
        yaw_rate_veh,                          # 车体系里绕 y（垂直轴）
        np.zeros_like(yaw_rate_veh),
    ])
    omega = (omega_veh @ _GOPRO_TO_VEHICLE.T) @ M_mount.T
    omega += rng.normal(0.0, 0.01, omega.shape)

    # ---- GRAV：相机坐标系下的重力方向 ----
    # 注意符号约定：真实 GoPro 的 GRAV 静止时与 ACCL **同向**（都指向"上"，
    # 即 +1g 的位置），而不是指向地心。实测已验证（静止时两者夹角 0.3°）。
    g_veh = np.tile(np.array([0.0, G0, 0.0]), (v_imu.size, 1))
    g_cam = (g_veh @ _GOPRO_TO_VEHICLE.T) @ M_mount.T
    # 真实相机的 GRAV 不是理想的：姿态融合有零点几度的残余漂移，幅值也有抖动。
    # 加上这些才不至于把自检变成一道送分题。
    g_hat = g_cam / np.linalg.norm(g_cam, axis=1, keepdims=True)
    ref = np.tile(np.array([0.0, 0.0, 1.0]), (g_hat.shape[0], 1))
    p1 = np.cross(g_hat, ref)
    p1 /= np.linalg.norm(p1, axis=1, keepdims=True) + 1e-12
    p2 = np.cross(g_hat, p1)
    tilt = np.radians(0.45)
    g_cam = g_cam + G0 * tilt * (
        np.sin(2 * np.pi * t_imu / 71.0)[:, None] * p1
        + np.cos(2 * np.pi * t_imu / 53.0)[:, None] * p2
    )
    g_cam *= (1.0 + 0.002 * np.sin(2 * np.pi * t_imu / 29.0))[:, None]

    # ---- GPS：抽样到 gps_rate，加定位噪声 ----
    step = max(1, int(round(imu_rate / gps_rate)))
    gidx = np.arange(0, t_imu.size, step)
    t_gps = t_imu[gidx]
    # 定位噪声：白噪声 + 缓慢的漂移（真实 GPS 的误差是两者叠加）
    lat_noise = rng.normal(0.0, 1.0, gidx.size) + 1.6 * np.sin(2 * np.pi * t_gps / 43.0)
    lon_noise = rng.normal(0.0, 1.0, gidx.size) + 1.6 * np.cos(2 * np.pi * t_gps / 37.0)
    x_true, y_true = px[gidx], py[gidx]

    lat0, lon0 = origin
    # 轨迹中心平移到目的经纬度附近
    cx, cy = float(x_true.mean()), float(y_true.mean())
    lat0 = lat0 - cy / geo.EARTH_R * 180 / np.pi
    lon0 = lon0 - (cx / (geo.EARTH_R * np.cos(np.radians(lat0)))) * 180 / np.pi

    lat_true, lon_true = geo.to_latlon(x_true, y_true, lat0, lon0)
    lat_gps = lat_true + lat_noise / geo.EARTH_R * 180 / np.pi
    lon_gps = lon_true + lon_noise / (geo.EARTH_R * np.cos(np.radians(lat0))) * 180 / np.pi
    speed_gps = np.interp(t_gps, t_imu, v_imu) + rng.normal(0.0, 0.12, t_gps.size)

    # ---- 组装 Telemetry ----
    trk = telemetry.derive_track(t_gps, lat_gps, lon_gps, speed=speed_gps)
    speed_gps = np.clip(speed_gps, 0.0, 120.0)

    tel = telemetry.Telemetry(
        source="[合成数据] 演示赛道",
        t=t_gps, lat=lat_gps, lon=lon_gps,
        alt=np.full(t_gps.size, 12.0),
        speed=speed_gps,
        speed3d=speed_gps,
        x=trk["x"], y=trk["y"],
        dist=trk["dist"], heading=trk["heading"], curvature=trk["curvature"],
        x_raw=trk["x_raw"], y_raw=trk["y_raw"],
        lat0=float(trk["lat0"]), lon0=float(trk["lon0"]),
        streams={},
    )
    tel.warnings.clear()

    telemetry.attach_imu(tel, t_imu, f_cam, grav=g_cam, t_grav=t_imu)

    truth = {
        "mount_pitch_deg": mount_pitch,
        "mount_roll_deg": mount_roll,
        "mount_yaw_deg": mount_yaw,
        "a_long_true": geo.interp_to(t_imu, a_long_imu / G0, tel.t_imu),
        "a_lat_true": geo.interp_to(t_imu, a_lat_imu / G0, tel.t_imu),
        "track_length": total,
        "lap_count": len(laps_spec),
    }

    if verbose:
        print(tel.summary())
        if tel.gfield:
            print("\n— G 值提取 —")
            print(tel.gfield.describe())

    return DemoSession(telemetry=tel, truth=truth)


def selftest(seed: int = 7, n_laps: int = 6) -> tuple[bool, list[str]]:
    """
    跑一遍合成数据并**验证标定是否准确**。

    因为生成时的安装角是真值，所以可以直接检查：
        · 标定出的重力大小是否接近 9.81
        · 标定算出的纵向 G 是否与真值高度相关
        · 标定算出的横向 G 是否与真值高度相关
    这是在没有真实视频的情况下，确认整条 IMU 链路正确的最直接办法。
    """
    lines: list[str] = []
    ok = True

    session = make(n_laps=n_laps, seed=seed)
    tel = session.telemetry
    truth = session.truth

    lines.append("【合成数据自检】")
    lines.append(f"  赛道周长      : {truth['track_length']:.1f} m")
    lines.append(f"  合成时长      : {tel.duration:.1f} s")

    if tel.gfield is None:
        lines.append("  ✗ G 值提取失败：没有生成重力解算结果")
        return False, lines

    gf = tel.gfield
    lines.append(f"  真值安装角    : pitch {truth['mount_pitch_deg']:+.1f}° / "
                 f"roll {truth['mount_roll_deg']:+.1f}° / yaw {truth['mount_yaw_deg']:+.1f}°")
    lines.append(f"  重力来源      : {gf.gravity_source}")
    lines.append(f"  解出的重力    : {gf.gravity_mag:.3f} m/s²  (真值 9.807)")
    lines.append(f"  重力轴变换    : {gf.gravity.describe_axis()}")

    # 合成数据里 GRAV 与 ACCL 本来就同轴系，所以解出来必须是恒等变换。
    # 这一条能直接抓出轴识别逻辑被改坏。
    if np.allclose(gf.gravity.axis_map, np.eye(3), atol=1e-6):
        lines.append("  ✓ 轴变换识别为恒等（合成数据 GRAV/ACCL 同轴系，符合预期）")
    else:
        ok = False
        lines.append("  ✗ 轴变换识别错误，本应为恒等：")
        lines.append("    " + np.array2string(gf.gravity.axis_map, precision=0).replace("\n", "\n    "))

    g_err = abs(gf.gravity_mag - G0)
    if g_err > 0.35:
        ok = False
        lines.append(f"  ✗ 重力估计偏差 {g_err:.3f} m/s²，超出容差")
    else:
        lines.append(f"  ✓ 重力估计偏差 {g_err:.3f} m/s²，在容差内")

    # 纵向 G：把提取结果和真值比较
    a_long_true = truth["a_long_true"]
    r_long = float(np.corrcoef(tel.a_long, a_long_true)[0, 1])
    lines.append(f"  纵向 G 相关性 : r = {r_long:.4f}")
    if r_long < 0.85:
        ok = False
        lines.append("  ✗ 纵向 G 与真值相关性偏低（应 > 0.85）")
    else:
        lines.append("  ✓ 纵向 G 与真值高度一致")

    # 横向 G：注意符号约定需要一致（本项目：正 = 左转）
    a_lat_true = truth["a_lat_true"]
    r_lat = float(np.corrcoef(tel.a_lat, a_lat_true)[0, 1])
    lines.append(f"  横向 G 相关性 : r = {r_lat:+.4f}  (正号表示符号约定正确)")
    if r_lat < 0.85:
        ok = False
        lines.append("  ✗ 横向 G 与真值不一致（可能是符号反了或方向估计失败）")
    else:
        lines.append("  ✓ 横向 G 与真值一致，符号约定正确（正 = 左转）")

    # 量级检查
    lines.append(f"  纵向 G 范围   : {np.min(tel.a_long):+.2f} ~ {np.max(tel.a_long):+.2f} g  "
                 f"(真值 {np.min(a_long_true):+.2f} ~ {np.max(a_long_true):+.2f})")
    lines.append(f"  横向 G 范围   : {np.min(tel.a_lat):+.2f} ~ {np.max(tel.a_lat):+.2f} g  "
                 f"(真值 {np.min(a_lat_true):+.2f} ~ {np.max(a_lat_true):+.2f})")

    return ok, lines


def make_test_video(
    path: str | Path,
    *,
    duration: float = 30.0,
    width: int = 1280,
    height: int = 720,
    fps: float = 30.0,
) -> Path:
    """
    用 ffmpeg 的测试图案生成一段占位视频。

    只在演示模式下需要 —— 让你在没有真实素材时也能验证 HUD 叠加功能是否正常。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("找不到 ffmpeg，请先安装：brew install ffmpeg")

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi",
        "-i", f"testsrc2=size={width}x{height}:rate={fps}:duration={duration:.3f}",
        "-f", "lavfi", "-i", f"sine=frequency=220:duration={duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k", "-shortest",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


__all__ = ["DemoSession", "make", "make_test_video", "selftest"]
