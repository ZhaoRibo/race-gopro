"""
轨迹几何与信号处理工具
====================

这一层解决两个问题：

1. **经纬度没法直接做数学运算**。度是角度，不是长度；直接对经纬度求导算速度会出错
   （同样的 0.00001° 在纬度方向和高纬度处的经度方向对应的米数不同）。
   所以先在赛道中心附近做一次**等距圆柱投影**（equirectangular projection），
   把经纬度变成以米为单位的平面坐标 (x, y)。赛道只有几百米，这个投影的
   相对误差在 0.01% 量级，对卡丁车完全够用。

   对照 Python 概念：相当于把 (index, column) 形式的球坐标数据
   重采样到一个局部笛卡尔坐标系，之后所有 numpy 运算都回归正常。

2. **GPS 定位有噪声**（GoPro 民用 GPS 约 1~3 m），**加速度计有振动**（卡丁车没有悬挂，
   200Hz 加速度计会被高频振动淹没）。所以需要有滤波工具。
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

EARTH_R = 6371008.8
"""地球平均半径（米）。WGS84 的平均半径。"""

G0 = 9.80665
"""标准重力加速度。G 值 = 加速度 / G0。"""


# ==========================================================================
# 坐标投影
# ==========================================================================
def to_local_xy(
    lat: np.ndarray, lon: np.ndarray, lat0: float | None = None, lon0: float | None = None
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    经纬度 → 局部平面米制坐标。

    返回 (x, y, lat0, lon0)，其中 x 向东为正，y 向北为正，原点在赛道中心。

    原理：
        x = (lon - lon0) * R * cos(lat0)     # 经度方向要乘 cos(纬度) 才是真实米数
        y = (lat - lat0) * R                 # 纬度方向 1° ≈ R 米
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    lat0 = float(lat.mean()) if lat0 is None else float(lat0)
    lon0 = float(lon.mean()) if lon0 is None else float(lon0)

    x = np.radians(lon - lon0) * EARTH_R * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_R
    return x, y, lat0, lon0


def to_latlon(x: np.ndarray, y: np.ndarray, lat0: float, lon0: float) -> tuple[np.ndarray, np.ndarray]:
    """平面坐标 → 经纬度（上面投影的逆运算）。"""
    lat = lat0 + np.degrees(np.asarray(y) / EARTH_R)
    lon = lon0 + np.degrees(np.asarray(x) / (EARTH_R * np.cos(np.radians(lat0))))
    return lat, lon


# ==========================================================================
# 轨迹几何
# ==========================================================================
def cumulative_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """累计行驶距离（米），单调递增，shape 与输入一致。"""
    d = np.hypot(np.diff(x), np.diff(y))
    return np.concatenate([[0.0], np.cumsum(d)])


def cumulative_distance_from_speed(t: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """
    由速度积分求里程（梯形积分）。

    比"把相邻两点的位置差加起来"准得多：GPS 定位有 1~3m 噪声，
    而相邻两点真实只隔 1~2m，噪声会被当成位移全部累加进去 ——
    实测 1km 的赛道能算出 1.8km。
    而 GoPro 的速度是多普勒测出来的，本身就平缓且精度高。
    """
    t = np.asarray(t, dtype=np.float64)
    v = np.asarray(speed, dtype=np.float64)
    if t.size < 2:
        return np.zeros_like(v)
    dt = np.diff(t)
    seg = 0.5 * (v[:-1] + v[1:]) * dt
    return np.concatenate([[0.0], np.cumsum(seg)])


def heading(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    航向角（弧度，已解卷绕 unwrap）。
    0 = 东，π/2 = 北，逆时针为正 —— 和数学上的极角一致。

    unwrap 很关键：角度在 ±π 处会跳变，不解卷绕的话求导会得到巨大的假尖峰。
    """
    if x.size < 2:
        return np.zeros_like(x, dtype=np.float64)
    h = np.arctan2(np.diff(y), np.diff(x))
    h = np.concatenate([h[:1], h])
    return np.unwrap(h)


def curvature(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    轨迹曲率 κ（1/米）。左转为正，直线为 0。
    用 "相邻三点定圆" 的离散公式：
        κ = 2 * cross(v1, v2) / (|v1| |v2| |v1+v2|) 的等价展开
    """
    if x.size < 3:
        return np.zeros_like(x, dtype=np.float64)
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = np.power(dx * dx + dy * dy, 1.5)
    denom = np.where(denom < 1e-9, 1e-9, denom)
    return (dx * ddy - dy * ddx) / denom


# ==========================================================================
# 一维数据重采样
# ==========================================================================
def interp_to(t_src: np.ndarray, v_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """
    把 (t_src, v_src) 线性插值到 t_dst 上。

    v_src 可以是 (N,) 也可以是多列 (N, K)，像 numpy 的 np.interp 的多通道版本。
    t_dst 超出范围的点用端点值（和 np.interp 的默认行为一致），
    这样不会因为边界外产生 NaN 而污染后续计算。
    """
    t_src = np.asarray(t_src, dtype=np.float64)
    t_dst = np.asarray(t_dst, dtype=np.float64)
    v_src = np.asarray(v_src, dtype=np.float64)
    if t_src.size == 0:
        raise ValueError("源时间轴为空，无法插值")
    if t_src.size == 1:
        shape = (t_dst.size,) if v_src.ndim == 1 else (t_dst.size, v_src.shape[1])
        return np.full(shape, v_src.reshape(-1)[0] if v_src.ndim == 1 else v_src[0])

    if v_src.ndim == 1:
        return np.interp(t_dst, t_src, v_src)
    return np.column_stack([np.interp(t_dst, t_src, v_src[:, k]) for k in range(v_src.shape[1])])


def resample_uniform(t: np.ndarray, v: np.ndarray, rate: float) -> tuple[np.ndarray, np.ndarray]:
    """把不等间隔的采样重采样成固定频率（Hz）的等间隔序列。"""
    t = np.asarray(t, dtype=np.float64)
    if t.size < 2:
        return t, v
    dt = 1.0 / rate
    t_new = np.arange(t[0], t[-1] + dt * 0.5, dt)
    return t_new, interp_to(t, v, t_new)


# ==========================================================================
# 滤波
# ==========================================================================
def _odd_window(window_seconds: float, dt: float, n: int) -> int:
    """把"以秒为单位的窗口"换算成 Savitzky-Golay 要求的奇数点数。"""
    w = int(round(window_seconds / dt))
    w = max(3, w)
    if w % 2 == 0:
        w += 1
    if w > n:
        w = n if n % 2 == 1 else n - 1
    return max(3, w)


def savgol(t: np.ndarray, v: np.ndarray, window_seconds: float, poly: int = 2) -> np.ndarray:
    """
    Savitzky-Golay 平滑。

    它和"滑动平均"的区别是：滑动平均会把弯道处的峰值削掉，
    SG 滤波器在窗口内拟合一个多项式，能保留峰的形状 ——
    测 G 值峰值时必须用它。
    """
    v = np.asarray(v, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    n = t.size
    if n < 5:
        return v.copy()
    dt = float(np.median(np.diff(t))) if n > 1 else 1.0
    if dt <= 0:
        return v.copy()
    w = _odd_window(window_seconds, dt, n)
    if w <= poly:
        return v.copy()
    if v.ndim == 1:
        return sps.savgol_filter(v, w, poly)
    return np.column_stack([sps.savgol_filter(v[:, k], w, poly) for k in range(v.shape[1])])


def lowpass(t: np.ndarray, v: np.ndarray, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """
    零相位巴特沃斯低通滤波。

    为什么必须零相位（filtfilt）：普通滤波器会让信号产生时间延迟，
    而 G 值曲线相对速度曲线的时间对齐是分析刹车点的前提，
    差 0.1 秒就会把刹车点算错好几米。
    """
    v = np.asarray(v, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    n = t.size
    if n < 15:
        return v.copy()
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return v.copy()
    nyq = 0.5 / dt
    wn = min(cutoff_hz / nyq, 0.99)
    if wn <= 0:
        return v.copy()
    b, a = sps.butter(order, wn, btype="low")
    padlen = min(3 * max(len(a), len(b)), n - 1)
    if v.ndim == 1:
        return sps.filtfilt(b, a, v, padlen=padlen)
    return np.column_stack([sps.filtfilt(b, a, v[:, k], padlen=padlen) for k in range(v.shape[1])])


def despike(t: np.ndarray, v: np.ndarray, window_seconds: float = 1.0, n_sigma: float = 4.0) -> np.ndarray:
    """
    去掉孤立尖峰（GPS 漂移、速度跳变）。

    做法：用中值滤波得到一个"可信基线"，凡是偏离基线超过 n_sigma 倍
    稳健标准差（MAD 换算）的点，就用基线值替换。
    用中值而不是均值是因为中值本身不被尖峰影响。
    """
    v = np.asarray(v, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    n = t.size
    if n < 5:
        return v.copy()
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return v.copy()
    w = _odd_window(window_seconds, dt, n)
    if w < 3:
        return v.copy()
    baseline = sps.medfilt(v, kernel_size=w)
    resid = v - baseline
    mad = float(np.median(np.abs(resid - np.median(resid))))
    sigma = 1.4826 * mad  # MAD → 标准差的换算系数
    if sigma <= 0:
        return v.copy()
    bad = np.abs(resid) > n_sigma * sigma
    out = v.copy()
    out[bad] = baseline[bad]
    return out


def wrap_to_pi(a: np.ndarray) -> np.ndarray:
    """把角度规整到 (-π, π]。"""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def robust_yaw_rate(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    smooth_pos: float = 2.0,
    smooth_head: float = 1.5,
) -> np.ndarray:
    """
    由轨迹求转弯角速度（rad/s，逆时针 / 左转为正）。

    必须走"重度平滑位置 → 航向角 → 变化率"这条路，**不能**对位置二次求导算曲率：
    10Hz GPS 配 1.5m 定位噪声，二次求导会把噪声放大到毫无意义的程度 ——
    实测同一份真实数据，曲率法定出半径 0 米、角速度 187 rad/s，
    而本方法给出中位 0.52 rad/s、峰值 6.8 rad/s（对应约 1.4 g），完全合理。

    原理上两者等价（ω = v·κ），差别全在数值稳定性：
    本方法只做一次微分，而且先对航向角做了长时间平滑（相当于拉长差分基线）。
    """
    t = np.asarray(t, dtype=np.float64)
    xs = savgol(t, np.asarray(x, dtype=np.float64), smooth_pos, poly=2)
    ys = savgol(t, np.asarray(y, dtype=np.float64), smooth_pos, poly=2)
    head = np.unwrap(np.arctan2(np.gradient(ys), np.gradient(xs)))
    head = savgol(t, head, smooth_head, poly=2)
    return np.gradient(head, t)


def fit_circle(points: np.ndarray) -> tuple[np.ndarray, float]:
    """
    最小二乘拟合圆（Kåsa 代数法），返回 (圆心, 半径)。
    用来把一段弯道轨迹概括成"这个弯的半径是多少"。
    """
    x = points[:, 0]
    y = points[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:  # pragma: no cover
        return np.array([x.mean(), y.mean()]), 0.0
    cx = sol[0] / 2.0
    cy = sol[1] / 2.0
    r = float(np.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0)))
    return np.array([cx, cy]), r


__all__ = [
    "EARTH_R",
    "G0",
    "cumulative_distance",
    "cumulative_distance_from_speed",
    "curvature",
    "despike",
    "fit_circle",
    "heading",
    "interp_to",
    "lowpass",
    "resample_uniform",
    "robust_yaw_rate",
    "savgol",
    "to_latlon",
    "to_local_xy",
    "wrap_to_pi",
]
