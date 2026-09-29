"""
G 值提取 —— 从 IMU 的加速度计算出赛车意义上的纵向 / 横向 G
========================================================

只有两步，没有"标定"这一步了
----------------------------
    1. 扣掉重力      用 GoPro 自己记录的 `GRAV` 流，**逐时刻**扣
    2. 投影到速度坐标系   `|A|` 来自 IMU，方向 φ 来自 GPS

第 1 步：扣重力用 GRAV，不用"标定出一个固定方向"
------------------------------------------------
相机自己会融合出一个重力方向存进 `GRAV` 流（59.94 Hz，单位向量）。用它逐时刻扣，
比"整场算一个固定重力方向"好在**它跟着相机转**：

    · 录制中途相机被碰了一下       → 自动跟上，不会把扰动留在后半段
    · 支架有弹性、过坎时相机在晃   → 自动跟上
    · 传感器温漂                  → 自动跟上

而且逐时刻扣重力**天然没有直流偏置**：跑完一圈速度守恒，纵向加速度的整圈均值
必须为 0，实测本实现能做到 ±0.02 g 以内。

⚠ 但 `GRAV` 和 `ACCL` **不在同一个轴系**，不能直接相减。详见 `_signed_permutations()`
和 `gravity_from_grav()` 里的说明 —— 这是本项目第二个深坑。

第 2 步：投影到速度坐标系，不是固定车身轴
----------------------------------------
**这是本项目第一个、也是最深的坑。**

卡丁车漂移时侧滑角 β 在 ±20~30° 之间变化，所以"固定在相机上的纵向轴"测到的
根本不是 dv/dt，而是 |A|·sin(β − β̄) —— 被峰值 ±15 m/s² 的横向加速度调制。
实测它与 GPS 求导的相关性逐圈只有 −0.38~+0.10（等于噪声）。

正解：

    |A|  ← IMU。矢量模长不受坐标系选择影响（实测与 GPS 的 |A| 相关性 0.85）
    φ    ← GPS。加速度相对速度的夹角，tanφ = (v·ω)/(dv/dt)
    a纵 = |A|·cosφ,   a横 = |A|·sinφ

实测逐圈纵向 r 从 −0.07 提升到 +0.92（中位）、斜率 +1.11，
横向 r 从 0.96 提升到 0.99、斜率 1.03。

对照 Python 概念：这不是 `np.linalg.lstsq` 那类拟合问题，而是一次坐标变换 ——
`np.linalg.norm()` 给出大小、`np.arctan2()` 那一类给出方向，两者拼起来。

验证手段
--------
· **积分守恒**（最硬）：一圈内 ∫max(a纵,0)dt 必须等于速度总增量。实测比值 0.8~1.1。
· **逐圈相关性**：横向与 GPS 的 v·ω 比、纵向与 dv/dt 比。
· ⚠ **不要用"单圈最大 G"比车**：极值被短尖峰主导。实测有辆二冲程卡丁车全场最大的
  "纵向加速 +1.18 g"出现在**车速 0.0 km/h** 的时候 —— 那是怠速把车抖出来的。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from . import geo

G0 = geo.G0

CAL_FILTER_HZ = 4.0
"""扣重力前对加速度计做低通的截止频率。

为什么必须有这一步：卡丁车没有悬挂，200Hz 采样里混着几十 m/s² 的振动尖峰。
拿原始数据直接算相关性，真实驾驶信号会被噪声淹没 ——
实测同一份数据，不滤波时一致性 0.09，先低通是 0.85。
4Hz 保留了刹车/转向动作（0~5Hz），足以覆盖卡丁车的驾驶动态。
"""

PEAK_SMOOTH_S = 1.0
"""算「峰值 G」这类极值指标时的时间平滑窗口，秒。

**为什么必须平滑**：单点极值被短尖峰主导，根本没法横向比车、比圈。实测有辆二冲程
卡丁车全场最大的"纵向加速"是 **+1.18 g**，出现在**车速 0.0 km/h** 的时候 ——
那是发动机怠速把车抖出来的，属于 $E|A+噪声| > |A|$ 的噪声底效应，和驾驶动作无关。
按原始单点取值，这台车看起来比四冲程还猛；平滑 1 秒后就变成 0.58 g，才是真相。

**为什么是 1 秒**：既要压掉尖峰，又不能把真实峰值抹掉。卡丁车最短的加速/刹车段
也有 2~3 秒，1 秒窗口够短。

用峰值指标时**一律走这条路**（`peak_g()`），不要拿裸信号直接 `np.max`。
"""


def peak_g(
    a: np.ndarray, t: np.ndarray, smooth_seconds: float = PEAK_SMOOTH_S
) -> np.ndarray:
    """
    按**时间**平滑后的 G 值，用来算"峰值"这类极值指标。

    注意是按时间平滑，不是按距离 —— 距离网格上同样 1 米的窗口，在 20 km/h 和
    90 km/h 处对应的时间差 4 倍以上，平滑力度完全不一致。
    """
    return geo.savgol(t, np.asarray(a, dtype=np.float64), smooth_seconds, 2)


def _signed_permutations() -> tuple[np.ndarray, ...]:
    """全部 48 个「带符号的置换矩阵」（3 个轴的 6 种排法 × 每种轴 2 个符号）。

    为什么要老老实实枚举，而不是写死一个换轴公式：
        GRAV 与 ACCL 的轴约定**并不相同**。实测同一台 HERO11 的两段素材里，
        GRAV 恰好把 ACCL 的前两个分量对调了 —— 注意这是一个**转置**，
        det = −1，**不是旋转**。所以「求一个旋转矩阵」的思路（SVD / Kabsch）
        从原理上就找不到它，只能枚举符号置换。
        不同固件 / 机型还可能不同，因此这里不写死，让数据自己说话。
    """
    out: list[np.ndarray] = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            m = np.zeros((3, 3))
            for i, (j, sg) in enumerate(zip(perm, signs)):
                m[i, j] = sg
            out.append(m)
    return tuple(out)


_GRAV_AXIS_CANDIDATES = _signed_permutations()


@dataclass
class GravityEstimate:
    """由 GRAV 流求出的**逐时刻**重力矢量。"""

    vector: np.ndarray
    """(N, 3) 重力矢量，单位 m/s²，已经换算到 ACCL 的坐标系、长度对齐 t_acc。"""

    axis_map: np.ndarray
    """(3, 3) GRAV → ACCL 的轴变换。正常是某个带符号置换矩阵（det 可能是 −1）。"""

    magnitude: float
    """有效重力大小 m/s²。由 mean(acc · 重力方向) 求得，
    顺带把加速度计的标度误差一起校掉了（实测 9.84~9.90）。"""

    axis_score: float
    """轴映射判据的得分（越高越好）。"""

    axis_corr: float
    """扣重力后加速度模长与 GPS 平面加速度模长的相关系数。"""

    axis_ratio: float
    """两者模长之比的中位数。理想 = 1。"""

    def describe(self) -> str:
        return (f"GRAV 轴变换  : 判据 r={self.axis_corr:+.3f}、模长比 {self.axis_ratio:.3f}"
                f"（det={np.linalg.det(self.axis_map):+.0f}"
                f"{'，即前后分量对调' if np.linalg.det(self.axis_map) < 0 else ''}）")

    def describe_axis(self) -> str:
        """把轴变换写成人类可读的形式，例如 'GRAV 的 (x,y,z) = ACCL 的 (y,x,z)'。"""
        src = []
        for out_axis in range(3):
            j = int(np.argmax(np.abs(self.axis_map[out_axis])))
            sign = "-" if self.axis_map[out_axis, j] < 0 else ""
            src.append(f"{sign}{'xyz'[j]}")
        return f"GRAV(x,y,z) = ACCL({', '.join(src)})"


def gravity_from_grav(
    acc: np.ndarray,
    t_acc: np.ndarray,
    grav: np.ndarray,
    t_grav: np.ndarray,
    *,
    speed_at_acc: np.ndarray | None = None,
    ref_mag: np.ndarray | None = None,
) -> GravityEstimate:
    """
    把 GRAV 流变成**逐时刻的重力矢量** —— G 值提取的第 1 步。

    轴约定：GRAV 和 ACCL 的轴序不一样（实测是前两个分量对调，见
    `_signed_permutations` 的说明），所以不能直接相减。这里用物理判据自动挑：

        正确的换轴 → 扣掉重力后剩下的动态加速度，其模长应该与 GPS 独立算出的
        平面加速度模长 sqrt((dv/dt)² + (v·ω)²) 高度相关，且两者量级接近。
        错误的换轴会在数据里留下 1 g 以上的虚假"重力残差"，
        相关系数明显下降、模长比明显偏离 1。实测正解 r≈0.72~0.84、比值 1.02~1.06，
        次优解比值就跳到 1.2 以上。

    重力大小的求法：赛道是平的，车不会一直向上加速，所以竖直方向的动态分量
    在一圈里平均为零，于是 g = mean(acc · 重力方向)。这一步顺带把加速度计的
    标度误差一起校掉了（实测 9.84~9.90 m/s²，理论 9.807）。

    参数
    ----
    ref_mag : 可选，(N,) GPS 推算的平面加速度模长（对齐 t_acc），
              **单位必须是 m/s²**（要拿它和加速度计的模长比），
              即 sqrt((dv/dt)² + (v·ω)²) 本身，不要先除以 g。
              不给就跳过自动识别、退回「前两个分量对调」这一实测默认值。
    speed_at_acc : 可选，(N,) 对齐到 t_acc 的车速 m/s。只用它挑出"正在行驶"的
              样本 —— 静止时的加速度是噪声底，混进来会污染判据。
    """
    acc = np.asarray(acc, dtype=np.float64)
    t_acc = np.asarray(t_acc, dtype=np.float64)
    grav = np.asarray(grav, dtype=np.float64).reshape(-1, 3)
    t_grav = np.asarray(t_grav, dtype=np.float64)

    acc_f = geo.lowpass(t_acc, acc, CAL_FILTER_HZ)
    g_on_acc = np.column_stack(
        [geo.interp_to(t_grav, grav[:, k], t_acc) for k in range(3)]
    )
    g_norm = np.linalg.norm(g_on_acc, axis=1, keepdims=True)
    g_hat = g_on_acc / np.where(g_norm > 1e-9, g_norm, 1.0)

    if speed_at_acc is None:
        moving = np.ones(t_acc.size, dtype=bool)
    else:
        moving = np.asarray(speed_at_acc, dtype=np.float64) > 3.0
    if int(moving.sum()) < 200:
        moving = np.ones(t_acc.size, dtype=bool)

    # 默认值：实测把前两个分量对调（真实素材上验证过）。只有在没有 GPS 参考量
    # 时才用它，否则下面的枚举一定会重新选一次。
    swap01 = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    best: tuple[float, np.ndarray, float, float, float] | None = None

    if ref_mag is not None and int(moving.sum()) > 500:
        ref = np.asarray(ref_mag, dtype=np.float64)
        for m in _GRAV_AXIS_CANDIDATES:
            u = g_hat @ m.T
            # 重力大小：竖直方向的动态分量在一圈里平均为零，
            # 所以 mean(acc·上方向) 就等于 g
            g_eff = float(np.mean(np.sum(acc_f[moving] * u[moving], axis=1)))
            if g_eff <= 4.0:  # 方向反了或映射错了
                continue
            dyn = acc_f - g_eff * u
            dyn = dyn - np.sum(dyn * u, axis=1)[:, None] * u  # 去掉竖直残留
            mag = geo.savgol(t_acc, np.linalg.norm(dyn, axis=1), 0.5, 2)
            r = _corr(mag[moving], ref[moving])
            ratio = float(np.median(mag[moving]) / max(np.median(ref[moving]), 1e-9))
            # 相关性越高、量级越接近 1 越好；量级偏离用相对量惩罚，
            # 这样 r 只差一点点时不会因为比值差一倍而误选
            score = r / (1.0 + abs(ratio - 1.0))
            if best is None or score > best[0]:
                best = (score, m, g_eff, r, ratio)

    if best is None:
        m = swap01
        u = g_hat @ m.T
        g_eff = float(np.mean(np.sum(acc_f[moving] * u[moving], axis=1)))
        if g_eff <= 4.0:  # 连默认映射都不对，退成不换轴并把符号正过来
            m = np.eye(3)
            u = g_hat @ m.T
            g_eff = abs(float(np.mean(np.sum(acc_f[moving] * u[moving], axis=1))))
        best = (float("nan"), m, g_eff, float("nan"), float("nan"))

    score, axis_map, g_eff, r, ratio = best
    g_dir = g_hat @ axis_map.T
    return GravityEstimate(
        vector=g_dir * g_eff,
        axis_map=axis_map,
        magnitude=g_eff,
        axis_score=score,
        axis_corr=r,
        axis_ratio=ratio,
    )


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """皮尔逊相关系数，任一边没有变化时返回 0。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return 0.0
    return float(np.corrcoef(a[m], b[m])[0, 1])


@dataclass
class GField:
    """G 值提取的结果与质量指标。由 `telemetry.attach_imu()` 组装。"""

    gravity: GravityEstimate
    """逐时刻重力矢量及其轴变换。"""

    lateral_r: float = float("nan")
    """横向 G 与 GPS 推出的 v·ω 的相关系数。**最可信的校验指标**（实测 ~0.94）。"""

    longitudinal_r: float = float("nan")
    """纵向 G 与 GPS 速度变化率 dv/dt 的相关系数。"""

    quality: float = float("nan")
    """整体一致性：两条相关系数绝对值的平均。>0.8 很好。"""

    notes: list[str] = field(default_factory=list)

    @property
    def gravity_mag(self) -> float:
        return self.gravity.magnitude

    @property
    def gravity_source(self) -> str:
        return "GRAV 流（逐时刻）"

    def describe(self) -> str:
        lines = [
            f"重力大小      : {self.gravity_mag:.3f} m/s²  (理论 9.807)",
            f"重力来源      : {self.gravity_source}",
            "输出投影      : 速度坐标系（纵向/横向按**速度方向**分解）",
            f"G 值一致性    : {self.quality:.3f}   "
            f"(横向 r={self.lateral_r:+.3f} / 纵向 r={self.longitudinal_r:+.3f})",
            self.gravity.describe(),
            "横向符号约定  : 正值 = 左转",
        ]
        lines.extend(f"提示: {n}" for n in self.notes)
        return "\n".join(lines)


def velocity_frame_g(
    acc: np.ndarray,
    t_acc: np.ndarray,
    gravity: np.ndarray,
    t_gps: np.ndarray,
    a_long_gps: np.ndarray,
    a_lat_gps: np.ndarray,
    *,
    cutoff_hz: float = 7.0,
    in_g: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    扣掉重力 → 投影到**速度坐标系**，得到纵向 / 横向 G —— G 值提取的第 2 步。

    原理见模块顶部的说明。这里只强调两处实现上的讲究：

    · 去掉竖直分量用**逐时刻**的重力方向，不是整场一个固定方向。车在弯里侧倾时
      真实的水平面是跟着变的，用固定方向会留下残留。
    · 方向对 (dv/dt, v·ω) 这个**二维向量**平滑后再取单位方向，不能对 atan2 的
      结果直接平滑 —— 角度在 ±π 处会跳变；而且加速度接近零时角度本身是纯噪声，
      向量平滑天然处理了这一点（两个分量都很小时，方向对结果毫无影响）。

    参数
    ----
    gravity : (N, 3) 逐时刻重力矢量（m/s²，和 acc 同一坐标系、同一时间轴），
              由 `gravity_from_grav()` 给出。
    a_long_gps, a_lat_gps : `gps_derived_g()` 的输出（单位无所谓，只取方向）

    已知残留误差
    ------------
    · **幅度在 ±10% 量级**（实测逐圈斜率 1.04~1.41）。根源是参照量本身：
      GPS 的 v·ω 由强平滑后的位置算出，峰值被衰减约 5~10%，于是方向偏"纵向"、
      幅度偏高（横向通道同样能看到 0.90~0.96 的斜率，可佐证不是本模块的问题）。
    · **方向带宽只有约 1 Hz**：快变部分的细节来自 IMU 的幅度，不是方向。
      好在侧滑角本身是按"秒"变化的慢量。
    · 低加速度（|A|<3 m/s²）时 IMU 的幅度有约 1.5 倍的正向偏置（噪声底，
      E|A+噪声| > |A|），会让极低速段（如出场圈起步、停车怠速）虚高。
      所以"单圈最大 G"这类极值指标不可靠，**不要拿它比车**。
    """
    acc = np.asarray(acc, dtype=np.float64)
    t_acc = np.asarray(t_acc, dtype=np.float64)

    # 先低通压掉振动，再逐时刻扣重力
    acc_f = geo.lowpass(t_acc, acc, CAL_FILTER_HZ)
    g_vec = np.asarray(gravity, dtype=np.float64)
    g_norm = np.linalg.norm(g_vec, axis=1, keepdims=True)
    up = g_vec / np.where(g_norm > 1e-9, g_norm, 1.0)

    dyn = acc_f - g_vec
    dyn = dyn - np.sum(dyn * up, axis=1)[:, None] * up  # 只留水平分量
    mag = np.linalg.norm(dyn, axis=1)

    # 方向：来自 GPS，无积分、不漂移
    lo = geo.savgol(t_gps, np.asarray(a_long_gps, dtype=np.float64), 0.8, 2)
    la = geo.savgol(t_gps, np.asarray(a_lat_gps, dtype=np.float64), 0.8, 2)
    n = np.hypot(lo, la)
    ok = n > 1e-9
    safe = np.where(ok, n, 1.0)
    cos_phi = geo.interp_to(t_gps, np.where(ok, lo / safe, 1.0), t_acc)
    sin_phi = geo.interp_to(t_gps, np.where(ok, la / safe, 0.0), t_acc)

    a_long = geo.lowpass(t_acc, mag * cos_phi, cutoff_hz)
    a_lat = geo.lowpass(t_acc, mag * sin_phi, cutoff_hz)
    if in_g:
        a_long = a_long / G0
        a_lat = a_lat / G0
    return a_long, a_lat


def gps_derived_g(
    t_gps: np.ndarray,
    speed: np.ndarray,
    heading_rad: np.ndarray,
    kappa: np.ndarray | None = None,
    smooth_seconds: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    只用 GPS 推算 G 值 —— 不依赖加速度计，作为交叉验证。

    纵向：v 对时间求导（数值微分）
    横向：a_lat = v²·κ（向心加速度公式）。给了 kappa 就用曲率，
         否则退而用航向角变化率 a_lat = v·ω。
         能不用航向角求导就不用：GPS 位置噪声经过一次求导会大幅放大。

    这条路线的绝对尺度和符号是可靠的，但精度不如加速度计，
    所以主要用途有两个：**给第 2 步提供方向 φ**，以及**验证加速度计的标定**。
    """
    t_gps = np.asarray(t_gps, dtype=np.float64)
    speed = np.asarray(speed, dtype=np.float64)
    heading_rad = np.asarray(heading_rad, dtype=np.float64)

    v = geo.savgol(t_gps, speed, smooth_seconds, poly=2)
    a_long = np.gradient(v, t_gps) / G0

    if kappa is not None:
        k = geo.savgol(t_gps, np.asarray(kappa, dtype=np.float64), 0.6, poly=2)
        a_lat = v * v * k / G0
    else:
        h = geo.savgol(t_gps, heading_rad, smooth_seconds, poly=2)
        a_lat = v * np.gradient(h, t_gps) / G0

    return a_long, a_lat


__all__ = [
    "PEAK_SMOOTH_S",
    "GField",
    "GravityEstimate",
    "gps_derived_g",
    "gravity_from_grav",
    "peak_g",
    "velocity_frame_g",
]
