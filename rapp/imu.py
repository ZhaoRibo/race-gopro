"""
加速度计安装角标定 —— 把相机坐标系的三轴加速度变成赛车意义上的 G 值
=================================================================

问题在哪
--------
GoPro 的 `ACCL` 流给出的是**相机自己坐标系**下的三轴加速度：
    X 轴 = 画面右方
    Y 轴 = 画面下方
    Z 轴 = 镜头指向方向（向前）

但我们真正关心的是赛车坐标系的：
    纵向加速度（加速 / 刹车）
    横向加速度（转向，也就是"几个 G"）

这两个坐标系之间差了一个**固定的刚体旋转** —— 因为支架装好后相机相对车是不动的。
只要把这个旋转求出来，就能算出真正的 G 值。

而且这个旋转**不能靠猜**：
- 相机装在前整流罩上有俯仰角（通常是向下 10~30°），
- 也会有滚转角（歪着装），
- 还会有偏航角（镜头稍微偏左或偏右）。

三步标定法（全部由数据自动求出，无需人工输入安装角度）
----------------------------------------------------
第 1 步：找重力方向
    加速度计静止时会读到 +1g，方向指向"上"。整场练习里加速和刹车的时间大致相等、
    转弯左右也大致对称，所以整段数据的**均值**基本就等于重力向量（还顺便吸收了
    零偏 bias）。用均值而不是单点，是为了抗噪。

第 2 步：扣掉重力、投影到水平面
    减去均值后剩下的就是纯运动加速度。把它投影到垂直于重力方向的平面上，
    就得到一个二维平面上的加速度向量 (a1, a2)。此时还差一个绕重力轴的旋转未知。

第 3 步：用 GPS 求那个剩下的旋转角
    这台车上唯一已知方向的参考量是 **GPS 速度的变化率 dv/dt** —— 它必然沿着车的
    纵轴。设 (a1, a2) 旋转 θ 角后得到纵向加速度，要求它最接近 dv/dt：

        最小化  Σ (a1·cosθ + a2·sinθ − dv/dt)²

    令 c = cosθ, s = sinθ，这是关于 (c, s) 的**线性最小二乘问题**，
    有解析解（一个 2×2 线性方程组），不用迭代：

        [Σa1²   Σa1a2] [c]   [Σ(dv/dt)·a1]
        [Σa1a2  Σa2² ] [s] = [Σ(dv/dt)·a2]

    解出来再归一化，(c, s) 就是纵轴的朝向。这个方法的妙处是**符号自动正确**：
    如果方向反了 180°，最小二乘会给出负解，正好把符号纠正过来。

对照 Python 概念：这三步就是 `np.linalg.lstsq` 的一个特例，只是矩阵只有 2×2。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from . import geo

G0 = geo.G0

CAL_FILTER_HZ = 4.0
"""标定前对加速度计做低通的截止频率。

为什么必须有这一步：卡丁车没有悬挂，200Hz 采样里混着几十 m/s² 的振动尖峰。
拿原始数据直接做最小二乘拟合、算相关系数，真实驾驶信号会被噪声淹没 ——
实测同一份数据，不滤波时标定质量 0.09，先低通是 0.85。
4Hz 保留了刹车/转向动作（0~5Hz），足以覆盖卡丁车的驾驶动态。
"""


def _signed_permutations() -> tuple[np.ndarray, ...]:
    """全部 48 个「带符号的置换矩阵」（3 个轴的 6 种排法 × 每种轴 2 个符号）。

    为什么要老老实实枚举，而不是写死一个换轴公式：
        GRAV 与 ACCL 的轴约定**并不相同**。实测同一台 HERO11 的两段素材里，
        GRAV 恰好把 ACCL 的前两个分量对调了 —— 注意这是一个**转置**，
        det = −1，**不是旋转**。所以「求一个旋转矩阵」的思路（SVD/Kabsch）
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
    顺带把加速度计的标度误差一起校掉了（实测 9.90~9.93）。"""

    axis_score: float
    """轴映射判据的得分（越高越好）。"""

    axis_corr: float
    """扣重力后加速度模长与 GPS 平面加速度模长的相关系数。"""

    axis_ratio: float
    """两者模长之比的中位数。理想 = 1。"""

    def describe(self) -> str:
        return (f"GRAV 轴变换 det={np.linalg.det(self.axis_map):+.0f}，"
                f"判据得分 {self.axis_score:.3f}"
                f"（r={self.axis_corr:+.3f}，模长比 {self.axis_ratio:.3f}）")


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
    把 GRAV 流变成**逐时刻的重力矢量** —— 这是"仅用 GRAV 扣重力"的核心一步。

    为什么逐时刻比"整个录制定一个固定重力方向"好：
        GRAV 是相机自己融合出来的重力方向，它跟着相机一起转。所以即使相机在
        录制中途被人碰了一下、或支架有弹性、或传感器温漂，它也能自动跟上 ——
        而固定方向的做法会把那一次扰动**永久**留在后半段数据里。

    轴约定：GRAV 和 ACCL 的轴序不一样（实测是前两个分量对调，见
    `_signed_permutations` 的说明），所以不能直接相减。这里用物理判据自动挑：
        正确的换轴 → 扣掉重力后剩下的动态加速度，其模长应该与 GPS 独立算出的
        平面加速度模长 sqrt((dv/dt)² + (v·ω)²) 高度相关，且两者量级接近。
        错误的换轴会在数据里留下 1 g 以上的虚假"重力残差"，
        相关系数明显下降、模长比明显偏离 1。实测正解 r≈0.72~0.84、比值 1.04~1.10，
        次优解比值就跳到 1.2 以上。

    参数
    ----
    ref_mag : 可选，(N,) GPS 推算的平面加速度模长（对齐 t_acc），
              **单位必须是 m/s²**（要拿它和加速度计的模长比），
              即 sqrt((dv/dt)² + (v·ω)²) 本身，不要先除以 g。
              不给就跳过自动识别、退回「前两个分量对调」这一实测默认值。
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
class MountCalibration:
    """标定结果。"""

    up_hat: np.ndarray
    """相机坐标系下的"上"方向单位向量，shape (3,)。"""

    gravity_mag: float
    """估计出的重力大小 m/s²。正常应该接近 9.81；明显偏离说明相机标定有问题。"""

    yaw: float
    """绕重力轴安装角，弧度。0 表示相机 X 轴正好对着车的右侧。"""

    quality: float
    """标定质量：纵向与横向两条物理一致性的平均。>0.8 很好，<0.5 说明支架不牢或相机在动。"""

    lateral_r: float
    """横向 G 与 GPS 推出的横向加速度的相关系数。这是**最可信的校验指标**。"""

    longitudinal_r: float
    """纵向 G 与 GPS 速度变化率的相关系数。"""

    lateral_sign: float
    """横向符号，+1 表示最终输出里"正值 = 左转"（ISO 8855 车辆坐标系约定）。"""

    dc_long: float
    """纵向低频直流偏移 m/s²，已从 GPS 参考量对齐（project() 会自动减掉）。"""

    dc_lat: float
    """横向低频直流偏移 m/s²，同上。"""

    projection: str
    """输出 G 用的投影方式：'velocity' = 投影到速度坐标系（推荐），
    'body' = 固定车身轴（仅在 GPS 参考量不可用时退回）。

    注意：calibrate() 里一律先填 'body'，真正的输出由 telemetry.attach_imu()
    决定 —— 标定只看得到加速度计，不知道 GPS 参考量能不能用。
    """

    accel_has_gravity: bool
    """原始 ACCL 是否含重力分量。"""

    gravity_source: str
    """最终采用的重力方向来源。"""

    tilt_disagreement_deg: float
    """最佳候选与其它候选的最大夹角（度），用来反映估计的不确定度。"""

    candidates: list[tuple[str, float, float]]
    """所有候选的 (来源, 与最佳方向的夹角°, 得分)，供排查用。"""

    notes: list[str]

    def describe(self) -> str:
        proj = {
            "grav": "速度坐标系 + **逐时刻** GRAV 扣重力",
            "velocity": "速度坐标系（纵向/横向按**速度方向**分解）",
            "body": "固定车身轴（**退化方案**，纵向 G 不可信）",
        }.get(self.projection, self.projection)
        lines = [
            f"重力大小      : {self.gravity_mag:.3f} m/s²  (理论 9.807)",
            f"重力来源      : {self.gravity_source}",
            f"输出投影      : {proj}",
            f"G 值一致性    : {self.quality:.3f}   "
            f"(横向 r={self.lateral_r:+.3f} / 纵向 r={self.longitudinal_r:+.3f})",
            f"横向符号约定  : 正值 = {'左转' if self.lateral_sign > 0 else '右转'}",
        ]
        # 走 GRAV 逐时刻路线时，下面这些量是"固定车身轴"那套标定算出来的，
        # 已经不再参与 G 值计算。留着会让人误以为它们在起作用，所以整段不打印。
        if self.projection == "grav":
            return "\n".join(lines) + "\n" + "\n".join(f"提示: {n}" for n in self.notes)

        lines.append(f"安装偏航角    : {np.degrees(self.yaw):+.1f}°")
        lines.append(
            f"低频直流修正  : 纵向 {self.dc_long:+.3f} / 横向 {self.dc_lat:+.3f} m/s² "
            f"({np.hypot(self.dc_long, self.dc_lat) / G0:.3f} g)"
        )
        if np.isfinite(self.tilt_disagreement_deg):
            lines.append(f"候选方向最大夹角: {self.tilt_disagreement_deg:.1f}°")
        if len(self.candidates) > 1:
            lines.append("候选对比:")
            for name, ang, sc in self.candidates:
                lines.append(f"    {name:<14} 与最佳夹角 {ang:5.1f}°   得分 {sc:.3f}")
        lines.extend(f"提示: {n}" for n in self.notes)
        return "\n".join(lines)


def _horizontal_basis(up_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    构造一组与重力方向垂直的正交基 (e1, e2)，构成水平面。

    任意取一个不与 up 平行的参考向量做叉积即可，得到的 e1、e2 都在水平面内。
    """
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(ref, up_hat))) > 0.9:
        ref = np.array([0.0, 0.0, 1.0])
    e1 = np.cross(up_hat, ref)
    n = np.linalg.norm(e1)
    e1 = e1 / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    e2 = np.cross(up_hat, e1)
    return e1, e2


def calibrate(
    acc: np.ndarray,
    t_acc: np.ndarray,
    gps_speed: np.ndarray,
    t_gps: np.ndarray,
    gps_x: np.ndarray | None = None,
    gps_y: np.ndarray | None = None,
    grav: np.ndarray | None = None,
    gyro: np.ndarray | None = None,
) -> MountCalibration:
    """
    标定加速度计 → 赛车坐标系。

    参数
    ----
    acc : (N, 3) 相机坐标系原始加速度，m/s²
    t_acc : (N,) 对应时间，秒
    gps_speed : (M,) GPS 地速，m/s
    t_gps : (M,) 对应时间，秒
    gps_x, gps_y : 可选，GPS 平面坐标（米）。给了就能算出横向加速度参考量 ——
                  这是**唯一能验证标定对不对**的手段
    grav : 可选，(N, 3) GRAV 流。只作为诊断对照，不作为主要依据
    gyro : 可选，(N, 3) 陀螺仪 rad/s。**强烈建议提供** —— 行驶中转弯的
           旋转轴就是竖直方向，这是定重力方向最可靠的判据
    """
    notes: list[str] = []
    acc = np.asarray(acc, dtype=np.float64)
    t_acc = np.asarray(t_acc, dtype=np.float64)

    # ---------- 第 0 步：先低通 ----------
    # 这一步是必须的，不是可选优化。卡丁车没有悬挂，200Hz 加速度计里混着
    # 几十 m/s² 的振动尖峰；直接拿原始数据做拟合和相关性评估，
    # 真实驾驶信号会被噪声完全淹没（实测相关性从 0.85 掉到 0.09）。
    acc_f = geo.lowpass(t_acc, acc, CAL_FILTER_HZ)

    # ---------- 参考量 ----------
    speed_clean = geo.despike(t_gps, gps_speed, window_seconds=1.0, n_sigma=4.0)
    speed_smooth = geo.savgol(t_gps, speed_clean, window_seconds=0.6, poly=2)
    # 纵向参考 dv/dt 是对 10Hz 速度求导：噪声 0.77 m/s²、信号本身只有 1.14 m/s²
    # （信噪比约 1.5）。再平滑一次，否则它会因为纯噪声而毁掉相关性评估。
    lon_raw = np.gradient(speed_smooth, t_gps)
    lon_ref = geo.interp_to(t_gps, geo.savgol(t_gps, lon_raw, 0.4, 2), t_acc)

    # 横向参考 a = v·ω。ω 必须由**平滑后航向角的变化率**得到，
    # 不能对位置二次求导算曲率 —— 10Hz 定位配 1.5m 噪声，二次求导会把噪声
    # 放大到离谱的程度（实测曲率半径算出 0 米、角速度算出 187 rad/s）。
    lat_ref = None
    if gps_x is not None and gps_y is not None:
        omega = geo.robust_yaw_rate(t_gps, np.asarray(gps_x), np.asarray(gps_y))
        lat_ref = geo.interp_to(t_gps, speed_smooth * omega, t_acc)

    # ---------- 第 1 步：定出"上"方向 ----------
    # 【最关键的一步，也是栽过跟头的地方】
    #
    # 直觉做法是"取整场加速度均值"或"取车辆静止时的均值"，但它们都有前提：
    # 相机在整个录制过程中必须**刚性固定**。实测这份 HERO11 素材恰恰不满足 ——
    # 行驶结束后（约 640 s 起）人把相机取了下来，朝向变了 43°~123°。
    # 把这些样本混进平均值，平均矢量会被"拉短"（实测只剩 6.28 m/s²，远小于 9.81），
    # 后果是重力被低估 36%、纵向 G 整体偏 -0.4 g、G-G 图整个歪掉。
    #
    # 可靠的判据是**陀螺仪主轴**：相机刚性装在车上跑圈时，最主要的旋转就是
    # 绕竖直轴的偏航，所以行驶期间角速度向量的主方向（协方差最大特征向量）
    # 就是竖直方向。它不需要任何"静止"假设，也不要求全片刚性 ——
    # 实测逐圈稳定在 8° 以内。
    speed_at_acc = geo.interp_to(t_gps, gps_speed, t_acc)
    mean_acc = acc_f.mean(axis=0)
    acc_mean_mag = float(np.linalg.norm(mean_acc))
    driving = speed_at_acc > 3.0  # 只取真正在行驶的样本
    ref = acc_f[driving] if int(driving.sum()) > 200 else acc_f

    up_hat: np.ndarray | None = None
    gravity_source = ""
    if gyro is not None and np.asarray(gyro).size >= 3 and int(driving.sum()) > 200:
        gyro_f = geo.lowpass(t_acc, np.asarray(gyro, dtype=np.float64), CAL_FILTER_HZ)
        _, vec = np.linalg.eigh(np.cov(gyro_f[driving].T))
        u = vec[:, -1]
        n = float(np.linalg.norm(u))
        if n > 1e-9:
            up_hat = u / n
            gravity_source = "陀螺仪主轴（行驶中转弯的旋转轴 = 竖直方向）"

    if up_hat is None:
        # 退路：只能靠加速度均值。此时请先确认相机全程没有被移动过 ——
        # 一个粗略自查方法是看 |mean(acc)| 是否接近 9.8，明显偏小就说明被动过。
        m = ref.mean(axis=0)
        n = float(np.linalg.norm(m))
        if n > 1.0:
            up_hat = m / n
            gravity_source = "加速度均值"
        else:
            up_hat = np.array([0.0, 0.0, 1.0])
            gravity_source = "默认朝向（不可靠）"
            notes.append(
                f"加速度均值只有 {n:.2f} m/s²，判断该流已做重力补偿；"
                "将用相机 Z 轴估计上下方向，G 值精度会下降。"
            )

    # 重力大小：赛道是平的，车不会一直向上加速，所以竖直方向的**动态分量
    # 在一圈里平均为零** —— 行驶样本在 up 方向的投影均值就等于 g。
    # 这一步顺带把加速度计的标度误差自动校准掉了（本例实测 9.94 m/s²）。
    gravity_mag = float(np.mean(ref @ up_hat))
    if gravity_mag < 0.0:
        up_hat = -up_hat
        gravity_mag = -gravity_mag
    accel_has_gravity = 5.0 < gravity_mag < 15.0
    if not accel_has_gravity:
        notes.append(
            f"重力估计为 {gravity_mag:.2f} m/s²，偏离 9.81 较多，请检查相机支架是否松动。"
        )

    # 诊断用：其它几种"上方向"估计与最终结果差多少
    candidate_up: list[tuple[str, np.ndarray]] = []
    if acc_mean_mag > 1.0:
        candidate_up.append(("加速度均值", mean_acc / acc_mean_mag))
    if grav is not None and np.asarray(grav).size >= 3:
        g = np.asarray(grav, dtype=np.float64).reshape(-1, 3)
        g = g[np.isfinite(g).all(axis=1)]
        if g.shape[0] >= 10:
            g_mean = g.mean(axis=0)
            n = float(np.linalg.norm(g_mean))
            if n > 1e-6:
                # GRAV 的幅值在不同固件里或是 m/s²、或是归一化到 ±1，
                # 但方向总是有的；符号对准最终选定的 up。
                up_g = g_mean / n
                if float(np.dot(up_g, up_hat)) < 0:
                    up_g = -up_g
                candidate_up.append(("GRAV 流", up_g))
    if ref.shape[0] > 50:
        _, vec2 = np.linalg.eigh(np.cov((ref - ref.mean(axis=0)).T))
        u2 = vec2[:, 0]
        if float(np.dot(u2, up_hat)) < 0:
            u2 = -u2
        candidate_up.append(("加速度方差最小方向", u2))

    # ---------- 第 2 步：拟合安装偏航角 ----------
    # 重力已经由静止段精确确定，这一步只需求出车辆纵轴朝向。
    #
    # 关键选择：用**横向参考**（v·ω）拟合，而不是纵向参考（dv/dt）。
    # 因为 dv/dt 是对 10Hz 速度求导：实测噪声量级 0.77 m/s²，
    # 而信号本身标准差只有 1.14 m/s²（信噪比约 1.5）—— 拿它定朝向会被噪声带偏；
    # 横向参考的标准差有 6.7 m/s²，可靠得多。
    #
    # 反面教材：曾经用"两个参考量一起做三元回归"（acc = dv/dt·e纵 + vω·e横 + g），
    # 看似优雅，但弱噪的纵向列会把解带偏 —— 实测算出的重力方向偏了 49°，
    # 后果是纵向 G 出现 -0.52 g 的直流偏置（跑完一圈速度守恒，纵向加速度均值必须为 0）。
    a_dyn = acc_f - up_hat * gravity_mag
    e1, e2 = _horizontal_basis(up_hat)
    a1_all, a2_all = a_dyn @ e1, a_dyn @ e2
    # 只在行驶样本上拟合偏航角：行驶结束后的片段里相机已被移动，
    # 那些样本的"车辆坐标系"本身就不存在了，加进来只会添乱。
    fit = driving if driving.shape[0] == a_dyn.shape[0] else np.ones(a_dyn.shape[0], dtype=bool)
    a1, a2 = a1_all[fit], a2_all[fit]
    lon_fit = lon_ref[fit]
    lat_fit = lat_ref[fit] if lat_ref is not None else None

    def _fit_dir(target: np.ndarray, u1: np.ndarray | None = None, u2: np.ndarray | None = None) -> tuple[float, float]:
        """解出与 target 最匹配的水平方向（在基下的单位向量）。"""
        A = np.column_stack([a1 if u1 is None else u1, a2 if u2 is None else u2])
        A = A - A.mean(axis=0)
        b = target - target.mean()
        try:
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:  # pragma: no cover
            return 1.0, 0.0
        n = float(np.hypot(sol[0], sol[1]))
        return (float(sol[0] / n), float(sol[1] / n)) if n > 1e-9 else (1.0, 0.0)

    if lat_fit is not None:
        left_axis = _fit_dir(lat_fit)  # 目标换成横向加速度，解出的就是"车向左"
    else:
        fc, fs = _fit_dir(lon_fit)
        left_axis = (-fs, fc)

    cl, sl = left_axis
    # 前方向由矢量代数唯一确定：车辆坐标系恒满足「前 × 左 = 上」，即
    #     前 = 左 × 上
    # _horizontal_basis 给的是右手基（e1×e2 = up），代入得 (c,s) = (sl, -cl)。
    # 这样 project() 里的 (-a1*s + a2*c) 恰好还原成拟合出的"左"方向 (cl, sl)。
    #
    # 【为什么不能用纵向参考量去定正负号】dv/dt 的信噪比只有约 1.5（噪声 0.77、
    # 信号 1.14 m/s²），靠它判断"哪边是车头"会判错，而一旦判错横向 G 就会整体
    # 左右翻转 —— 实测与 GPS 参考的相关性从 +0.90 变成 -0.90，G-G 图左右镜像。
    c, s = sl, -cl

    r_lat = _corr(a1 * cl + a2 * sl, lat_fit) if lat_fit is not None else 0.0

    # 纵向一致性**只在直线段评估**。转弯时车身有侧滑角 β，车体系纵向加速度是
    #     a纵 = v̇·cosβ - v·β̇·sinβ - v·ω·sinβ
    # 最后一项在卡丁车漂移时极大（实测能到 ±0.8 g），会把直线段的真实相关性淹掉 ——
    # 全样本算出来是负的。所以先筛出 |ω| 很小的样本，这里 a纵 才真正等于 v̇。
    straight_fit = np.ones(a1.shape[0], dtype=bool)
    if gyro is not None and np.asarray(gyro).size >= 3:
        gy_om = geo.lowpass(t_acc, np.asarray(gyro, dtype=np.float64), CAL_FILTER_HZ) @ up_hat
        straight_fit = (np.abs(gy_om) < 0.15)[fit]
        if int(straight_fit.sum()) < 200:
            straight_fit = np.ones(a1.shape[0], dtype=bool)
    r_lon = _corr((a1 * c + a2 * s)[straight_fit], lon_fit[straight_fit])
    best_score = 0.5 * (abs(r_lon) + abs(r_lat)) if lat_fit is not None else abs(r_lon)
    quality = max(0.0, float(best_score))

    cand_report: list[tuple[str, float, float]] = []
    for src, up in candidate_up:
        ang = float(np.degrees(np.arccos(np.clip(abs(float(np.dot(up, up_hat))), -1.0, 1.0))))
        # 顺带算出"如果改用这个候选方向，横向一致性能做到多少"，
        # 一眼就能看出哪条流（加速度计均值 / GRAV / 方差最小方向）可信
        score = 0.0
        if lat_ref is not None:
            ad = acc_f - up * float(np.dot(mean_acc, up))
            b1, b2 = _horizontal_basis(up)
            cc, ss = _fit_dir(lat_ref, ad @ b1, ad @ b2)
            score = abs(_corr((ad @ b1) * cc + (ad @ b2) * ss, lat_ref))
        cand_report.append((src, ang, score))
    tilt_disagreement = max((a for _, a, _ in cand_report), default=float("nan"))

    if quality < 0.5:
        notes.append(
            "标定质量偏低。常见原因：相机戴在头盔上（随头部转动，无法标定）、"
            "支架松动、或该段视频里几乎没有加速/刹车动作。请把相机刚性固定在车身上。"
        )
    elif quality < 0.75:
        notes.append("标定质量中等，G 值趋势可信，绝对值可能有 10% 左右误差。")
    if accel_has_gravity and not (0.5 * G0 < gravity_mag < 1.5 * G0):
        notes.append(f"重力估计为 {gravity_mag:.2f} m/s²，偏离 9.81 较多，请检查相机支架是否松动。")
    if tilt_disagreement > 25.0:
        notes.append(
            f"各候选重力方向最多相差 {tilt_disagreement:.0f}°，已按静止段定出的方向为准。"
            "同一份文件里 GRAV 与加速度计的轴约定可能并不一致，不要盲信 GRAV。"
        )

    # 横向符号：参考量 lat_ref 本身就是"左转为正"，相关性为正说明符号已经对了
    lateral_sign = 1.0 if r_lat >= 0 else -1.0

    # ---------- 第 3 步：低频直流对齐 ----------
    # 加速度计存在一个低频率的直流偏移（实测这份素材行驶段高达 0.25 g），
    # 而它物理上不可能是真的：跑完一圈速度守恒，纵向加速度的整圈积分必须为 0。
    # 成因本工具无法完备解释（已排除削顶 / 尖峰 / 相机移动 / 轴约定错误），
    # 但 GPS 能独立给出两个可信的低频绝对量：
    #     纵向 mean(dv/dt)  = 0        —— 一圈净速度变化为 0
    #     横向 mean(v·ω)             —— 环形赛道的平均向心加速度，本例 +1.455 m/s²
    # 所以把 IMU 的直流平移到 GPS 给出的值上：低频用 GPS 兜底，
    # 刹车/转向等 0.1Hz 以上的细节仍然全部来自 IMU（这才是装加速度计的意义）。
    e_long_vec = c * e1 + s * e2
    e_lat_vec = lateral_sign * (-s * e1 + c * e2)
    dyn_fit = acc_f[fit] - up_hat * gravity_mag
    dc_long = float(np.mean(dyn_fit @ e_long_vec)) - float(np.mean(lon_fit))
    dc_lat = (
        float(np.mean(dyn_fit @ e_lat_vec)) - float(np.mean(lat_fit))
        if lat_fit is not None
        else 0.0
    )
    if np.hypot(dc_long, dc_lat) > 0.08 * G0:
        notes.append(
            f"加速度计有约 {np.hypot(dc_long, dc_lat) / G0:.2f} g 的低频直流偏移，"
            "已按 GPS 参考量对齐（低频用 GPS、高频用 IMU）。常见原因是支架有弹性、"
            "相机离车辆旋转中心较远，或传感器温漂。"
        )

    return MountCalibration(
        up_hat=up_hat,
        gravity_mag=gravity_mag,
        yaw=float(np.arctan2(s, c)),
        quality=quality,
        lateral_r=float(r_lat),
        longitudinal_r=float(r_lon),
        lateral_sign=lateral_sign,
        dc_long=dc_long,
        dc_lat=dc_lat,
        projection="body",
        accel_has_gravity=accel_has_gravity,
        gravity_source=gravity_source,
        tilt_disagreement_deg=float(tilt_disagreement),
        candidates=cand_report,
        notes=notes,
    )


def project(
    acc: np.ndarray,
    t_acc: np.ndarray,
    cal: MountCalibration,
    cutoff_hz: float = 7.0,
    in_g: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    按标定结果把相机坐标系加速度投影成赛车纵向 / 横向加速度。

    返回 (a_long, a_lat)，单位默认是 g。

    cutoff_hz 是低通截止频率。卡丁车没有悬挂，200Hz 的加速度计里
    大部分能量是发动机和路面传来的振动（10~50Hz），真正的转向/刹车动作在 0~5Hz。
    切在 7Hz 既保留了刹车动作的陡峭前沿，又把振动压掉。
    """
    acc = np.asarray(acc, dtype=np.float64)
    t_acc = np.asarray(t_acc, dtype=np.float64)

    mean_acc = cal.up_hat * cal.gravity_mag
    a_dyn = acc - mean_acc if cal.accel_has_gravity else acc.copy()

    e1, e2 = _horizontal_basis(cal.up_hat)
    a1 = a_dyn @ e1
    a2 = a_dyn @ e2

    c = float(np.cos(cal.yaw))
    s = float(np.sin(cal.yaw))
    a_long = a1 * c + a2 * s
    a_lat = cal.lateral_sign * (-a1 * s + a2 * c)

    # 减去低频直流偏移（由 calibrate() 按 GPS 参考量算出，见那里的说明）
    a_long = a_long - cal.dc_long
    a_lat = a_lat - cal.dc_lat

    # 滤掉振动
    a_long = geo.lowpass(t_acc, a_long, cutoff_hz)
    a_lat = geo.lowpass(t_acc, a_lat, cutoff_hz)

    if in_g:
        a_long = a_long / G0
        a_lat = a_lat / G0
    return a_long, a_lat


def velocity_frame_g(
    acc: np.ndarray,
    t_acc: np.ndarray,
    cal: MountCalibration,
    t_gps: np.ndarray,
    a_long_gps: np.ndarray,
    a_lat_gps: np.ndarray,
    *,
    gravity: np.ndarray | None = None,
    direction_seconds: float = 0.8,
    cutoff_hz: float = 7.0,
    in_g: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    把平面加速度投影到**速度坐标系**，得到纵向 / 横向 G。

    【为什么不能用"固定在相机上的车身轴"—— 本项目最深的坑】

    卡丁车漂移时侧滑角 β 在 ±20~30° 之间变化，所以那个固定的"纵向轴"测到的
    根本不是 dv/dt，而是 |A|·sin(β − β̄) —— 被峰值 ±15 m/s² 的横向加速度调制。
    实测：固定车身轴与 GPS 求导的纵向相关性逐圈只有 −0.38~+0.10（等于噪声）。

    正解是把平面加速度矢量投影到**速度方向**上：

        |A|       ← IMU。矢量模长不受坐标系选择影响（实测与 GPS 的 |A| 相关性 0.85）
        φ         ← GPS。加速度相对速度的夹角，tanφ = (v·ω)/(dv/dt)
        a纵 = |A|·cosφ,   a横 = |A|·sinφ

    实测逐圈纵向 r 从 −0.07 提升到 +0.92（中位）、斜率 +1.11，
    横向 r 从 0.96 提升到 0.99、斜率 1.03。

    【为什么不用陀螺仪积分出航向】积分会漂移：866 秒后偏航误差累积到几十度，
    实测后几圈会重新退化。而 φ 完全由 GPS 给出，无积分、不漂移。
    代价是 φ 的带宽只有约 1 Hz（GPS 10Hz 且航向噪声大），方向的快变部分丢失 ——
    好在侧滑角本身是按"秒"变化的慢量，而**幅度**仍由 IMU 以 200 Hz 提供。

    参数
    ----
    a_long_gps, a_lat_gps : `gps_derived_g()` 的输出（单位 g，这里只取方向，单位无所谓）

    gravity : 可选，(N, 3) **逐时刻**重力矢量（m/s²，和 acc 同一坐标系、同一时间轴），
              由 `gravity_from_grav()` 给出。给了就用它扣重力 —— 这是"仅用 GRAV"的
              简化路线；不给就退回用 `cal` 里那个整场固定的重力方向 + 低频直流对齐。

    【两条路线的差别】
        `cal` 路线：重力方向由陀螺仪主轴在一次标定里定死，再靠 GPS 补低频直流。
                    好处是不依赖 GRAV 流；坏处是整场只能有一个重力方向。
        `gravity` 路线：GRAV 是相机自己融合的，逐时刻都跟着相机转，
                    天然免疫"录制中途相机被碰了一下 / 支架有弹性 / 温漂"。
                    也没有低频直流可对齐 —— 重力已经逐时刻扣干净了。

    已知残留误差
    ------------
    · 幅度标定在 ±10% 量级（实测逐圈斜率 1.04~1.41）。根源不是本模块，
      而是参照量本身：GPS 的 v·ω 由强平滑后的位置算出，峰值被衰减了约 5~10%，
      于是方向偏"纵向"、幅度偏高（横向通道同样能看到 0.90~0.96 的斜率）。
    · 方向带宽约 1 Hz：快变部分的细节来自 IMU 的幅度，不是方向。
    · 低加速度（|A|<3 m/s²）时 IMU 的幅度有约 1.5 倍的正向偏置（噪声底，
      E|A+噪声| > |A|），会让极低速段（如出场圈起步的抖动）虚高。
    """
    acc = np.asarray(acc, dtype=np.float64)
    t_acc = np.asarray(t_acc, dtype=np.float64)

    # 平面加速度矢量：先低通压掉振动，再去掉重力、只留水平分量
    acc_f = geo.lowpass(t_acc, acc, CAL_FILTER_HZ)
    if gravity is not None:
        # 逐时刻重力（GRAV 路线）。"上方向"也逐时刻取，所以只减掉沿重力方向的
        # 分量 —— 车在弯里侧倾时，真实的水平面是跟着变的。
        g_vec = np.asarray(gravity, dtype=np.float64)
        gn = np.linalg.norm(g_vec, axis=1, keepdims=True)
        up = g_vec / np.where(gn > 1e-9, gn, 1.0)
        dyn = acc_f - g_vec
        dyn = dyn - np.sum(dyn * up, axis=1)[:, None] * up
    else:
        dyn = acc_f - cal.up_hat * cal.gravity_mag if cal.accel_has_gravity else acc_f.copy()
        dyn = dyn - np.outer(dyn @ cal.up_hat, cal.up_hat)
    mag = np.linalg.norm(dyn, axis=1)

    # 方向：对 (dv/dt, v·ω) 这个**二维向量**做平滑再取单位方向。
    # 不能对 atan2 的结果直接平滑 —— 角度在 ±π 处会跳变；
    # 而且加速度接近零时角度本身是纯噪声，向量平滑天然处理了这一点
    # （|v·ω| 和 |dv/dt| 都很小的时候，方向对结果毫无影响）。
    lo = geo.savgol(t_gps, np.asarray(a_long_gps, dtype=np.float64), direction_seconds, 2)
    la = geo.savgol(t_gps, np.asarray(a_lat_gps, dtype=np.float64), direction_seconds, 2)
    n = np.hypot(lo, la)
    ok = n > 1e-9
    safe = np.where(ok, n, 1.0)
    cos_phi = np.where(ok, lo / safe, 1.0)
    sin_phi = np.where(ok, la / safe, 0.0)
    cos_phi = geo.interp_to(t_gps, cos_phi, t_acc)
    sin_phi = geo.interp_to(t_gps, sin_phi, t_acc)

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
         能不用航向角求导就不用：GPS 位置噪声经过一次求导会大幅放大，
         18Hz 采样下航向角的逐点噪声可能达到几十度。

    这条路线的绝对尺度和符号是可靠的，但精度不如标定后的加速度计，
    所以主要用途是**验证加速度计标定是否正确**。
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
    "GravityEstimate",
    "MountCalibration",
    "calibrate",
    "gps_derived_g",
    "gravity_from_grav",
    "project",
    "velocity_frame_g",
]
