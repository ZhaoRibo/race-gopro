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
        lines = [
            f"重力大小      : {self.gravity_mag:.3f} m/s²  (理论 9.807)",
            f"重力来源      : {self.gravity_source}",
            f"安装偏航角    : {np.degrees(self.yaw):+.1f}°",
            f"标定质量      : {self.quality:.3f}   "
            f"(横向 r={self.lateral_r:+.3f} / 纵向 r={self.longitudinal_r:+.3f})",
            f"低频直流修正  : 纵向 {self.dc_long:+.3f} / 横向 {self.dc_lat:+.3f} m/s² "
            f"({np.hypot(self.dc_long, self.dc_lat) / G0:.3f} g)",
            f"横向符号约定  : 正值 = {'左转' if self.lateral_sign > 0 else '右转'}",
        ]
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


__all__ = ["MountCalibration", "calibrate", "gps_derived_g", "project"]
