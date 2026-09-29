"""
圈速识别 —— 自动找起终点线、切圈、算分段
=====================================

一个关键认识
------------
**赛道上的任意一点都可以当起终点线。** 因为圈速的定义是"同一位置连续两次经过之间的
时间"，所以只要你选的那个点每一圈都会被经过一次，算出来的圈速就是真实圈速
（只是"第 1 圈"的起点不同而已）。

所以真正的问题不是"哪一点才是起点线"，而是两个工程问题：

    1. 哪些点**不安全**？—— 如果赛道在这里和自己靠得很近（比如发夹弯的两条直道
       只隔 3 米），或者车辆会在这里短暂停留/来回移动，就会误判出多余的"过线"。
    2. 选哪个点最符合用户期待？—— 默认选一个"最偏僻"的（离赛道上其他部分最远），
       这样最不容易误判；同时让第 1 圈尽量从录制开始处算起。

算法
----
第 1 步：沿轨迹每隔几米取一个候选点，算出该点的"门的朝向"（用局部切线方向）。

第 2 步：对每个候选点，求轨迹穿过这个门的时刻。
        门的定义：过点 p、法向为 n 的一条线。点 P 相对门的"带符号距离"是
            s = (P − p) · n
        s 由负变正就是一次正向穿门。穿门时刻用线性插值精确定位：

            t_cross = t_i + (t_{i+1} − t_i) · (−s_i) / (s_{i+1} − s_i)

        这一步就是线性插值找零点，精度取决于 GPS 采样间隔内速度变化不大 —— 
        在 18Hz、20 m/s 时相邻两点间隔约 1.1 m，插值精度约 1 cm，完全够用。

第 3 步：相邻两次穿门的时间差就是单圈时间。要求：
        - 圈时在合理范围（默认 8~300 s）
        - 圈时的一致性够好（变异系数 CoV 不超标），否则说明这个门漏穿/多穿了

第 4 步：给通过检验的门打"偏僻度"分（距离赛道上时间上相隔较远的那些点的最小距离），
        选最偏僻的那一个，避免把发夹弯的两条直道搞混。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import geo, telemetry

MIN_LAP_SECONDS = 8.0
"""比这更短的一定是误判（卡丁车最快圈也在 20 s 以上）。"""

MAX_LAP_SECONDS = 300.0

GATE_HALF_WIDTH = 8.0
"""门的半宽（米）：穿线点必须落在这个范围内才算数。"""


@dataclass
class Gate:
    """起终点线。"""

    x: float
    y: float
    lat: float
    lon: float
    direction: np.ndarray
    """过线方向（单位向量），车必须沿这个方向穿过才算一圈。"""

    clearance: float
    """偏僻度：离"时间上相隔较远"的赛道其他部分的最小距离（米）。越大越安全。"""

    n_laps: int
    lap_cov: float
    """圈时变异系数 = 标准差 / 均值。衡量这个门切出来的圈时是否自洽。"""

    score: float

    gate_index: int
    """在轨迹采样点里的下标，调试用。"""

    def describe(self) -> str:
        return (
            f"起终点线坐标 : {self.lat:.6f}, {self.lon:.6f}\n"
            f"过线方向     : {np.degrees(np.arctan2(self.direction[1], self.direction[0])) % 360:.0f}° (北=90°)\n"
            f"偏僻度       : {self.clearance:.1f} m\n"
            f"圈时变异系数 : {self.lap_cov * 100:.2f}%  (越小越说明切圈正确)"
        )


@dataclass
class Lap:
    """一圈的数据，已重采样到统一的距离网格上，可直接和别的圈逐点比较。"""

    index: int
    """圈号，从 1 开始（不含出场圈）。"""

    t_start: float
    t_end: float
    duration: float
    """单圈时间，秒。"""

    start_index: int
    end_index: int
    """对应 Telemetry 数组里的下标区间。"""

    length: float
    """本圈行驶距离，米。实测值 —— 走线不同长度就不同。"""

    sectors: list[float] = field(default_factory=list)
    """分段用时，秒。"""

    # ---- 统一距离网格上的信号，shape (M,) ----
    grid: np.ndarray | None = None
    """距离网格，米，从起点线起算。"""

    speed: np.ndarray | None = None
    """m/s。"""

    a_long: np.ndarray | None = None
    """纵向 G，+ 加速 / − 刹车。"""

    a_lat: np.ndarray | None = None
    """横向 G，+ 左转 / − 右转。"""

    gx: np.ndarray | None = None
    gy: np.ndarray | None = None
    """轨迹平面坐标，米。"""

    time: np.ndarray | None = None
    """该点在圈内所对应的时间（相对本圈起点，秒）。与 grid 一一对应。
    有了它就能算"圈间 delta 曲线"：同样距离处两圈的时间差。"""

    # ---- 统计量 ----
    max_speed: float = 0.0
    min_speed: float = 0.0
    max_lat_g: float = 0.0
    max_brake_g: float = 0.0
    max_accel_g: float = 0.0
    accel_time_pct: float = 0.0
    """本圈“纵向 G 为正”的时间占比。

    【别当成全油门】GoPro 测不到油门开度。而且卡丁车漂移时车体系纵向加速度里
    混着一项 -v·ω·sinβ（能到 ±0.8 g），所以这个数主要反映“加速过程占了多久”，
    与油门开度没有对应关系。原先叫 full_throttle_pct 是个误导。
    """

    valid: bool = True
    """是否计入稳定性统计。False 表示这是出场圈之类的非正常圈。"""

    @property
    def time_str(self) -> str:
        return format_lap_time(self.duration)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Lap {self.index} {self.time_str}>"


@dataclass
class LapSet:
    """一次练习的完整圈速分析结果。"""

    gate: Gate
    laps: list[Lap]
    out_lap_duration: float | None
    """出场圈（起点线之前的零头）。"""

    sector_count: int
    sector_bounds: list[float]
    """分段边界，以距离（米）表示。"""

    grid: np.ndarray
    """所有圈共用的距离网格。"""

    telemetry: telemetry.Telemetry | None = None

    # ---- 汇总指标 ----
    @property
    def timed_laps(self) -> list[Lap]:
        """计入统计的圈。排除出场圈之类明显偏慢的圈。"""
        out = [l for l in self.laps if l.valid]
        return out or self.laps

    @property
    def best_lap(self) -> Lap | None:
        pool = self.timed_laps
        return min(pool, key=lambda l: l.duration) if pool else None

    @property
    def best_sectors(self) -> list[float]:
        """每个分段的历史最好成绩。"""
        out: list[float] = []
        for k in range(self.sector_count):
            vals = [l.sectors[k] for l in self.timed_laps if len(l.sectors) > k and np.isfinite(l.sectors[k])]
            out.append(min(vals) if vals else float("nan"))
        return out

    @property
    def theoretical_best(self) -> float:
        """理论最佳圈 = 各分段最好成绩之和。这是你"理论上能做到"的极限。"""
        bs = self.best_sectors
        return float(np.nansum(bs)) if bs else float("nan")

    @property
    def mean_lap(self) -> float:
        pool = self.timed_laps
        return float(np.mean([l.duration for l in pool])) if pool else float("nan")

    @property
    def std_lap(self) -> float:
        pool = self.timed_laps
        return float(np.std([l.duration for l in pool])) if pool else float("nan")

    def consistency(self) -> str:
        """用变异系数描述稳定性。赛车上的经验：<0.5% 是职业水准，>3% 说明还在熟悉赛道。"""
        m, s = self.mean_lap, self.std_lap
        if not np.isfinite(m) or m <= 0:
            return "无法评估"
        cov = s / m
        if cov < 0.005:
            level = "极稳定（职业水准）"
        elif cov < 0.01:
            level = "很稳定"
        elif cov < 0.02:
            level = "较稳定"
        elif cov < 0.04:
            level = "一般，仍有提升空间"
        else:
            level = "波动较大，建议先稳定节奏"
        n_excluded = len(self.laps) - len(self.timed_laps)
        tail = f"（已排除 {n_excluded} 个非正常圈）" if n_excluded else ""
        return f"{cov * 100:.2f}% —— {level}{tail}"


# ==========================================================================
def format_lap_time(seconds: float) -> str:
    """把秒数格式化成赛车圈的写法：1:02.345。"""
    if not np.isfinite(seconds):
        return "  --.---"
    m = int(seconds // 60)
    s = seconds - m * 60
    if m:
        return f"{m}:{s:06.3f}"
    return f"{s:.3f}"


def format_delta(seconds: float) -> str:
    """格式化差值，带正负号。"""
    if not np.isfinite(seconds):
        return " --.---"
    return f"{seconds:+.3f}"


# ==========================================================================
# 候选门生成
# ==========================================================================
def _estimate_lap_period(tel: telemetry.Telemetry, min_seconds: float = 10.0) -> float:
    """
    估计"跑一圈要多少秒"。

    原理：把轨迹看成复数序列 z = x + i·y，它每跑一圈就会回到几乎相同的取值，
    所以**自相关函数**在"一圈"这个滞后处会出现峰值。用 FFT 算自相关，
    复杂度 O(n log n)，比逐点比较快得多。

    这个周期后面有两个用途：
        · 只从"一圈"的轨迹里撒候选点（轨迹会重复，多圈撒点纯属浪费）
        · 判断"哪些采样点和当前点属于同一圈"，这是算偏僻度的前提
    """
    t = tel.t
    x = np.asarray(tel.x, dtype=np.float64)
    y = np.asarray(tel.y, dtype=np.float64)
    if t.size < 60:
        return float("nan")

    step = max(1, t.size // 4000)
    tt, xx, yy = t[::step], x[::step], y[::step]
    n = tt.size
    if n < 40:
        return float("nan")
    dt = float(np.median(np.diff(tt)))
    if dt <= 0:
        return float("nan")

    lo = max(2, int(min_seconds / dt))
    hi = min(n // 2, int(600.0 / dt))
    if hi <= lo:
        return float("nan")

    z = (xx - xx.mean()) + 1j * (yy - yy.mean())
    # 用 FFT 做自相关。注意 numpy 的 rfft 只吃实数，这里输入是复数，
    # 所以要先补零到 2n 以上再做完整的 fft，否则会算成循环相关。
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    Z = np.fft.fft(z, n=nfft)
    ac = np.fft.ifft(Z * np.conj(Z)).real[:n]
    ac = ac / np.maximum(np.arange(n, 0, -1), 1)  # 归一化重叠长度
    lag = lo + int(np.argmax(ac[lo:hi]))
    return float(lag * dt)


def _candidate_gates(
    tel: telemetry.Telemetry, spacing: float, lap_period: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    沿轨迹按固定弧长间隔撒候选点，返回 (下标数组, 切向单位向量, 偏僻度)。

    只在**最近一圈**的轨迹上撒点：轨迹本身会重复好几圈，多撒没有意义，
    而且会让 O(候选数 × 采样点数) 的计算量翻好几倍。
    """
    x, y, t, d = tel.x, tel.y, tel.t, tel.dist

    # 只在一圈的轨迹上撒候选点，但要选对是哪一圈。
    #
    # 这里有个很实际的坑：**不能直接取最后一段**。卡丁车练习经常以停车收尾，
    # 实测某次录制的最后 53 秒里车是静止的（全程 25% 的时间速度低于 1 m/s），
    # 在那段上撒点只会撒出一堆相距一两米的点，偏僻度全部接近 0，切圈必然失败。
    #
    # 正确做法：滑窗找"同样时长内实际行驶距离最长"的那一段，也就是最干净的一圈。
    win_len = lap_period * 0.98
    if np.isfinite(lap_period) and win_len > 0:
        starts = np.searchsorted(t, t - win_len)
        span = d - d[starts]                      # 各窗口内覆盖的里程
        usable = (t - t[0]) >= win_len
        if usable.any():
            i_end = int(np.argmax(np.where(usable, span, -1.0)))
            t_lo, t_hi = float(t[starts[i_end]]), float(t[i_end])
        else:
            t_lo, t_hi = float(t[0]), float(t[-1])
    else:
        t_lo, t_hi = float(t[0]), float(t[-1])

    win = (t >= t_lo) & (t <= t_hi)
    if win.sum() < 20:
        win = np.ones_like(t, dtype=bool)

    idx_all = np.where(win)[0]
    dw = d[idx_all] - d[idx_all[0]]
    targets = np.arange(0.0, dw[-1], spacing)
    sel = idx_all[np.clip(np.searchsorted(dw, targets), 0, idx_all.size - 1)]
    idx = np.unique(sel)

    # 再去一次重：低速时多个目标点会落到同一位置
    keep: list[int] = []
    for i in idx:
        if not keep:
            keep.append(int(i))
            continue
        if np.hypot(x[i] - x[keep[-1]], y[i] - y[keep[-1]]) > spacing * 0.5:
            keep.append(int(i))
    idx = np.array(keep, dtype=np.int64)
    if idx.size == 0:
        raise RuntimeError("轨迹太短，无法生成候选起终点线。")

    # 切向：用前后各 3 个点做中心差分，比单点差分抗噪
    lo = np.clip(idx - 3, 0, len(x) - 1)
    hi = np.clip(idx + 3, 0, len(x) - 1)
    tx, ty = x[hi] - x[lo], y[hi] - y[lo]
    norm = np.hypot(tx, ty)
    norm = np.where(norm < 1e-9, 1e-9, norm)
    tangent = np.column_stack([tx / norm, ty / norm])

    # ------------------------------------------------------------------
    # 偏僻度：这个点离"同一圈里其它部分"的赛道最近有多远。
    #
    # 这里有个很容易踩的坑：如果把"时间上相隔较远"当作判据，那么**下一圈的同一个
    # 位置**也会被算进来 —— 而它和当前点的距离本来就是 0（跑的是同一条线）。
    # 正确做法是只看"同一圈内"的采样点，也就是时间差落在一个圈时之内。
    # ------------------------------------------------------------------
    clearance = np.zeros(idx.size, dtype=np.float64)
    span = lap_period * 0.85 if np.isfinite(lap_period) else t_hi - float(t[0])
    for k, i in enumerate(idx):
        lag = np.abs(t - t[i])
        far = (lag > 3.0) & (lag < span)
        if far.sum() < 5:
            clearance[k] = 50.0  # 数据不足，给一个中性值
            continue
        clearance[k] = float(np.min(np.hypot(x[far] - x[i], y[far] - y[i])))

    return idx, tangent, clearance


def _crossings(tel: telemetry.Telemetry, px: float, py: float, direction: np.ndarray) -> np.ndarray:
    """
    求轨迹正向穿过"过 (px, py)、朝向 direction 的起终点门"的所有时刻。

    门的几何关系（这一步很容易搞反，这里写清楚）：
        起终点线是一条**垂直于行进方向**的线，不是顺着行进方向的线。

            direction  = 过线时的行进方向（门的法向）
            n          = (−dy, dx) = 门的横线方向

        于是对轨迹上的点 P：
            s = (P − p) · direction   → 相对门的前后位置，由负变正就是正向过线
            w = (P − p) · n           → 在门宽方向上的偏移，必须落在门宽以内

        过线时刻用线性插值定位（找 s 的零点）：
            t_cross = t_i + (t_{i+1} − t_i) · (−s_i) / (s_{i+1} − s_i)

        在 18Hz、20 m/s 下相邻两点约相隔 1.1 m，插值精度在厘米级，够用。
    """
    direction = np.asarray(direction, dtype=np.float64)
    norm = float(np.hypot(direction[0], direction[1]))
    if norm > 1e-9:
        direction = direction / norm
    n = np.array([-direction[1], direction[0]])

    dx = tel.x - px
    dy = tel.y - py
    s = dx * direction[0] + dy * direction[1]  # 沿行进方向的前后距离
    w = dx * n[0] + dy * n[1]                  # 门宽方向上的横向偏移

    # 负 → 正 的符号变化 = 沿正确方向过线（反向通过不算圈）
    prev, cur = s[:-1], s[1:]
    i = np.where((prev < 0) & (cur >= 0))[0]
    if i.size == 0:
        return np.zeros(0)

    denom = cur[i] - prev[i]
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    frac = np.clip(-prev[i] / denom, 0.0, 1.0)

    # 穿线点必须落在门的宽度内，否则只是从门旁边擦过去
    w_cross = w[i] + frac * (w[i + 1] - w[i])
    inside = np.abs(w_cross) <= GATE_HALF_WIDTH

    # 必须是在行驶中（停车时的定位抖动不算过线）
    moving = tel.speed[i] > 1.0

    ok = inside & moving
    if not ok.any():
        return np.zeros(0)
    i, frac = i[ok], frac[ok]

    t_cross = np.sort(tel.t[i] + frac * (tel.t[i + 1] - tel.t[i]))

    # 去掉过近的重复穿越
    keep = [0]
    for k in range(1, t_cross.size):
        if t_cross[k] - t_cross[keep[-1]] > MIN_LAP_SECONDS:
            keep.append(k)
    return t_cross[keep]


def find_gate(
    tel: telemetry.Telemetry,
    *,
    spacing: float = 3.0,
    max_cov: float = 0.25,
    max_dist_cov: float = 0.12,
    min_laps: int = 2,
) -> Gate:
    """
    自动搜索最佳起终点线。

    一个候选点要同时过三关才会被采用：

        1. **圈时合理**：每一圈都在 8~300 s 之间
        2. **圈时一致**：圈时的变异系数不超过 max_cov（默认 25%）
        3. **每圈距离一致**：每圈行驶距离的变异系数不超过 max_dist_cov（默认 12%）

    第 3 条是抓"门开在赛道自己旁边"最直接的判据 —— 一旦发生漏穿或多穿，
    算出来的每圈距离会变成半圈或两圈，离散度立刻爆掉。

    最后在所有合格的候选里，先筛掉偏僻度不足的（避免误判），
    再选"第一次过线最早"的那个，让第 1 圈尽量贴近录制起点。
    """
    lap_period = _estimate_lap_period(tel)
    idx, tangents, clearances = _candidate_gates(tel, spacing, lap_period)

    candidates: list[tuple[float, float, int, np.ndarray, float]] = []
    for k in range(idx.size):
        i = int(idx[k])
        tc = _crossings(tel, tel.x[i], tel.y[i], tangents[k])
        if tc.size < min_laps + 1:
            continue

        durs = np.diff(tc)
        if np.any(durs < MIN_LAP_SECONDS) or np.any(durs > MAX_LAP_SECONDS):
            continue
        d_mean = float(np.mean(durs))
        if d_mean <= 0:
            continue
        cov = float(np.std(durs) / d_mean)

        # 每圈行驶距离是否一致
        bounds = np.clip(np.searchsorted(tel.t, tc), 0, tel.t.size - 1)
        dists = np.diff(tel.dist[bounds])
        if dists.size == 0 or float(np.mean(dists)) <= 1.0:
            continue
        dist_cov = float(np.std(dists) / np.mean(dists))

        if cov > max_cov or dist_cov > max_dist_cov:
            continue

        candidates.append((cov, dist_cov, i, tangents[k], float(clearances[k])))

    if not candidates:
        raise RuntimeError(
            "没能找到合适的起终点线。可能是：\n"
            "  · 本次录制里连续完成的有效圈数不足 2 圈\n"
            "  · GPS 信号质量太差（室内 / 被建筑遮挡 / 起步阶段还没搜到星）\n"
            "  · 车辆在赛道上长时间停车或多次出场进场\n"
            "可以试试用 --gate 纬度,经度 手动指定赛道上的一个位置。"
        )

    # 偏僻度筛选：门开在赛道自己旁边的话，同一圈会穿过两次，圈速必然算错
    max_clear = max(c[4] for c in candidates)
    floor = max(0.55 * max_clear, GATE_HALF_WIDTH + 2.0)
    safe = [c for c in candidates if c[4] >= floor] or candidates

    def _first_cross(c: tuple) -> float:
        tc = _crossings(tel, tel.x[c[2]], tel.y[c[2]], c[3])
        return float(tc[0]) if tc.size else float("inf")

    safe.sort(key=_first_cross)
    cov, dist_cov, i, tangent, clear = safe[0]

    tc = _crossings(tel, tel.x[i], tel.y[i], tangent)
    lat, lon = geo.to_latlon(np.array([tel.x[i]]), np.array([tel.y[i]]), tel.lat0, tel.lon0)

    return Gate(
        x=float(tel.x[i]),
        y=float(tel.y[i]),
        lat=float(lat[0]),
        lon=float(lon[0]),
        direction=tangent,
        clearance=clear,
        n_laps=int(np.diff(tc).size),
        lap_cov=cov,
        score=1.0 / (1.0 + cov + dist_cov),
        gate_index=i,
    )


# ==========================================================================
# 切圈
# ==========================================================================
def _project_to_centerline(
    px: np.ndarray, py: np.ndarray, cx: np.ndarray, cy: np.ndarray, s_center: np.ndarray
) -> np.ndarray:
    """
    把一串轨迹点投影到参考中心线上，返回每个点对应的中心线弧长。

    纯"最近点投影"在赛道自我靠近的地方（发夹弯的两条直道、并行的维修区）
    会跳到另一支线上。所以加了一个**连续性惩罚**：匹配点离上一个匹配点越远，
    代价越高。相邻采样点本来就只差一两米，而跳支会差上百米，很容易区分。
    """
    m = cx.size
    out = np.empty(px.size, dtype=np.float64)
    idx_axis = np.arange(m, dtype=np.float64)
    ds = float(s_center[-1]) / max(m - 1, 1)
    prev = 0
    for k in range(px.size):
        d2 = (px[k] - cx) ** 2 + (py[k] - cy) ** 2
        pen = (np.abs(idx_axis - prev) * ds) ** 2 * 0.1
        j = int(np.argmin(d2 + pen))
        out[k] = s_center[j]
        prev = j
    return out


def _build_grid(tel: telemetry.Telemetry, gate: Gate, crossing_times: np.ndarray, step: float) -> tuple[np.ndarray, list[Lap], float | None]:
    """根据过线时刻把整段遥测切成一个个 Lap，并重采样到统一距离网格上。"""
    t = tel.t

    # 每一圈的 GPS 下标区间
    spans: list[tuple[int, int]] = []
    for k in range(crossing_times.size - 1):
        i0 = int(np.searchsorted(t, crossing_times[k]))
        i1 = int(np.searchsorted(t, crossing_times[k + 1]))
        i0 = max(0, min(i0, t.size - 2))
        i1 = max(i0 + 1, min(i1, t.size - 1))
        spans.append((i0, i1))

    if not spans:
        raise RuntimeError("没有切出任何完整的圈。")

    # 每圈的行驶距离。走线宽窄不同、GPS 噪声不同，各圈里程本来就有几米差异。
    real_lengths = np.array([float(tel.dist[i1] - tel.dist[i0]) for i0, i1 in spans])

    # ------------------------------------------------------------------
    # 第一步：先按"各自里程"粗对齐，取平均走线作为参考中心线
    # ------------------------------------------------------------------
    grid0 = np.arange(0.0, float(np.min(real_lengths)), step)
    if grid0.size < 20:
        raise RuntimeError("单圈距离太短，请检查起终点线设置。")

    cx_list, cy_list = [], []
    for i0, i1 in spans:
        sl = slice(i0, i1 + 1)
        d = tel.dist[sl] - tel.dist[i0]
        cx_list.append(geo.interp_to(d, tel.x[sl], grid0))
        cy_list.append(geo.interp_to(d, tel.y[sl], grid0))
    cx = np.append(np.mean(cx_list, axis=0), np.mean(cx_list, axis=0)[0])
    cy = np.append(np.mean(cy_list, axis=0), np.mean(cy_list, axis=0)[0])
    s_center = geo.cumulative_distance(cx, cy)
    L_center = float(s_center[-1])

    # ------------------------------------------------------------------
    # 第二步：把每一圈投影到参考中心线，改按**中心线弧长**对齐
    #
    # 为什么要多这一道：不同圈的行驶里程能差近十米（走线宽窄 + GPS 噪声），
    # 如果按"各自累计里程"对齐，同一个距离值在不同圈对应的**赛道位置并不相同**，
    # 末端能错开小半个弯 —— 表现就是 delta 曲线在终点附近冒出负值，
    # 看起来"这一圈比最快圈还快"，其实只是拿两个不同位置在比。
    # 投影到同一条中心线之后，每圈覆盖的弧长都是同一个赛道长度，才真正可比。
    # ------------------------------------------------------------------
    arc: list[np.ndarray] = []
    for i0, i1 in spans:
        sl = slice(i0, i1 + 1)
        s = _project_to_centerline(
            np.asarray(tel.x[sl]), np.asarray(tel.y[sl]), cx, cy, s_center
        )
        s = np.mod(s - s[0], L_center)  # 绕回起点归零
        # 解卷绕：相邻差值接近一整圈，说明跨过了 0 点
        for j in np.where(np.diff(s) < -L_center * 0.5)[0]:
            s[j + 1 :] += L_center
        arc.append(np.maximum.accumulate(s))

    # 共同可比长度取各圈实际覆盖弧长里的最小值
    lap_arc = float(np.min([a[-1] for a in arc]))
    grid = np.arange(0.0, lap_arc, step)
    if grid.size < 20:
        raise RuntimeError("单圈弧长太短，请检查起终点线设置。")

    laps: list[Lap] = []
    for k, (i0, i1) in enumerate(spans):
        sl = slice(i0, i1 + 1)
        t_seg = t[sl]
        s = arc[k]

        # 中心线弧长 → 时间 的映射
        t_of_s = geo.interp_to(s, t_seg, grid)
        al, at = tel.imu_g(t_of_s)

        lap = Lap(
            index=k + 1,
            t_start=float(t[i0]),
            t_end=float(t[i1]),
            duration=float(t[i1] - t[i0]),
            start_index=i0,
            end_index=i1,
            length=float(real_lengths[k]),
            grid=grid,
            speed=geo.interp_to(s, tel.speed[sl], grid),
            a_long=al,
            a_lat=at,
            gx=geo.interp_to(s, tel.x[sl], grid),
            gy=geo.interp_to(s, tel.y[sl], grid),
            time=t_of_s - t_of_s[0],
        )

        lap.max_speed = float(np.max(lap.speed))
        lap.min_speed = float(np.min(lap.speed))
        lap.max_lat_g = float(np.max(np.abs(lap.a_lat)))
        lap.max_brake_g = float(abs(np.min(lap.a_long)))
        lap.max_accel_g = float(np.max(lap.a_long))
        # 只统计“纵向 G 为正”的时间占比。不要叫它全油门 —— GoPro 测不到
        # 油门开度，而且纵向 G 里还混着侧滑项 -v·ω·sinβ，与油门无对应关系。
        lap.accel_time_pct = float(np.mean(lap.a_long > 0.0) * 100.0)

        laps.append(lap)

    # 出场圈识别：第 1 圈通常是从维修区起步的，速度明显偏低，
    # 把它算进"稳定性"统计会白白拉大标准差，掩盖真实的驾驶波动。
    if len(laps) >= 3:
        rest = float(np.median([l.duration for l in laps[1:]]))
        if laps[0].duration > rest * 1.12:
            laps[0].valid = False

    out_lap = float(crossing_times[0] - t[0]) if crossing_times[0] > t[0] + 3.0 else None
    return grid, laps, out_lap


def _sector_times(
    tel: telemetry.Telemetry, gate: Gate, crossing_times: np.ndarray, lap_len: float, n_sectors: int
) -> tuple[list[float], list[float]]:
    """
    按固定距离把每圈切成若干分段，返回 (每圈的分段用时列表, 分段边界距离)。

    分段边界取在"距离"上而不是"时间"上 —— 这样不同圈的分段是可比的
    （这一点和真实赛事的 sector 定义一致）。
    """
    bounds = [lap_len * k / n_sectors for k in range(1, n_sectors)]
    t = tel.t

    per_lap: list[list[float]] = []
    for k in range(crossing_times.size - 1):
        i0 = int(np.searchsorted(t, crossing_times[k]))
        i1 = int(np.searchsorted(t, crossing_times[k + 1]))
        i0 = max(0, min(i0, t.size - 2))
        i1 = max(i0 + 1, min(i1, t.size - 1))
        d_seg = tel.dist[i0 : i1 + 1] - tel.dist[i0]
        t_seg = t[i0 : i1 + 1]

        marks = [0.0, *bounds, float(d_seg[-1])]
        times = [float(t_seg[0])]
        for b in marks[1:]:
            bb = min(b, float(d_seg[-1]))
            times.append(float(np.interp(bb, d_seg, t_seg)))
        per_lap.append([times[j + 1] - times[j] for j in range(len(times) - 1)])

    return per_lap, bounds


def compute_lapset(
    tel: telemetry.Telemetry,
    *,
    gate: Gate | None = None,
    gate_latlon: tuple[float, float] | None = None,
    sectors: int = 3,
    grid_step: float = 1.0,
    verbose: bool = True,
) -> LapSet:
    """
    完整的切圈流程：找起终点线 → 切圈 → 算分段 → 重采样到统一网格。
    """
    if gate is None and gate_latlon is not None:
        lat, lon = gate_latlon
        gx, gy, _, _ = geo.to_local_xy(np.array([lat]), np.array([lon]), tel.lat0, tel.lon0)
        # 方向取轨迹上最近点的切向
        j = int(np.argmin(np.hypot(tel.x - gx[0], tel.y - gy[0])))
        lo, hi = max(j - 3, 0), min(j + 3, len(tel.x) - 1)
        d = np.array([tel.x[hi] - tel.x[lo], tel.y[hi] - tel.y[lo]])
        n = np.linalg.norm(d)
        d = d / n if n > 1e-9 else np.array([1.0, 0.0])
        gate = Gate(
            x=float(gx[0]), y=float(gy[0]), lat=lat, lon=lon, direction=d,
            clearance=float("nan"), n_laps=0, lap_cov=float("nan"), score=1.0, gate_index=j,
        )
    elif gate is None:
        gate = find_gate(tel)

    crossing_times = _crossings(tel, gate.x, gate.y, gate.direction)
    if crossing_times.size < 2:
        raise RuntimeError(
            "在选定的起终点线处只检测到不到 2 次过线，无法计算圈速。"
            "请用 --gate 纬,经 手动指定一个赛道上的位置。"
        )

    grid, laps, out_lap = _build_grid(tel, gate, crossing_times, grid_step)
    if gate.clearance is None or not np.isfinite(gate.clearance):
        gate = Gate(
            x=gate.x, y=gate.y, lat=gate.lat, lon=gate.lon, direction=gate.direction,
            clearance=float("nan"), n_laps=len(laps),
            lap_cov=float(np.std([l.duration for l in laps]) / np.mean([l.duration for l in laps])),
            score=1.0, gate_index=gate.gate_index,
        )
    else:
        gate.n_laps = len(laps)

    # 分段边界按网格的实际覆盖长度来定，保证和上面的距离对齐一致
    step = float(grid[1] - grid[0]) if grid.size > 1 else grid_step
    per_lap_sectors, bounds = _sector_times(
        tel, gate, crossing_times, float(grid[-1]) + step, sectors
    )
    for lap, secs in zip(laps, per_lap_sectors):
        lap.sectors = secs

    if verbose:
        print("\n— 圈速分析 —")
        print(gate.describe())

    return LapSet(
        gate=gate,
        laps=laps,
        out_lap_duration=out_lap,
        sector_count=sectors,
        sector_bounds=bounds,
        grid=grid,
        telemetry=tel,
    )


__all__ = [
    "GATE_HALF_WIDTH",
    "Gate",
    "Lap",
    "LapSet",
    "compute_lapset",
    "find_gate",
    "format_delta",
    "format_lap_time",
]
