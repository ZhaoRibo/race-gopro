"""
弯道分析与专业指标
================

这一层回答教练真正关心的问题：

    · 每个弯我最快能以多少速度过？（顶点速度）
    · 我的刹车点在每一圈到底是不是同一个位置？（刹车点一致性）
    · 我横向最多压出了几个 G？距离轮胎极限还有多少？（摩擦圆）
    · 我比自己的最快圈，在哪个弯开始落后？（delta 曲线）
    · 我的"理论最佳圈"是多少？（各分段最好成绩之和）

三个核心概念
------------
1. **顶点速度（apex speed）** —— 弯道里速度最低的那一点。卡丁车没有悬挂和
   下压力，弯速几乎完全由轮胎摩擦圆决定，所以顶点速度是衡量"敢不敢开"的最直接指标。

2. **摩擦圆 / G-G 图** —— 把每个时刻的 (横向 G, 纵向 G) 画成散点，
   数据点会围出一个近似椭圆的区域，那就是这条轮胎的抓地力包线。
   点到包线的距离，就是"还剩多少余量"。

3. **Delta 曲线** —— 以最快圈为基准，逐点算时间差。
   这是唯一能同时看清"在哪里快、在哪里慢"的工具：
   曲线往上升 = 正在丢时间，往下降 = 正在追回来。

刹车点的检测方式
---------------
在弯道入口前的一段距离内（默认 80 m），从后往前找第一个"纵向 G ≤ −0.12 且
持续至少 3 个采样点"的位置。用"持续"这个条件是为了滤掉路面颠簸造成的瞬时尖峰。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import laps
from .geo import G0

BRAKE_THRESHOLD_G = -0.12
"""判定为"正在刹车"的纵向 G 阈值。"""

BRAKE_MIN_RUN = 3
"""必须连续这么多个采样点都在刹车，才算数。"""

CORNER_G_THRESHOLD = 0.25
"""横向 G 超过这个值才算进入弯道。"""

CORNER_MIN_LENGTH = 8.0
"""弯道的最小长度（米），滤掉路面起伏造成的假信号。"""

CORNER_MAX_GAP = 6.0
"""两段横向 G 之间的间隔小于这个值就合并。

设小一点很重要：卡丁车场地很紧凑，实测一条 800 米、官方标 16 个弯的赛道，
两个相邻弯之间可能只隔十几米；阈值给到 12 米会把连续弯全部并成一个。
"""


@dataclass
class Corner:
    """一个弯道。"""

    index: int
    direction: int
    """+1 = 左弯，−1 = 右弯。"""

    d_start: float
    d_end: float
    d_apex: float
    """起点线起算的距离，米。"""

    length: float

    radius: float
    """等效半径（米），由 a = v²/R 反推。半径越小弯越急。"""

    # ---- 每圈指标，shape (n_laps,) ----
    apex_speed: np.ndarray = field(default_factory=lambda: np.zeros(0))
    entry_speed: np.ndarray = field(default_factory=lambda: np.zeros(0))
    exit_speed: np.ndarray = field(default_factory=lambda: np.zeros(0))
    max_lat_g: np.ndarray = field(default_factory=lambda: np.zeros(0))
    brake_point: np.ndarray = field(default_factory=lambda: np.zeros(0))
    """刹车开始的绝对距离（米，起点线起算）。NaN 表示这一圈没在这个弯刹车。"""

    valid_mask: np.ndarray | None = None
    """哪些圈计入统计（排除出场圈之类的非正常圈）。与上面各数组一一对应。"""

    def _valid(self, arr: np.ndarray) -> np.ndarray:
        """取出只属于有效圈的那部分。"""
        if self.valid_mask is None:
            return arr
        return arr[self.valid_mask]

    def valid_apex_speed(self) -> np.ndarray:
        """只含有效圈的顶点速度。"""
        return self._valid(self.apex_speed)

    @property
    def name(self) -> str:
        return f"弯{self.index}{'L' if self.direction > 0 else 'R'}"

    def best_lap_position(self) -> int:
        """顶点速度最高的那一圈在 laps 数组里的**下标**（0 起）。"""
        v = self.apex_speed
        if v.size == 0:
            return 0
        idx = np.where(self.valid_mask)[0] if self.valid_mask is not None else np.arange(v.size)
        if idx.size == 0:
            return 0
        vals = np.where(np.isfinite(v[idx]), v[idx], -np.inf)
        return int(idx[int(np.argmax(vals))])

    def best_lap_number(self) -> int:
        """哪一圈在这个弯的顶点速度最高。"""
        return self.best_lap_position() + 1

    def apex_spread(self) -> float:
        """顶点速度的离散程度，m/s。数值大说明这个弯跑得还不稳定。"""
        pool = self.valid_apex_speed()
        pool = pool[np.isfinite(pool)]
        return float(np.std(pool)) if pool.size >= 2 else float("nan")

    def brake_spread(self) -> float:
        """刹车点的离散程度，米。数值大说明刹车点不稳定。"""
        bp = self._valid(self.brake_point)
        bp = bp[np.isfinite(bp)]
        return float(np.std(bp)) if bp.size >= 2 else float("nan")

    def is_decisive(self) -> bool:
        """顶点速度在圈与圈之间差异大（>1 m/s），说明这是一个决定成绩的弯。"""
        s = self.apex_spread()
        return np.isfinite(s) and s > 1.0


@dataclass
class SessionAnalysis:
    """一次练习的完整分析结果。"""

    lapset: laps.LapSet
    corners: list[Corner]
    deltas: list[np.ndarray]
    """每一圈相对最快圈的累计时间差（秒），与 lapset.grid 一一对应。正值 = 落后。"""

    ref_lap: laps.Lap | None

    rolling_best: float
    """连续分段最好成绩之和（比"分段各自最好"更现实的极限）。"""

    def to_dict(self) -> dict:
        """导出成 JSON 友好的结构。"""
        ls = self.lapset
        t = ls.telemetry
        return {
            "source": t.source if t else None,
            "duration_s": round(t.duration, 2) if t else None,
            "gps_rate_hz": round(t.gps_rate, 2) if t else None,
            "distance_m": round(float(t.dist[-1]), 1) if t else None,
            "mount_calibration": (
                {
                    "quality_r": round(t.cal.quality, 3),
                    "gravity": round(t.cal.gravity_mag, 3),
                    "yaw_deg": round(float(np.degrees(t.cal.yaw)), 2),
                    "lateral_positive": "left" if t.cal.lateral_sign > 0 else "right",
                }
                if t and t.cal
                else None
            ),
            "gate": {
                "lat": ls.gate.lat,
                "lon": ls.gate.lon,
                "clearance_m": None if not np.isfinite(ls.gate.clearance) else round(ls.gate.clearance, 1),
            },
            "out_lap_s": None if ls.out_lap_duration is None else round(ls.out_lap_duration, 3),
            "best_lap_s": round(ls.best_lap.duration, 3) if ls.best_lap else None,
            "best_lap_number": ls.best_lap.index if ls.best_lap else None,
            "mean_lap_s": round(ls.mean_lap, 3) if np.isfinite(ls.mean_lap) else None,
            "std_lap_s": round(ls.std_lap, 3) if np.isfinite(ls.std_lap) else None,
            "theoretical_best_s": round(ls.theoretical_best, 3) if np.isfinite(ls.theoretical_best) else None,
            "rolling_best_s": round(self.rolling_best, 3) if np.isfinite(self.rolling_best) else None,
            "sector_count": ls.sector_count,
            "best_sectors_s": [None if not np.isfinite(v) else round(v, 3) for v in ls.best_sectors],
            "laps": [
                {
                    "index": l.index,
                    "valid": l.valid,
                    "duration_s": round(l.duration, 3),
                    "time": laps.format_lap_time(l.duration),
                    "sectors_s": [round(s, 3) for s in l.sectors],
                    "length_m": round(l.length, 1),
                    "max_speed_kmh": round(l.max_speed * 3.6, 1),
                    "min_speed_kmh": round(l.min_speed * 3.6, 1),
                    "max_lat_g": round(l.max_lat_g, 2),
                    "max_brake_g": round(l.max_brake_g, 2),
                    "max_accel_g": round(l.max_accel_g, 2),
                    "accel_time_pct": round(l.accel_time_pct, 1),
                }
                for l in ls.laps
            ],
            "corners": [
                {
                    "index": c.index,
                    "direction": "L" if c.direction > 0 else "R",
                    "d_start_m": round(c.d_start, 1),
                    "d_apex_m": round(c.d_apex, 1),
                    "d_end_m": round(c.d_end, 1),
                    "radius_m": round(c.radius, 1),
                    "apex_speed_kmh": [round(v * 3.6, 1) for v in c.apex_speed],
                        "apex_speed_spread_kmh": round(c.apex_spread() * 3.6, 2)
                        if np.isfinite(c.apex_spread())
                        else None,
                    "max_lat_g": [round(float(v), 2) for v in c.max_lat_g],
                    "brake_point_m": [None if not np.isfinite(v) else round(float(v), 1) for v in c.brake_point],
                    "brake_spread_m": None if not np.isfinite(c.brake_spread()) else round(c.brake_spread(), 1),
                }
                for c in self.corners
            ],
        }


# ==========================================================================
CORNER_MIN_SEPARATION = 16.0
"""两个弯的顶点至少隔这么远（米），比这更近的峰会被合并。

卡丁车场地很紧凑：实测一条 820 米、官方标了 16 个弯的赛道，
平均每 51 米就有一个弯，连续弯之间只隔二三十米。
阈值给大了会把连续弯并成一个，给小了会把同一个弯拆成两个。
"""


def _detect_corner_spans(
    grid: np.ndarray, ref: np.ndarray, g_peak: float, min_sep_m: float, min_len_m: float
) -> list[tuple[int, int, int]]:
    """
    用**横向 G 峰值**而不是"超过阈值的区间"来划分弯道。

    为什么不能按阈值切：卡丁车弯道密集，车几乎全程都在转向，
    |a_lat| 大于阀值的区间会连成一大片 —— 实测这条赛道按阈值只切出 5~6 个
    "大弯"，而横向 G 信号里实际有 30 多个峰、官方赛道图标了 16 个弯。

    做法：先找显著峰（带最小间隔约束），再以"降到峰值 45% 以下"为边界
    向两侧扩展，最后合并重叠区间。
    """
    from scipy.signal import find_peaks

    step = float(grid[1] - grid[0]) if grid.size > 1 else 1.0
    peaks, _ = find_peaks(
        np.abs(ref), height=g_peak, distance=max(1, int(min_sep_m / step))
    )

    spans: list[tuple[int, int, int]] = []
    for p in peaks:
        h = float(abs(ref[p]))
        thr = max(g_peak * 0.5, h * 0.45)
        a = int(p)
        while a > 0 and abs(ref[a - 1]) > thr and (grid[p] - grid[a - 1]) < 150.0:
            a -= 1
        b = int(p)
        while b < ref.size - 1 and abs(ref[b + 1]) > thr and (grid[b + 1] - grid[p]) < 150.0:
            b += 1
        spans.append((a, b, int(p)))

    # 合并重叠的区间（连续弯会被两次扩展覆盖到）
    merged: list[tuple[int, int, int]] = []
    for a, b, p in spans:
        if merged and a <= merged[-1][1]:
            pa, pb, pp = merged[-1]
            if abs(ref[p]) > abs(ref[pp]):
                pp = p
            merged[-1] = (pa, max(pb, b), pp)
        else:
            merged.append((a, b, p))

    return [(a, b, p) for a, b, p in merged if (b - a) * step >= min_len_m]


def _brake_point(lap: laps.Lap, d_corner_start: float, search_back: float) -> float:
    """
    在弯道入口前 search_back 米的范围内，找刹车开始的位置（绝对距离，米）。
    找不到（比如全油门通过的弯）返回 NaN。
    """
    grid = lap.grid
    lo = max(0, int(np.searchsorted(grid, d_corner_start - search_back)))
    hi = int(np.searchsorted(grid, d_corner_start))
    if hi - lo < BRAKE_MIN_RUN + 1:
        return float("nan")

    a = lap.a_long[lo:hi]
    below = a <= BRAKE_THRESHOLD_G
    # 找第一个"从这里开始连续 BRAKE_MIN_RUN 个点都在刹车"的位置
    for k in range(below.size - BRAKE_MIN_RUN):
        if below[k : k + BRAKE_MIN_RUN].all():
            return float(grid[lo + k])
    return float("nan")


def detect_corners(
    lapset: laps.LapSet,
    *,
    g_threshold: float = CORNER_G_THRESHOLD,
    min_length: float = CORNER_MIN_LENGTH,
    max_gap: float = CORNER_MAX_GAP,
    search_back: float = 80.0,
) -> list[Corner]:
    """
    从横向 G 曲线里识别弯道。

    用"所有圈的中位数"作为参考曲线，而不是用某一圈 ——
    这样个别圈的异常（比如被前车挡了一下）不会影响弯道的划分，
    保证所有圈的分段是可比的。
    """
    if not lapset.laps:
        return []
    grid = lapset.grid

    stack = np.vstack([l.a_lat for l in lapset.laps if l.a_lat is not None])
    if stack.size == 0:
        return []
    # 取中位数：个别圈的异常（比如被前车挡了一下）不会影响弯道的划分
    ref = np.median(stack, axis=0)

    groups = _detect_corner_spans(
        grid, ref, g_threshold, CORNER_MIN_SEPARATION, min_length
    )

    corners: list[Corner] = []
    speeds = [l.speed for l in lapset.laps]
    latgs = [l.a_lat for l in lapset.laps]
    n = len(lapset.laps)
    valid_mask = np.array([l.valid for l in lapset.laps], dtype=bool)

    for ci, (a, b, apex) in enumerate(groups, start=1):
        seg = ref[a : b + 1]
        direction = 1 if float(np.sum(seg)) >= 0 else -1
        d_apex = float(grid[apex])
        d_start = float(grid[a])
        d_end = float(grid[b])

        corner = Corner(
            index=ci,
            direction=direction,
            d_start=d_start,
            d_end=d_end,
            d_apex=d_apex,
            length=d_end - d_start,
            radius=float("nan"),
            apex_speed=np.full(n, np.nan),
            entry_speed=np.full(n, np.nan),
            exit_speed=np.full(n, np.nan),
            max_lat_g=np.full(n, np.nan),
            brake_point=np.full(n, np.nan),
            valid_mask=valid_mask,
        )

        radii = []
        for k in range(n):
            sp = speeds[k][a : b + 1]
            lg = latgs[k][a : b + 1]
            if sp is None or lg is None:
                continue
            corner.entry_speed[k] = float(sp[0])
            corner.exit_speed[k] = float(sp[-1])
            corner.apex_speed[k] = float(np.min(sp))
            corner.max_lat_g[k] = float(np.max(np.abs(lg)))

            # 由 a = v²/R 反推半径，取顶点处
            ia = int(np.argmin(sp))
            v_apex, g_apex = float(sp[ia]), abs(float(lg[ia]))
            if g_apex > 0.05 and v_apex > 1.0:
                radii.append(v_apex * v_apex / (g_apex * G0))

            corner.brake_point[k] = _brake_point(lapset.laps[k], d_start, search_back)

        corner.radius = float(np.median(radii)) if radii else float("nan")
        corners.append(corner)

    return corners


def delta_vs_best(lapset: laps.LapSet, lap: laps.Lap | None = None) -> np.ndarray:
    """
    逐距离点的时间差：某圈用时 − 最快圈用时，与 lapset.grid 一一对应。
    正数表示在这一点上比最快圈慢了多少秒。

    所有圈都已经重采样到同一个距离网格，所以直接相减即可 ——
    这就是电视转播里那个上下跳动的 "+/−0.342" 数字的来源。
    """
    ref = lapset.best_lap
    lap = lap or ref
    if ref is None or lap is None or ref.time is None or lap.time is None:
        return np.zeros_like(lapset.grid)
    return lap.time - ref.time


def all_deltas(lapset: laps.LapSet) -> list[np.ndarray]:
    """每一圈相对最快圈的时间差，顺序与 lapset.laps 一致。"""
    return [delta_vs_best(lapset, l) for l in lapset.laps]


def rolling_best_sectors(lapset: laps.LapSet) -> float:
    """
    连续分段最佳：在整场练习的分段序列里，找连续 sectors 个分段用时之和的最小值。

    它比"各分段各自最好之和"更现实 —— 后者可能来自不同圈、互相矛盾的跑法，
    实际上跑不出来。这个指标是职业车队用来估算"真实极限"的做法。
    """
    seq: list[float] = []
    for l in lapset.laps:
        seq.extend(l.sectors)
    n = lapset.sector_count
    if len(seq) < n:
        return float("nan")
    if len(seq) == n:
        return float(np.sum(seq))
    sums = [sum(seq[i : i + n]) for i in range(len(seq) - n + 1)]
    return float(min(sums))


def gg_points(lapset: laps.LapSet) -> tuple[np.ndarray, np.ndarray]:
    """G-G 图的所有散点（横向 G, 纵向 G）。"""
    if not lapset.laps:
        return np.zeros(0), np.zeros(0)
    lat = np.concatenate([l.a_lat for l in lapset.laps])
    lon = np.concatenate([l.a_long for l in lapset.laps])
    return lat, lon


def friction_envelope(lat: np.ndarray, lon: np.ndarray, bins: int = 36) -> tuple[np.ndarray, np.ndarray]:
    """
    提取摩擦圆包线：把圆周方向分成若干扇区，每个扇区取半径最大的那个点。
    连起来就是这条轮胎的抓地力外沿。
    """
    if lat.size == 0:
        return np.zeros(0), np.zeros(0)
    ang = np.arctan2(lon, lat)
    rad = np.hypot(lat, lon)
    edges = np.linspace(-np.pi, np.pi, bins + 1)
    out_ang, out_rad = [], []
    for i in range(bins):
        m = (ang >= edges[i]) & (ang < edges[i + 1])
        if not m.any():
            continue
        out_ang.append(0.5 * (edges[i] + edges[i + 1]))
        out_rad.append(float(np.percentile(rad[m], 99.0)))  # 用 99 分位而不是最大值，抗噪
    if not out_ang:
        return np.zeros(0), np.zeros(0)
    # 首尾相接，画图时闭合
    out_ang.append(out_ang[0])
    out_rad.append(out_rad[0])
    return np.array(out_ang), np.array(out_rad)


def analyze(lapset: laps.LapSet, *, verbose: bool = True) -> SessionAnalysis:
    """跑完所有分析，返回一个可直接喂给报表/图表/看板的对象。"""
    corners = detect_corners(lapset)
    deltas = all_deltas(lapset)
    rb = rolling_best_sectors(lapset)

    if verbose and corners:
        decisive = [c for c in corners if c.is_decisive()]
        print("\n— 弯道识别 —")
        print(f"共识别出 {len(corners)} 个弯，其中 {len(decisive)} 个是圈速的关键弯"
              "（顶点速度波动 > 1 m/s）")

    return SessionAnalysis(
        lapset=lapset,
        corners=corners,
        deltas=deltas,
        ref_lap=lapset.best_lap,
        rolling_best=rb,
    )


__all__ = [
    "Corner",
    "SessionAnalysis",
    "all_deltas",
    "analyze",
    "delta_vs_best",
    "detect_corners",
    "friction_envelope",
    "gg_points",
    "rolling_best_sectors",
]
