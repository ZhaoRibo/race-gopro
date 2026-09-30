"""
输出层：专业图表
==============

生成 6 张图，每张对应一个具体的分析问题：

    lap_times.png        每圈的圈时，一眼看出哪圈最快、稳定度如何
    speed_trace.png      速度—距离曲线，所有圈叠在一起，看走线一致性
    delta.png            相对最快圈的时间差，定位"从哪个弯开始落后"
    gg_diagram.png       G-G 图 / 摩擦圆，看轮胎抓地力用到了几成
    track_map.png        赛道俯视图，按速度着色，标出起点线和各弯
    corner_apex.png      逐弯顶点速度对比，找出最该练的弯

中文字体：macOS 上用 PingFang SC / Heiti TC，这两个系统自带，不用额外装字体。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # 不弹窗，直接出图文件
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

from . import analysis as ana  # noqa: E402
from . import laps  # noqa: E402

_BEST_COLOR = "#7B2FBE"  # 最快圈统一用紫罗兰色
_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
            "#aec7e8", "#ffbb78", "#98df8a", "#ff9896"]


def setup_style() -> None:
    """设置中文字体与整体风格。"""
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti TC", "Songti SC", "Arial Unicode MS",
        "Noto Sans CJK SC", "Microsoft YaHei", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False  # 负号显示成方块的经典问题
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = 160
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["axes.axisbelow"] = True
    plt.rcParams["font.size"] = 10


def _lap_color(k: int, lap: laps.Lap, best: laps.Lap | None) -> str:
    if best is not None and lap is best:
        return _BEST_COLOR
    return _PALETTE[k % len(_PALETTE)]


def _label(lap: laps.Lap, best: laps.Lap | None) -> str:
    tag = f"#{lap.index} {laps.format_lap_time(lap.duration)}"
    return f"{tag} ★" if best is not None and lap is best else tag


def _title_with_excluded(title: str, ls: laps.LapSet) -> str:
    """如果图里去掉了非正常圈，在标题里说明一下，避免看图的人以为数据缺失。"""
    skipped = [l.index for l in ls.laps if not l.valid]
    if not skipped:
        return title
    names = ", ".join(f"#{i}" for i in skipped)
    return f"{title}\n（已排除非正常圈 {names}）"


# ==========================================================================
def plot_lap_times(sa: ana.SessionAnalysis, path: Path) -> Path:
    ls = sa.lapset
    best = ls.best_lap
    if best is None:
        return path

    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = [l.index for l in ls.laps]
    y = [l.duration for l in ls.laps]
    colors = [_lap_color(k, l, best) for k, l in enumerate(ls.laps)]
    bars = ax.bar(x, y, color=colors, alpha=0.85, edgecolor="white", linewidth=0.8)

    # 纵轴范围只按**有效圈**定，否则一圈慢十几秒的出场圈会把其他圈挤成一条线
    vy = [l.duration for l in ls.timed_laps] or y
    lo = min([*vy, ls.theoretical_best, sa.rolling_best]) - 0.6
    hi = max([*vy, ls.theoretical_best, sa.rolling_best]) + 1.4

    for b, l in zip(bars, ls.laps):
        top = b.get_height()
        clipped = top > hi
        ax.text(
            b.get_x() + b.get_width() / 2,
            min(top, hi) + 0.05,
            ("↑ " if clipped else "") + laps.format_lap_time(l.duration),
            ha="center", va="bottom", fontsize=8, clip_on=False,
            color="#666" if clipped else "black",
        )

    ax.axhline(ls.mean_lap, ls="--", color="#444", linewidth=1.1,
               label=f"平均 {laps.format_lap_time(ls.mean_lap)}")
    ax.axhline(ls.theoretical_best, ls=":", color="#c0392b", linewidth=1.4,
               label=f"理论最佳 {laps.format_lap_time(ls.theoretical_best)}")
    if np.isfinite(sa.rolling_best):
        ax.axhline(sa.rolling_best, ls="-.", color="#e67e22", linewidth=1.2,
                   label=f"连续分段最佳 {laps.format_lap_time(sa.rolling_best)}")

    n_skipped = len(ls.laps) - len(ls.timed_laps)
    title = (f"圈速分布   最快 {laps.format_lap_time(best.duration)}（第 {best.index} 圈）"
             f"   标准差 {ls.std_lap:.3f}s")
    if n_skipped:
        title += "\n（灰色 ↑ 标记的圈超出纵轴范围，已排除在统计之外）"
    ax.set_title(title)
    ax.set_xlabel("圈号")
    ax.set_ylabel("圈时 (s)")
    ax.set_xticks(x)
    ax.set_ylim(lo, hi)
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_speed_trace(sa: ana.SessionAnalysis, path: Path) -> Path:
    ls = sa.lapset
    best = ls.best_lap
    fig, ax = plt.subplots(figsize=(12, 4.6))
    # 只画有效圈：出场圈速度低一大截，画进来会把其他圈压成一条带子
    for k, lap in enumerate(ls.laps):
        if not lap.valid:
            continue
        is_best = best is not None and lap is best
        ax.plot(ls.grid, lap.speed * 3.6,
                color=_lap_color(k, lap, best),
                linewidth=2.2 if is_best else 1.0,
                alpha=1.0 if is_best else 0.55,
                zorder=3 if is_best else 2,
                label=_label(lap, best))

    # 弯道区间用阴影标出来，这样看图时能立刻把速度变化对应到具体的弯
    for c in sa.corners:
        ax.axvspan(c.d_start, c.d_end, color="#000000", alpha=0.045, zorder=0)
        ax.text(c.d_apex, ax.get_ylim()[1], f"{c.name}", ha="center", va="top",
                fontsize=7.5, color="#555")

    ax.set_xlabel("距起点线距离 (m)")
    ax.set_ylabel("速度 (km/h)")
    ax.set_title(_title_with_excluded("速度—距离曲线", ls))
    ax.set_xlim(ls.grid[0], ls.grid[-1])
    ax.legend(ncol=4, fontsize=8, loc="lower right")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_delta(sa: ana.SessionAnalysis, path: Path) -> Path:
    ls = sa.lapset
    best = ls.best_lap
    if best is None or not sa.deltas:
        return path

    fig, ax = plt.subplots(figsize=(12, 4.6))
    for k, (lap, dl) in enumerate(zip(ls.laps, sa.deltas)):
        if lap is best or not lap.valid:
            continue
        ax.plot(ls.grid, dl, color=_lap_color(k, lap, best),
                linewidth=1.2, alpha=0.75, label=_label(lap, best))

    ax.axhline(0, color=_BEST_COLOR, linewidth=2.0, alpha=0.9,
               label=f"最快圈 {laps.format_lap_time(best.duration)} (基准)")

    for c in sa.corners:
        ax.axvspan(c.d_start, c.d_end, color="#000000", alpha=0.045, zorder=0)
        ax.text(c.d_apex, ax.get_ylim()[0], f"{c.name}", ha="center", va="bottom",
                fontsize=7.5, color="#555")

    ax.set_xlabel("距起点线距离 (m)")
    ax.set_ylabel("时间差 (s)")
    ax.set_title(_title_with_excluded("相对最快圈的累计时间差（曲线上升 = 正在丢时间，下降 = 正在追回）", ls))
    ax.set_xlim(ls.grid[0], ls.grid[-1])
    ax.legend(ncol=4, fontsize=8)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_gg(sa: ana.SessionAnalysis, path: Path) -> Path:
    lat, lon = ana.gg_points(sa.lapset)
    if lat.size == 0:
        return path

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.scatter(lat, lon, s=2.0, alpha=0.10, color="#1f77b4", edgecolors="none",
               label="全部采样点")

    ang, rad = ana.friction_envelope(lat, lon, bins=48)
    if ang.size:
        ax.plot(rad * np.cos(ang), rad * np.sin(ang), color="#c0392b",
                linewidth=2.0, label="抓地力包线 (99 分位)")

    # 参考圆：0.5g / 1.0g / 1.5g / 2.0g
    theta = np.linspace(0, 2 * np.pi, 361)
    for g in (0.5, 1.0, 1.5, 2.0):
        ax.plot(g * np.cos(theta), g * np.sin(theta), color="#999",
                linewidth=0.6, linestyle=":", zorder=1)
        ax.text(g, 0.02, f"{g:g}g", fontsize=7, color="#888", ha="left", va="bottom")

    ax.axhline(0, color="#666", linewidth=0.7)
    ax.axvline(0, color="#666", linewidth=0.7)
    ax.set_aspect("equal")
    ax.set_xlabel("横向 G   (正值 = 左转)")
    ax.set_ylabel("纵向 G   (正值 = 加速 / 负值 = 刹车)")
    ax.set_title("G-G 图（摩擦圆）\n散点外沿就是这条轮胎的抓地力极限")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_track_map(sa: ana.SessionAnalysis, path: Path) -> Path:
    ls = sa.lapset
    best = ls.best_lap
    if best is None or best.gx is None:
        return path

    # 赛道线取**所有有效圈的平均走线**，而不是单圈的轨迹。
    # GPS 单圈配 1~3m 定位噪声，只画一圈的话赛道看起来又毛糙又变形；
    # 多圈一平均，走线差异和噪声都被抵消掉，出来的才是赛道真正的形状。
    laps_ok = [l for l in ls.laps if l.valid and l.gx is not None]
    if not laps_ok:
        return path
    gx = np.mean([l.gx for l in laps_ok], axis=0)
    gy = np.mean([l.gy for l in laps_ok], axis=0)
    spd = np.mean([l.speed for l in laps_ok], axis=0)

    # 图幅按赛道的真实长宽比来定，配合下面的 set_aspect("equal")：
    # 等比例是硬要求（否则赛道形状会被拉变形），但如果不按数据比例开图幅，
    # matplotlib 会在窄边留下大片空白。
    w = float(np.ptp(gx))
    h = float(np.ptp(gy))
    longest = max(w, h, 1e-6)
    base = 8.4
    figsize = (base * max(w / longest, 0.35), base * max(h / longest, 0.35))
    fig, ax = plt.subplots(figsize=figsize)
    pts = np.column_stack([gx, gy])
    segs = np.concatenate([pts[:-1, None, :], pts[1:, None, :]], axis=1)
    lc = LineCollection(segs, cmap="turbo",
                        norm=Normalize(vmin=float(np.min(spd)) * 3.6,
                                       vmax=float(np.max(spd)) * 3.6))
    lc.set_array(spd[:-1] * 3.6)
    lc.set_linewidth(5.0)
    ax.add_collection(lc)

    # 起点线：一个垂直于行进方向的短横线
    g = ls.gate
    n = np.array([-g.direction[1], g.direction[0]])
    half = 14.0
    ax.plot([g.x - n[0] * half, g.x + n[0] * half],
            [g.y - n[1] * half, g.y + n[1] * half],
            color="black", linewidth=3.0, solid_capstyle="butt", zorder=5)
    ax.annotate("起点线", (g.x, g.y), textcoords="offset points", xytext=(0, 14),
                ha="center", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", alpha=0.85))

    for c in sa.corners:
        i = int(np.searchsorted(ls.grid, c.d_apex))
        i = min(i, best.gx.size - 1)
        ax.annotate(c.name, (best.gx[i], best.gy[i]),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8, color="#2c3e50")

    cb = fig.colorbar(lc, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label("速度 (km/h)")
    ax.set_aspect("equal")
    ax.set_xlabel("东向 (m)")
    ax.set_ylabel("北向 (m)")
    # 标题里必须写清这是"平均走线"：数据是所有有效圈按距离对齐后逐点取平均，
    # 不是某一圈的实测轨迹。写"第 N 圈"会让人以为在看那一圈的真实走线。
    ax.set_title(
        f"赛道俯视图 — 按速度着色（{len(laps_ok)} 个有效圈的平均走线）\n"
        f"最快圈 #{best.index} {laps.format_lap_time(best.duration)}"
    )
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_lap_lines(sa: ana.SessionAnalysis, path: Path) -> Path:
    """
    每圈走线对比图。

    左：把所有有效圈的轨迹叠在一起，一眼看出哪几圈走线不同。
    右：每圈相对**平均走线**的横向偏差（+ 左 / − 右），纵轴是"距起点线多少米"。

    为什么需要右边这一栏：GPS 单点噪声就有 1~3 米，而真实走线差异往往只有
    零点几米到一两米 —— 直接叠加两条线时**噪声会把差异盖住**。右边把相对于
    平均走线的偏移单独抽出来画，差异才看得出来（而且平滑之后噪声只有几十厘米）。
    """
    ls = sa.lapset
    best = ls.best_lap
    if best is None or best.gx is None:
        return path
    ok = [l for l in ls.laps if l.valid and l.gx is not None]
    if not ok:
        return path

    # 每圈的走线先做一次轻平滑。窗口按**距离**取 2.5 米（对应 0.35 秒左右，
    # 见 README 里"轨迹平滑窗口不能大"的说明）—— 再大就会把卡丁车的弯抹圆。
    smooth_m = 2.5
    step = float(ls.grid[1] - ls.grid[0]) if ls.grid.size > 1 else 1.0
    win = max(3, int(round(smooth_m / max(step, 1e-6))))
    if win % 2 == 0:
        win += 1

    def _smooth(a: np.ndarray) -> np.ndarray:
        from scipy.signal import savgol_filter

        w = min(win, a.size - (1 - a.size % 2))
        if w < 3 or w > a.size:
            return a
        return savgol_filter(a, w, 2, mode="interp")

    lines = [(l, _smooth(l.gx), _smooth(l.gy)) for l in ok]
    mean_x = np.mean([g[1] for g in lines], axis=0)
    mean_y = np.mean([g[2] for g in lines], axis=0)

    # 平均走线的切向 → 法向（+ 指向行进方向的左手边）
    tx = np.gradient(mean_x)
    ty = np.gradient(mean_y)
    tn = np.hypot(tx, ty)
    tn = np.where(tn < 1e-9, 1e-9, tn)
    nx, ny = -ty / tn, tx / tn

    # 图幅按赛道真实长宽比开，配合 set_aspect("equal") 才不会把赛道拉变形
    w = float(np.ptp(mean_x))
    h = float(np.ptp(mean_y))
    longest = max(w, h, 1e-6)
    base = 7.2
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(base * 2 + 1.6, base * max(h / longest, 0.45)),
        gridspec_kw={"width_ratios": [max(w / longest, 0.45), 1.0]},
    )

    # ---- 左：叠加走线 ----
    for k, (lap, gx, gy) in enumerate(lines):
        is_best = lap is best
        ax.plot(gx, gy, color=_lap_color(k, lap, best), linewidth=2.6 if is_best else 1.5,
                alpha=1.0 if is_best else 0.75, solid_capstyle="round",
                zorder=6 if is_best else 3, label=_label(lap, best))

    g = ls.gate
    nv = np.array([-g.direction[1], g.direction[0]])
    ax.plot([g.x - nv[0] * 14, g.x + nv[0] * 14],
            [g.y - nv[1] * 14, g.y + nv[1] * 14],
            color="black", linewidth=3.0, solid_capstyle="butt", zorder=7)
    for c in sa.corners:
        i = int(np.searchsorted(ls.grid, c.d_apex))
        i = min(i, mean_x.size - 1)
        ax.annotate(c.name, (mean_x[i], mean_y[i]), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=7.5, color="#2c3e50", zorder=8)
    ax.set_aspect("equal")
    ax.set_xlabel("东向 (m)")
    ax.set_ylabel("北向 (m)")
    ax.set_title("每圈走线叠加", fontsize=11)
    ax.legend(fontsize=7, ncols=2, loc="best", framealpha=0.9)
    ax.grid(alpha=0.15)

    # ---- 右：相对平均走线的横向偏差 ----
    for k, (lap, gx, gy) in enumerate(lines):
        offset = (gx - mean_x) * nx + (gy - mean_y) * ny
        is_best = lap is best
        ax2.plot(offset, ls.grid, color=_lap_color(k, lap, best),
                 linewidth=2.2 if is_best else 1.3,
                 alpha=1.0 if is_best else 0.7, zorder=6 if is_best else 3)
    ax2.axvline(0.0, color="black", linewidth=1.0, alpha=0.6, zorder=4)
    ax2.axhline(0.0, color="black", linewidth=1.0, alpha=0.6, zorder=4)
    for c in sa.corners:
        ax2.axhline(c.d_apex, color="#95a5a6", linewidth=0.6, alpha=0.35, zorder=1)
    ax2.set_xlabel("相对平均走线的横向偏移 (m)　← 右　　左 →")
    ax2.set_ylabel("距起点线 (m)")
    ax2.set_title("每圈走线偏差", fontsize=11)
    ax2.grid(alpha=0.15)
    ax2.invert_yaxis()  # 和赛道图一致：起点在上，往下走

    fig.suptitle(
        f"每圈走线对比（粗线 = 最快圈 #{best.index}；"
        f"GPS 单点噪声 1~3 m，已做 {smooth_m:.0f} m 平滑）",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_corner_apex(sa: ana.SessionAnalysis, path: Path) -> Path:
    ls = sa.lapset
    if not sa.corners or not ls.laps:
        return path
    best = ls.best_lap

    fig, ax = plt.subplots(figsize=(max(8.0, 0.65 * len(sa.corners) + 3), 4.4))
    shown = [k for k, l in enumerate(ls.laps) if l.valid]
    n_laps = max(len(shown), 1)
    width = 0.8 / n_laps
    xs = np.arange(len(sa.corners))

    for slot, k in enumerate(shown):
        lap = ls.laps[k]
        vals = [c.apex_speed[k] * 3.6 for c in sa.corners]
        offset = (slot - (n_laps - 1) / 2) * width
        ax.bar(xs + offset, vals, width=width * 0.92,
               color=_lap_color(k, lap, best), alpha=0.9, label=f"#{lap.index}")

    ax.set_xticks(xs)
    ax.set_xticklabels([c.name for c in sa.corners])
    ax.set_ylabel("顶点速度 (km/h)")
    ax.set_title(_title_with_excluded("逐弯顶点速度对比 — 柱子高度差越大，这个弯开得越不稳定", ls))
    ax.legend(ncol=min(n_laps, 8), fontsize=8)
    fig.savefig(path)
    plt.close(fig)
    return path


# ==========================================================================
def make_all(sa: ana.SessionAnalysis, outdir: str | Path) -> list[Path]:
    """一次生成全部图表。"""
    setup_style()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    jobs = [
        ("lap_times.png", plot_lap_times),
        ("speed_trace.png", plot_speed_trace),
        ("delta.png", plot_delta),
        ("gg_diagram.png", plot_gg),
        ("track_map.png", plot_track_map),
        ("lap_lines.png", plot_lap_lines),
        ("corner_apex.png", plot_corner_apex),
    ]
    out: list[Path] = []
    for name, fn in jobs:
        p = outdir / name
        try:
            fn(sa, p)
            if p.exists():
                out.append(p)
        except Exception as exc:  # noqa: BLE001 - 单张图失败不该影响整体流程
            print(f"  生成 {name} 失败：{exc}")
    return out


__all__ = ["make_all", "setup_style"]
