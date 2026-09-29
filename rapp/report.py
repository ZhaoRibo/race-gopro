"""
输出层：终端报表 + CSV / JSON 导出
==============================

终端表格用中文字段，所以需要处理**东亚字符宽度**：一个汉字显示占两格，
如果用 len() 去对齐，表格会歪。下面的 `_pad` 用 unicodedata.east_asian_width
判断字符实际占位，这是 Python 里处理中文对齐的标准做法。
"""

from __future__ import annotations

import csv
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np

from . import analysis as ana
from . import laps

# 终端高亮色（非 TTY 时自动关闭）
_COLOR = sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    if not _COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def _purple(s: str) -> str:
    """最快圈 / 最好分段 —— 紫罗兰色，赛车圈用来标记"全场最快"。"""
    return _c(s, "38;5;135")


def _bold(s: str) -> str:
    return _c(s, "1")


# ==========================================================================
# 中文对齐
# ==========================================================================
def _display_width(s: str) -> int:
    """字符串的实际显示宽度（汉字算 2）。"""
    w = 0
    for ch in s:
        w += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return w


def _pad(s: str, width: int, align: str = "left") -> str:
    """按显示宽度补空格。"""
    s = str(s)
    gap = max(0, width - _display_width(s))
    if align == "right":
        return " " * gap + s
    if align == "center":
        left = gap // 2
        return " " * left + s + " " * (gap - left)
    return s + " " * gap


def _table(headers: list[str], rows: list[list[str]], aligns: list[str] | None = None) -> str:
    """把一个二维表渲染成对齐的文本。"""
    aligns = aligns or (["left"] + ["right"] * (len(headers) - 1))
    widths = [_display_width(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], _display_width(cell))

    sep = "─┼─".join("─" * w for w in widths)
    head = " │ ".join(_pad(h, widths[i], aligns[i]) for i, h in enumerate(headers))
    body = [
        " │ ".join(_pad(cell, widths[i], aligns[i]) for i, cell in enumerate(r))
        for r in rows
    ]
    return "\n".join([head, sep, *body])


# ==========================================================================
# 终端报表
# ==========================================================================
def print_lap_table(ls: laps.LapSet, rolling_best: float = float("nan")) -> None:
    """圈速表：每一圈的用时、与最快圈的差距、三个分段、极值。"""
    if not ls.laps:
        return

    best = ls.best_lap
    best_sectors = ls.best_sectors
    # 列名里的"峰值"是实打实的：这几个数都按时间平滑过 1 秒（见 imu.PEAK_SMOOTH_S），
    # 不是裸信号的单点极值。
    # 「速度增量」= 一圈内所有正 Δv 之和，是唯一有精确物理答案的驾驶强度指标，
    # 比 G 峰值可靠，也比 G 峰值更能看出马力差异。
    headers = ["圈号", "圈时", "Δ最快", "极速", "峰值横G", "峰值刹G", "峰值加速",
               "速度增量", "加速占比"]
    headers = headers[: 2] + [f"分段{i + 1}" for i in range(ls.sector_count)] + headers[2:]

    rows: list[list[str]] = []
    for lap in ls.laps:
        delta = lap.duration - best.duration if best else float("nan")
        sector_cells = []
        for k in range(ls.sector_count):
            v = lap.sectors[k] if k < len(lap.sectors) else float("nan")
            cell = f"{v:.3f}"
            if best_sectors and k < len(best_sectors) and np.isfinite(best_sectors[k]) and abs(v - best_sectors[k]) < 1e-6:
                cell = _purple(cell)
            sector_cells.append(cell)
        tag = f"#{lap.index}" + ("" if lap.valid else " *")
        rows.append(
            [
                tag,
                laps.format_lap_time(lap.duration),
                "—" if lap is best else laps.format_delta(delta),
                *sector_cells,
                f"{lap.max_speed * 3.6:.1f}",
                f"{lap.peak_lat_g:.2f}",
                f"{lap.peak_brake_g:.2f}",
                f"{lap.peak_accel_g:.2f}",
                f"{lap.speed_gain_ms:.0f}",
                f"{lap.accel_time_pct:.0f}%",
            ]
        )

    print()
    print(_bold("═" * 78))
    print(_bold(" 圈速报表"))
    print(_bold("═" * 78))
    print(_table(headers, rows))

    print()
    print(f"  最快圈        : {_purple(laps.format_lap_time(ls.best_lap.duration))}"
          f"  (第 {ls.best_lap.index} 圈)")
    print(f"  平均圈        : {laps.format_lap_time(ls.mean_lap)}")
    print(f"  标准差        : {ls.std_lap:.3f} s    稳定性: {ls.consistency()}")
    if ls.out_lap_duration:
        print(f"  出场圈        : {ls.out_lap_duration:.3f} s（起点线之前，不计入统计）")
    excluded = [l.index for l in ls.laps if not l.valid]
    if excluded:
        print(f"  未计入统计的圈: {', '.join('#' + str(i) for i in excluded)}"
              f"（标有 * 的圈，通常是出场圈）")
    print()
    print(f"  理论最佳圈    : {_purple(laps.format_lap_time(ls.theoretical_best))}"
          f"   ← 各分段历史最好之和（实战中跑不出来）")
    if np.isfinite(rolling_best):
        print(f"  连续分段最佳  : {_purple(laps.format_lap_time(rolling_best))}"
              f"   ← 连续分段之和的最小值（真实可达极限）")


def print_corner_table(sa: ana.SessionAnalysis) -> None:
    """弯道表：每个弯的顶点速度、半径、刹车点一致性。"""
    if not sa.corners:
        return
    ls = sa.lapset

    headers = ["弯", "方向", "起点", "顶点", "长度", "半径", "顶点速度"]
    headers += [f"第{i + 1}圈" for i in range(len(ls.laps))]
    headers += ["速度波动", "刹车点离散"]

    rows = []
    for c in sa.corners:
        row = [
            f"{c.index}",
            "左" if c.direction > 0 else "右",
            f"{c.d_start:.0f}",
            f"{c.d_apex:.0f}",
            f"{c.length:.0f}",
            f"{c.radius:.0f}" if np.isfinite(c.radius) else "—",
            f"{np.mean(c.valid_apex_speed()) * 3.6:.1f}",
        ]
        best_k = int(np.argmax(c.apex_speed))
        for k, v in enumerate(c.apex_speed):
            cell = f"{v * 3.6:.1f}"
            if k == best_k:
                cell = _purple(cell)
            row.append(cell)
        row.append(f"{c.apex_spread() * 3.6:.1f}" if np.isfinite(c.apex_spread()) else "—")
        bs = c.brake_spread()
        row.append(f"{bs:.1f}" if np.isfinite(bs) else "—")
        rows.append(row)

    print()
    print(_bold("═" * 78))
    print(_bold(" 弯道分析   (速度单位 km/h，距离单位 m)"))
    print(_bold("═" * 78))
    print(_table(headers, rows))

    decisive = [c for c in sa.corners if c.is_decisive()]
    if decisive:
        print()
        print(_bold(" 关键弯（顶点速度波动最大，最值得练的几个弯）："))
        decisive.sort(key=lambda c: -c.apex_spread())
        for c in decisive[:5]:
            spread = c.apex_spread() * 3.6
            lo = float(np.min(c.valid_apex_speed())) * 3.6
            hi = float(np.max(c.valid_apex_speed())) * 3.6
            print(
                f"   {c.name}  顶点速度 {lo:.1f} ~ {hi:.1f} km/h"
                f"（差 {spread:.1f}），最好成绩在第 {c.best_lap_number()} 圈"
            )


def print_report(sa: ana.SessionAnalysis) -> None:
    """完整报表。"""
    ls = sa.lapset
    t = ls.telemetry

    print()
    print(_bold("═" * 78))
    print(_bold(" 遥测概览"))
    print(_bold("═" * 78))
    if t is not None:
        print(t.summary())
        print(f"有效圈数      : {len(ls.laps)}")
        max_lat = max((l.peak_lat_g for l in ls.laps), default=0.0)
        max_brk = max((l.peak_brake_g for l in ls.laps), default=0.0)
        print(f"全场最大横向G : {max_lat:.2f} g")
        print(f"全场最大刹车G : {max_brk:.2f} g")

    print_lap_table(ls, sa.rolling_best)
    print_corner_table(sa)

    # 逐圈 delta 曲线的最小/最大值：指出哪一圈在哪一段丢时间最多
    if sa.deltas and ls.best_lap is not None:
        print()
        print(_bold("═" * 78))
        print(_bold(" 对比最快圈：各圈最大落后点"))
        print(_bold("═" * 78))
        rows = []
        for lap, dl in zip(ls.laps, sa.deltas):
            if lap is ls.best_lap:
                continue
            i = int(np.argmax(dl))
            # 找到该距离处对应的弯
            corner_name = "—"
            for c in sa.corners:
                if c.d_start <= ls.grid[i] <= c.d_end:
                    corner_name = c.name
                    break
            rows.append(
                [
                    f"#{lap.index}",
                    laps.format_delta(lap.duration - ls.best_lap.duration),
                    f"{ls.grid[i]:.0f} m",
                    corner_name,
                    laps.format_delta(float(dl[i])),
                ]
            )
        if rows:
            print(_table(["圈号", "总差距", "落后位置", "所在弯", "该点落后"], rows))


# ==========================================================================
# 导出
# ==========================================================================
def export_csv(sa: ana.SessionAnalysis, outdir: str | Path) -> list[Path]:
    """导出四张 CSV 表，都能直接用 Excel / pandas 打开。"""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ls = sa.lapset
    t = ls.telemetry
    written: list[Path] = []

    # ---- 1. 全场高频遥测（含原始 G 值） ----
    if t is not None:
        p = outdir / "telemetry.csv"
        a_long, a_lat = t.imu_g(t.t)
        with p.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(
                ["time_s", "lat", "lon", "alt_m", "x_m", "y_m", "distance_m",
                 "speed_kmh", "heading_deg", "accel_long_g", "accel_lat_g"]
            )
            for i in range(t.t.size):
                w.writerow([
                    f"{t.t[i]:.4f}", f"{t.lat[i]:.8f}", f"{t.lon[i]:.8f}", f"{t.alt[i]:.2f}",
                    f"{t.x[i]:.2f}", f"{t.y[i]:.2f}", f"{t.dist[i]:.2f}",
                    f"{t.speed[i] * 3.6:.2f}",
                    f"{np.degrees(t.heading[i]) % 360:.2f}",
                    f"{a_long[i]:.4f}", f"{a_lat[i]:.4f}",
                ])
        written.append(p)

    # ---- 2. 每圈汇总 ----
    p = outdir / "laps.csv"
    best = ls.best_lap
    with p.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(
            ["lap", "time_s", "delta_to_best_s"]
            + [f"sector{i + 1}_s" for i in range(ls.sector_count)]
            + ["length_m", "max_speed_kmh", "min_speed_kmh", "avg_speed_kmh",
               "peak_lat_g", "peak_brake_g", "peak_accel_g", "speed_gain_ms",
               "accel_time_pct"]
        )
        for lap in ls.laps:
            w.writerow([
                lap.index, f"{lap.duration:.3f}",
                "" if lap is best else f"{lap.duration - best.duration:.3f}",
                *[f"{s:.3f}" for s in lap.sectors],
                f"{lap.length:.1f}", f"{lap.max_speed * 3.6:.1f}", f"{lap.min_speed * 3.6:.1f}",
                f"{float(np.mean(lap.speed)) * 3.6:.1f}",
                f"{lap.peak_lat_g:.3f}", f"{lap.peak_brake_g:.3f}", f"{lap.peak_accel_g:.3f}",
                f"{lap.speed_gain_ms:.2f}",
                f"{lap.accel_time_pct:.1f}",
            ])
    written.append(p)

    # ---- 3. 距离对齐宽表：每一圈在同一距离处的速度，方便画对比图 ----
    p = outdir / "laps_aligned_speed.csv"
    with p.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["distance_m"] + [f"lap{l.index}_kmh" for l in ls.laps])
        for i, d in enumerate(ls.grid):
            w.writerow([f"{d:.1f}"] + [f"{l.speed[i] * 3.6:.2f}" for l in ls.laps])
    written.append(p)

    p = outdir / "laps_aligned_glat.csv"
    with p.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["distance_m"] + [f"lap{l.index}_lat_g" for l in ls.laps])
        for i, d in enumerate(ls.grid):
            w.writerow([f"{d:.1f}"] + [f"{l.a_lat[i]:.4f}" for l in ls.laps])
    written.append(p)

    # ---- 4. 弯道 × 圈 明细 ----
    p = outdir / "corners.csv"
    with p.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([
            "corner", "direction", "d_start_m", "d_apex_m", "d_end_m", "radius_m",
            "lap", "apex_speed_kmh", "entry_speed_kmh", "exit_speed_kmh",
            "max_lat_g", "brake_point_m",
        ])
        for c in sa.corners:
            for k, lap in enumerate(ls.laps):
                w.writerow([
                    c.index, "L" if c.direction > 0 else "R",
                    f"{c.d_start:.1f}", f"{c.d_apex:.1f}", f"{c.d_end:.1f}",
                    f"{c.radius:.1f}" if np.isfinite(c.radius) else "",
                    lap.index,
                    f"{c.apex_speed[k] * 3.6:.2f}", f"{c.entry_speed[k] * 3.6:.2f}",
                    f"{c.exit_speed[k] * 3.6:.2f}", f"{c.max_lat_g[k]:.3f}",
                    "" if not np.isfinite(c.brake_point[k]) else f"{c.brake_point[k]:.1f}",
                ])
    written.append(p)

    return written


def export_json(sa: ana.SessionAnalysis, path: str | Path) -> Path:
    """导出结构化 JSON，方便喂给别的工具或自建看板。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(sa.to_dict(), f, ensure_ascii=False, indent=2)
    return path


__all__ = ["export_csv", "export_json", "print_corner_table", "print_lap_table", "print_report"]
