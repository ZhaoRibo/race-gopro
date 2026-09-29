"""
输出层：把 HUD 叠加到原视频上
==========================

做法
----
1. 用 Pillow 逐帧画一张**带透明通道**的 HUD 图层（RGBA）；
2. 把 RGBA 原始像素通过管道实时喂给 ffmpeg；
3. ffmpeg 用 `overlay` 滤镜把 HUD 合成到原视频上，重新编码。

为什么不用 ASS 字幕：ASS 画圆形仪表和动态折线要靠手写贝塞尔路径，很难维护；
Pillow 可以直接调用 FreeType 渲染系统字体，写出带抗锯齿的真字体文字，
而且布局调整起来直观得多。

性能：1080p 下单帧约 8~15 ms，20 分钟的视频（30fps）大约跑 5~8 分钟。
    · 用 `--overlay-fps 12` 可以把 HUD 压到 12 帧/秒，速度再快一倍多
      （计时器看起来仍然连续，因为小数点后三位本来也看不清）。
    · 用 `--overlay-range 开始,结束` 只处理你真正关心的那几圈。

HUD 布局（以 1080p 为基准，其它分辨率按高度等比缩放）
    左上：圈号 / 本圈计时 / 与最快圈的差距 / 最快圈
    右上：当前速度
    左下：油门—刹车指示条
    中下：G-G 圆盘（一个点表示当前横向+纵向 G）
    右下：本圈速度曲线 + 最快圈参考线 + 当前位置游标
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import analysis as ana
from . import laps

# 配色
_WHITE = (240, 245, 250, 255)
_DIM = (150, 162, 176, 255)
_PURPLE = (168, 85, 247, 255)
_GREEN = (34, 197, 94, 255)
_RED = (239, 68, 68, 255)
_CYAN = (34, 211, 238, 255)
_AMBER = (245, 158, 11, 255)
_PANEL = (8, 12, 18, 150)

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
_MONO_CANDIDATES = [
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
]


def _find_font(candidates: list[str]) -> str | None:
    for c in candidates:
        if Path(c).exists():
            return c
    return None


class _Fonts:
    """按字号缓存字体对象（PIL 每次新建 ImageFont 都有开销）。"""

    def __init__(self) -> None:
        self._sans = _find_font(_FONT_CANDIDATES)
        self._mono = _find_font(_MONO_CANDIDATES) or self._sans
        self._cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}

    def get(self, size: int, mono: bool = False):
        key = ("mono" if mono else "sans", max(8, int(size)))
        if key not in self._cache:
            path = self._mono if mono else self._sans
            try:
                self._cache[key] = ImageFont.truetype(path, key[1]) if path else ImageFont.load_default()
            except OSError:
                self._cache[key] = ImageFont.load_default()
        return self._cache[key]


# ==========================================================================
def video_info(path: str | Path) -> dict:
    """读取视频的分辨率、帧率、时长。"""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("找不到 ffprobe，请先安装 FFmpeg：brew install ffmpeg")
    cmd = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration",
        "-of", "json", str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    st = json.loads(out)["streams"][0]
    num, den = (st.get("r_frame_rate") or "30/1").split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    return {
        "width": int(st["width"]),
        "height": int(st["height"]),
        "fps": fps,
        "duration": float(st.get("duration") or 0.0),
    }


# ==========================================================================
def _hud_table(
    sa: ana.SessionAnalysis, fps: float, t0: float, t1: float
) -> dict[str, np.ndarray]:
    """
    预先把每一帧要显示的量算好（全部向量化，避免在绘图循环里做插值）。

    返回的每个数组长度都是帧数，第 k 个元素对应 t0 + k/fps 这一帧。
    """
    ls = sa.lapset
    tel = ls.telemetry
    best = ls.best_lap
    n = max(1, int(np.ceil((t1 - t0) * fps)))
    t = t0 + np.arange(n, dtype=np.float64) / fps

    # 每帧落在哪一圈（用圈的结束时刻做二分查找）
    ends = np.array([l.t_end for l in ls.laps])
    starts = np.array([l.t_start for l in ls.laps])
    lap_slot = np.searchsorted(ends, t, side="right")  # 可能等于 len(laps) 表示在最后一圈之后
    valid = (lap_slot < len(ls.laps)) & (t >= starts[np.clip(lap_slot, 0, len(ls.laps) - 1)])

    out: dict[str, np.ndarray] = {
        "t": t,
        "lap_index": np.where(valid, lap_slot + 1, 0),
        "lap_elapsed": np.full(n, np.nan),
        "speed": np.full(n, np.nan),
        "a_long": np.full(n, np.nan),
        "a_lat": np.full(n, np.nan),
        "dist": np.full(n, np.nan),
        "delta": np.full(n, np.nan),
    }

    # 全场速度与 G 值（按时间查表）
    spd_at = lambda tt: np.interp(tt, tel.t, tel.speed)  # noqa: E731
    al, at = tel.imu_g(tel.t)
    al_long_at = lambda tt: np.interp(tt, tel.t, al)  # noqa: E731
    al_lat_at = lambda tt: np.interp(tt, tel.t, at)  # noqa: E731

    for k, lap in enumerate(ls.laps):
        m = valid & (lap_slot == k)
        if not m.any():
            continue
        tt = t[m]
        out["lap_elapsed"][m] = tt - lap.t_start
        out["speed"][m] = spd_at(tt)
        out["a_long"][m] = al_long_at(tt)
        out["a_lat"][m] = al_lat_at(tt)
        # 本圈内已经跑过的距离：时间 → 距离
        d = np.interp(tt, lap.t_start + lap.time, lap.grid)
        out["dist"][m] = d
        # 与最快圈的差距（同样距离处）
        if best is not None and lap is not best:
            out["delta"][m] = np.interp(tt - lap.t_start, best.time, lap.time - best.time)

    return out


# ==========================================================================
def _rounded_panel(size: tuple[int, int], radius: int, fill: tuple[int, int, int, int]) -> Image.Image:
    """生成一块半透明圆角面板（单独一张图，之后用 alpha_composite 叠加）。"""
    base = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(base).rounded_rectangle([0, 0, size[0] - 1, size[1] - 1], radius=radius, fill=fill)
    return base


def _smooth_layer(size: tuple[int, int], render, ss: int = 3) -> Image.Image:
    """
    把圆、弧线、折线这类**需要抗锯齿**的图形画在一张**局部小图**上，再缩回来。

    PIL 的 ImageDraw 画圆和线是硬边的，直接画会有明显锯齿；
    超采样是标准解法。但关键是**只对需要平滑的那一小块区域超采样** ——
    如果对整帧做，1080p × 3 倍超采样等于每帧处理 4K 以上像素，
    20 分钟的视频要跑二十多分钟。局部化之后开销可以忽略。
    """
    w, h = size
    if w <= 0 or h <= 0:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    big = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
    render(ImageDraw.Draw(big), ss)
    return big.resize((w, h), Image.LANCZOS)


def _fmt_delta(v: float) -> str:
    return " --.---" if not np.isfinite(v) else f"{v:+.3f}"


def _fmt_time(v: float) -> str:
    if not np.isfinite(v):
        return "0:00.000"
    m = int(v // 60)
    s = v - m * 60
    return f"{m}:{s:06.3f}" if m else f"{s:06.3f}"


# ==========================================================================
def render_hud_frame(
    k: int,
    hud: dict[str, np.ndarray],
    sa: ana.SessionAnalysis,
    W: int,
    H: int,
    fonts: _Fonts,
    *,
    show_trace: bool = True,
) -> Image.Image:
    """画某一帧的 HUD，返回 RGBA 图。"""
    s = H / 1080.0  # 所有尺寸都按这个比例缩放，4K 和 720p 共用同一套布局
    px = lambda v: int(round(v * s))  # noqa: E731

    ls = sa.lapset
    best = ls.best_lap
    base = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(base)

    lap_no = int(hud["lap_index"][k])
    total = len(ls.laps)
    a_lat = float(hud["a_lat"][k])
    a_long = float(hud["a_long"][k])

    # ------------------------------------------------------------------
    # 左上：圈号 / 本圈计时 / 与最快圈的差距
    # ------------------------------------------------------------------
    panel_w, panel_h = px(566), px(238)
    base.alpha_composite(_rounded_panel((panel_w, panel_h), px(18), _PANEL), (px(32), px(32)))

    x0, y0 = px(60), px(52)
    if lap_no > 0:
        d.text((x0, y0), "LAP", font=fonts.get(px(30), mono=True), fill=_DIM)
        d.text((x0 + px(90), y0), f"{lap_no} / {total}", font=fonts.get(px(30), mono=True), fill=_WHITE)
        d.text((x0, y0 + px(40)), _fmt_time(hud["lap_elapsed"][k]),
               font=fonts.get(px(84), mono=True), fill=_WHITE)

        if best is not None and lap_no == best.index:
            d.text((x0, y0 + px(142)), "BEST LAP", font=fonts.get(px(38)), fill=_PURPLE)
        else:
            delta = float(hud["delta"][k])
            d.text((x0, y0 + px(138)), _fmt_delta(delta),
                   font=fonts.get(px(46), mono=True),
                   fill=_GREEN if delta < 0 else _RED)
    else:
        d.text((x0, y0 + px(30)), "OUT LAP", font=fonts.get(px(58), mono=True), fill=_AMBER)

    if best is not None:
        d.text((x0, y0 + panel_h - px(52)), f"最快圈  {laps.format_lap_time(best.duration)}",
               font=fonts.get(px(26)), fill=_DIM)

    # ------------------------------------------------------------------
    # 右上：速度
    # ------------------------------------------------------------------
    spd = float(hud["speed"][k])
    if np.isfinite(spd):
        d.text((W - px(240), px(30)), f"{spd * 3.6:.0f}",
               font=fonts.get(px(172), mono=True), fill=_WHITE, anchor="ra")
        d.text((W - px(48), px(198)), "km/h",
               font=fonts.get(px(34)), fill=_DIM, anchor="ra")

    # ------------------------------------------------------------------
    # 左下：油门 / 刹车指示条（由纵向 G 换算）
    # ------------------------------------------------------------------
    bar_w, bar_gap = px(58), px(16)
    bx, by = px(60), H - px(300)
    bar_h = px(212)
    d.text((bx, by - px(36)), "油门 / 刹车", font=fonts.get(px(24)), fill=_DIM)

    finite_long = hud["a_long"][np.isfinite(hud["a_long"])]
    acc_max = max(0.15, float(np.max(finite_long)) if finite_long.size else 0.3)
    brk_max = max(0.30, -float(np.min(finite_long)) if finite_long.size else 0.8)
    thr_frac = float(np.clip(a_long / acc_max, 0.0, 1.0)) if np.isfinite(a_long) else 0.0
    brk_frac = float(np.clip(-a_long / brk_max, 0.0, 1.0)) if np.isfinite(a_long) else 0.0

    for i, (frac, col, label) in enumerate(((thr_frac, _GREEN, "T"), (brk_frac, _RED, "B"))):
        x = bx + i * (bar_w + bar_gap)
        d.rounded_rectangle([x, by, x + bar_w, by + bar_h], radius=px(10), fill=(255, 255, 255, 34))
        if frac > 0.005:
            filled = int(bar_h * frac)
            d.rounded_rectangle([x, by + bar_h - filled, x + bar_w, by + bar_h],
                                radius=px(10), fill=col)
        d.text((x + bar_w / 2, by + bar_h + px(10)), label,
               font=fonts.get(px(26), mono=True), fill=_DIM, anchor="ma")

    # ------------------------------------------------------------------
    # 中下：G-G 圆盘（局部超采样，保证圆和点平滑）
    # ------------------------------------------------------------------
    r = px(96)
    cx, cy = px(430), H - px(196)
    finite_lat = hud["a_lat"][np.isfinite(hud["a_lat"])]
    g_max = 1.5
    if finite_lat.size:
        g_max = max(1.0, float(np.max(np.abs(finite_lat))))
    if finite_long.size:
        g_max = max(g_max, float(np.max(np.abs(finite_long))))
    g_scale = r / (g_max * 1.15)

    dot_x = cx + (a_lat if np.isfinite(a_lat) else 0.0) * g_scale
    dot_y = cy - (a_long if np.isfinite(a_long) else 0.0) * g_scale

    pad = px(8)
    gbox = (cx - r - pad, cy - r - pad, cx + r + pad, cy + r + pad)
    gw, gh = gbox[2] - gbox[0], gbox[3] - gbox[1]
    lw = max(1, int(1.4 * s))

    def _meter(dr: ImageDraw.ImageDraw, ss: int):
        R = r * ss
        cxx, cyy = (cx - gbox[0]) * ss, (cy - gbox[1]) * ss
        dr.ellipse([cxx - R, cyy - R, cxx + R, cyy + R], fill=(255, 255, 255, 30))
        for frac in (0.5, 1.0):
            rr = R * frac
            dr.ellipse([cxx - rr, cyy - rr, cxx + rr, cyy + rr],
                       outline=(255, 255, 255, 70), width=lw * ss)
        dr.line([cxx - R, cyy, cxx + R, cyy], fill=(255, 255, 255, 60), width=lw * ss)
        dr.line([cxx, cyy - R, cxx, cyy + R], fill=(255, 255, 255, 60), width=lw * ss)
        rd = px(11) * ss
        dx_, dy_ = (dot_x - gbox[0]) * ss, (dot_y - gbox[1]) * ss
        dr.ellipse([dx_ - rd, dy_ - rd, dx_ + rd, dy_ + rd], fill=_CYAN)

    base.alpha_composite(_smooth_layer((gw, gh), _meter, ss=3), (gbox[0], gbox[1]))

    d = ImageDraw.Draw(base)
    d.text((cx, cy + r + px(16)), "G", font=fonts.get(px(28), mono=True), fill=_DIM, anchor="ma")
    d.text((cx, cy - r - px(46)),
           f"{abs(a_lat):.2f} g" if np.isfinite(a_lat) else "-- g",
           font=fonts.get(px(28), mono=True), fill=_CYAN if np.isfinite(a_lat) else _DIM, anchor="ma")

    # ------------------------------------------------------------------
    # 右下：本圈速度曲线 + 最快圈参考 + 当前位置游标
    # ------------------------------------------------------------------
    if show_trace and best is not None and best.speed is not None and 1 <= lap_no <= len(ls.laps):
        lap = ls.laps[lap_no - 1]
        box_w, box_h = px(760), px(196)
        tb_x, tb_y = W - box_w - px(48), H - box_h - px(56)
        base.alpha_composite(_rounded_panel((box_w, box_h), px(14), _PANEL), (tb_x, tb_y))

        grid, bs = best.grid, best.speed
        vmax = max(float(np.max(bs)), float(np.max(lap.speed))) * 1.08
        vmin = 0.0
        span_d = float(grid[-1]) if grid[-1] > 0 else 1.0

        def to_local(dd, vv):
            return ((dd / span_d) * box_w, box_h - ((vv - vmin) / (vmax - vmin)) * box_h)

        cur_d = float(hud["dist"][k])
        n_cur = max(1, min(int(np.searchsorted(lap.grid, cur_d)), lap.grid.size - 1))
        ref_step = max(1, grid.size // 260)
        cur_step = max(1, lap.grid.size // 260)

        def _trace(dr: ImageDraw.ImageDraw, ss: int):
            sc = lambda p: (p[0] * ss, p[1] * ss)  # noqa: E731 — 局部坐标 → 超采样坐标
            pts = [sc(to_local(grid[i], bs[i])) for i in range(0, grid.size, ref_step)]
            if len(pts) >= 2:
                dr.line(pts, fill=(255, 255, 255, 60), width=max(1, px(2) * ss), joint="curve")
            cpts = [sc(to_local(lap.grid[i], lap.speed[i])) for i in range(0, n_cur, cur_step)]
            if len(cpts) >= 2:
                dr.line(cpts, fill=_CYAN, width=max(2, px(3) * ss), joint="curve")

        base.alpha_composite(_smooth_layer((box_w, box_h), _trace, ss=2), (tb_x, tb_y))

        d = ImageDraw.Draw(base)
        cxp, cyp = to_local(cur_d, float(hud["speed"][k]) if np.isfinite(hud["speed"][k]) else 0.0)
        d.line([tb_x + cxp, tb_y, tb_x + cxp, tb_y + box_h], fill=_PURPLE, width=px(2))
        d.ellipse([tb_x + cxp - px(6), tb_y + cyp - px(6),
                   tb_x + cxp + px(6), tb_y + cyp + px(6)], fill=_WHITE)
        d.text((tb_x + px(14), tb_y + px(8)), "本圈速度 / 最快圈参考",
               font=fonts.get(px(22)), fill=_DIM)

    return base


# ==========================================================================
def burn(
    video: str | Path,
    sa: ana.SessionAnalysis,
    out_path: str | Path,
    *,
    fps: float = 15.0,
    t_range: tuple[float, float] | None = None,
    crf: int = 20,
    preset: str = "medium",
    show_trace: bool = True,
    verbose: bool = True,
) -> Path:
    """把 HUD 烧进视频。"""
    video = Path(video)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    info = video_info(video)
    W, H = info["width"], info["height"]
    t0, t1 = (0.0, info["duration"]) if t_range is None else t_range
    t1 = min(t1, info["duration"])
    if t1 <= t0:
        raise ValueError("叠加时间范围无效。")

    tele = sa.lapset.telemetry
    if tele is not None:
        # HUD 的时间轴以"遥测相对录制起点"为准，和视频时间轴一致
        t1 = min(t1, tele.duration)

    hud = _hud_table(sa, fps, t0, t1)
    n_frames = hud["t"].size
    fonts = _Fonts()

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("找不到 ffmpeg，请先安装：brew install ffmpeg")

    # 用 -ss 从 t0 开始取源视频，这样 HUD 的时间轴和切出来的片段对齐
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-stats", "-y",
        "-ss", f"{t0:.3f}", "-t", f"{t1 - t0:.3f}", "-i", str(video),
        "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "-",
        "-filter_complex",
        f"[0:v]scale={W}:{H}:force_original_aspect_ratio=decrease,"
        f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2[v0];[v0][1:v]overlay=0:0:format=auto,format=yuv420p[out]",
        "-map", "[out]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-c:a", "copy", "-shortest",
        str(out_path),
    ]

    if verbose:
        print(f"\n开始生成 HUD 叠加视频：{W}×{H} @ {fps} fps，共 {n_frames} 帧")
        print("（进度条由 ffmpeg 输出，中途可以按 Ctrl+C 中断）")

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=None)
    assert proc.stdin is not None
    try:
        for k in range(n_frames):
            frame = render_hud_frame(k, hud, sa, W, H, fonts, show_trace=show_trace)
            proc.stdin.write(frame.tobytes())
            if verbose and k % max(1, n_frames // 10) == 0:
                pct = k / n_frames * 100
                sys.stderr.write(f"\r  渲染进度 {pct:5.1f}%")
                sys.stderr.flush()
    except BrokenPipeError:
        pass
    finally:
        if proc.stdin:
            proc.stdin.close()
        ret = proc.wait()
    if verbose:
        sys.stderr.write("\r  渲染进度 100.0%\n")

    if ret != 0:
        raise RuntimeError(f"ffmpeg 编码失败（退出码 {ret}）。")
    return out_path


__all__ = ["burn", "render_hud_frame", "video_info"]
