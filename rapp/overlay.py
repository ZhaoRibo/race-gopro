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

性能（2704×2028 实测的量级）：编一帧约 51 ms，而画一帧 HUD 才约 3 ms ——
**瓶颈是 x264 编码，不是画 HUD**，所以一秒素材大约要 3 秒机器时间。
    · `--overlay-preset veryfast` 约快 40%，画质基本看不出差别
    · `--overlay-range 开始,结束` 只处理你真正关心的那几圈，按比例省
    · 调 `--overlay-fps` **基本不提速**：输出的帧率跟的是源视频，HUD 少刷几次
      并不会让输出少编几帧

HUD 布局（以 1080p 为基准，其它分辨率按高度等比缩放）
    左上：圈号 / 本圈计时 / 与最快圈的差距 / 最快圈
    右上：本圈速度曲线 + 最快圈参考线 + 当前位置游标
    左下：纵向 G 指示条（向上绿 = 加速，向下红 = 刹车）+ G-G 圆盘与横向 G 数值
    右下：当前时速

不做油门/刹车指示：GoPro 测不到油门开度，任何“油门/刹车”都只能是纵向 G
的换算，而卡丁车漂移时纵向 G 里混着 -v·ω·sinβ 这一项（能到 ±0.8 g），
与真实操作无关。详见左下那段注释。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import analysis as ana
from . import imu, laps
from . import subproc

# 配色
_WHITE = (240, 245, 250, 255)
_DIM = (150, 162, 176, 255)
_PURPLE = (168, 85, 247, 255)
_GREEN = (34, 197, 94, 255)
_RED = (239, 68, 68, 255)
_CYAN = (34, 211, 238, 255)
_AMBER = (245, 158, 11, 255)
_PANEL = (8, 12, 18, 150)

# 【必须优先选含中日韩字形的字体】
# HUD 里有“最快圈”“纵向 G”这类中文标签。Arial / Helvetica 这些纯拉丁字体
# **不含 CJK 字形**，PIL 也不会自动回退，中文会被画成空白或豆腐块 ——
# 实测同一串“最快圈”：Arial 只有 618 个墨点，Hiragino 有 3873。
# 拉丁字母交给它们渲染也很干净，而大号数字走下面的等宽字体，不受影响。
#
# ⚠ 顺序和平台都很要紧：候选里**一个都加载不了**的时候，PIL 会退回到内置的
# 位图字体（固定 11 px、不能缩放），画到 4K 画面上就是“字小到几乎看不见”。
# v0.1.4 的 Windows 包正是这么翻车的 —— 当时这张表里全是 mac / Linux 路径。
# 所以现在：① 按平台真实路径找（Windows 的系统盘未必是 C:，读 WINDIR）；
# ② 找不到就扫字体目录按文件名猜；③ 再不行也要退到**可缩放**的字体。
# 按平台分开列，**本平台的必须排最前**。
# 为什么强调顺序：macOS 上 `~/Library/Fonts` 里可能恰好躺着一个叫 simhei.ttf 的
# 字体（用户自己装的），要是把 Windows 那批名字排在前面，mac 上就会去用那个 ——
# 而 HUD 的版面是按 Hiragino 的度量调过的，换字体会让版面变样。
_SANS_WINDOWS = [
    "msyhbd.ttc", "msyh.ttc",              # 微软雅黑（粗 / 常规）
    "simhei.ttf",                           # 黑体
    "msjhbd.ttc", "msjh.ttc",               # 微软正黑（繁体，同样含汉字）
    "simsunb.ttf", "simsun.ttc",            # 宋体
    "msyi.ttf",                             # 等线
    "YuGothB.ttc", "YugothB.ttc",           # 日文哥特体（汉字同源，能用）
    "arialbd.ttf", "arial.ttf",             # 纯拉丁：中文会变方块，但至少能读
]
_SANS_MACOS = [
    "Hiragino Sans GB.ttc",
    "STHeiti Medium.ttc",
    "PingFang.ttc",
    "Supplemental/Songti.ttc",
    "Supplemental/Arial Bold.ttf",
    "Helvetica.ttc",
    "SFNS.ttf",
]
_SANS_LINUX = [
    "opentype/noto/NotoSansCJK-Bold.ttc",
    "truetype/wqy/wqy-zenhei.ttc",
    "truetype/dejavu/DejaVuSans-Bold.ttf",
]
# 大号数字走等宽，计时器跳动时才不会左右抖 —— 所以这几个必须找得到
_MONO_WINDOWS = [
    "consolab.ttf", "consola.ttf",          # Consolas（Windows 自带）
    "lucon.ttf",                             # Lucida Console
    "courbd.ttf", "cour.ttf",                # Courier New
]
_MONO_MACOS = [
    "SFNSMono.ttf",
    "Menlo.ttc",
    "Supplemental/Courier New Bold.ttf",
]
_MONO_LINUX = [
    "truetype/dejavu/DejaVuSansMono-Bold.ttf",
]

_SANS_BY_PLATFORM = {"nt": _SANS_WINDOWS, "darwin": _SANS_MACOS, "linux": _SANS_LINUX}
_MONO_BY_PLATFORM = {"nt": _MONO_WINDOWS, "darwin": _MONO_MACOS, "linux": _MONO_LINUX}

_BUNDLED_SANS = ["NotoSansSC-Regular.otf"]
_BUNDLED_MONO = ["NotoSansMono-Regular.ttf"]
"""构建时打进包里的字体（由 packaging/build.py 下载），**排在最后当保底**。

为什么不排在前面：平台自带的那几个字更好看 —— macOS 的 Hiragino、Windows 的
微软雅黑都是各自系统里调好的字体，HUD 的版面当初也是按它们的度量调的。
自带这份的意义是“**一定兜得住**”：系统里一个字体都找不到时（v0.1.4 的 Windows
现场就是这样）不至于退到那个 11 px 的位图字体。
代价：各平台的字不完全一样。真想完全一致，把这两个列表挪到最前面就行。
"""


def _platform_key() -> str:
    if os.name == "nt":
        return "nt"
    return "darwin" if sys.platform == "darwin" else "linux"


def _ordered(by_platform: dict[str, list[str]]) -> list[str]:
    """本平台的候选排最前，其他平台跟在后面兜底（路径不存在会自动跳过）。"""
    key = _platform_key()
    out = list(by_platform[key])
    for other, names in by_platform.items():
        if other != key:
            out += names
    return out


_FONT_NAMES = _ordered(_SANS_BY_PLATFORM) + _BUNDLED_SANS
_MONO_NAMES = _ordered(_MONO_BY_PLATFORM) + _BUNDLED_MONO
# 扫目录兜底时，按文件名猜的关键字（按优先级）
_SANS_HINTS = ("msyh", "yahei", "simhei", "heiti", "notosanscjk", "sourcehan",
               "pingfang", "hiragino", "arial", "dejavu")
_MONO_HINTS = ("consol", "lucon", "cour", "mono", "menlo", "sfns")


def _bundled_font_dir() -> Path | None:
    """包里自带的那份字体在哪。"""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        d = Path(meipass) / "fonts"           # PyInstaller 解到 _MEIPASS/fonts
        if d.is_dir():
            return d
    # 源码模式：直接用构建时下好的那份，让本地和打出来的包行为一致
    d = Path(__file__).resolve().parent.parent / "packaging" / "fonts"
    return d if d.is_dir() else None


def _font_dirs() -> list[Path]:
    """找字体的目录。

    包内自带的那份也一并列进来（它排在最后当保底）；Windows 的系统盘未必是 C:，
    所以读环境变量。
    """
    dirs: list[Path] = []
    if os.name == "nt":
        dirs.append(Path(os.environ.get("WINDIR") or r"C:\Windows") / "Fonts")
        local = os.environ.get("LOCALAPPDATA")
        if local:                      # 用户自己装的字体放这儿，也要找
            dirs.append(Path(local) / "Microsoft" / "Windows" / "Fonts")
    else:
        dirs += [
            Path("/System/Library/Fonts"),
            Path("/System/Library/Fonts/Supplemental"),
            Path("/Library/Fonts"),
            Path.home() / "Library/Fonts",
            Path("/usr/share/fonts"),
            Path("/usr/local/share/fonts"),
        ]
    bundled = _bundled_font_dir()
    if bundled:
        dirs.append(bundled)           # 保底，排在系统目录后面
    return [d for d in dirs if d.is_dir()]


def _loadable(path: Path) -> bool:
    """能不能真的加载出来。

    只判断文件存在还不够：损坏的文件或打不开的 ttc 会让 PIL 报错，
    从而静默退化成那个又小又丑的位图字体。先试加载一次再认定。
    """
    try:
        ImageFont.truetype(str(path), 32)
    except OSError:
        return False
    return True


def _find_font(names: list[str], hints: tuple[str, ...]) -> str | None:
    """先按写死的名字找，再扫字体目录按关键字猜。都不行返回 None。"""
    dirs = _font_dirs()
    for name in names:
        for d in dirs:
            p = d / name
            if p.exists() and _loadable(p):
                return str(p)

    files: list[Path] = []
    for d in dirs:
        files += list(d.rglob("*.tt[cf]"))
    for hint in hints:
        for p in files:
            if hint in p.name.lower().replace(" ", "") and _loadable(p):
                return str(p)
    return None


def _default_font(size: int):
    """
    一个字体都没找着时的兜底。

    关键：**必须传 size**。不传的话 PIL 给的是固定 11 px 的内置位图字体，
    画在 4K 画面上等于看不见（v0.1.4 的 Windows 包就是这么翻车的）。
    Pillow ≥ 10.1 的 `load_default(size=...)` 返回的是可缩放的字体，
    中文仍会缺字形，但至少字号是对的。
    """
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                  # 老版本 Pillow 没有 size 参数
        return ImageFont.load_default()


class _Fonts:
    """按字号缓存字体对象（PIL 每次新建 ImageFont 都有开销）。"""

    def __init__(self) -> None:
        self._sans = _find_font(_FONT_NAMES, _SANS_HINTS)
        self._mono = _find_font(_MONO_NAMES, _MONO_HINTS) or self._sans
        self._cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}
        # 这一行是排 Windows 上“字特别小”那类问题的第一手线索，别删
        print(f"  HUD 字体：正文 {Path(self._sans).name if self._sans else '（兜底）'}"
              f" ／ 数字 {Path(self._mono).name if self._mono else '（兜底）'}"
              f"（共 {len(_font_dirs())} 个字体目录）", flush=True)
        if not self._sans or not self._mono:
            print("  ⚠ 没找到可缩放的字体，HUD 文字会退到兜底字体"
                  "（中文可能显示成方块）。装一个中文字体会好很多："
                  "Windows 装「微软雅黑」、Linux 装 Noto Sans CJK。", flush=True)

    def get(self, size: int, mono: bool = False):
        key = ("mono" if mono else "sans", max(8, int(size)))
        if key not in self._cache:
            path = self._mono if mono else self._sans
            try:
                self._cache[key] = (ImageFont.truetype(path, key[1]) if path
                                    else _default_font(key[1]))
            except OSError:
                self._cache[key] = _default_font(key[1])
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
    out = subproc.run(cmd, capture_output=True, text=True, check=True).stdout
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
def _g_limits(sa: ana.SessionAnalysis) -> tuple[float, float]:
    """
    整场（而不是当前 --overlay-range 片段）的横向 / 纵向 G 刻度。

    用整场极值而不是片段极值，是为了让 HUD 的刻度在整个视频里保持一致 ——
    否则换一个 --overlay-range，同一个 G 值会跳到圆盘的不同位置，没法对比。

    极值本身取**按时间平滑 1 秒**后的信号（见 `imu.PEAK_SMOOTH_S`）：裸信号的单点
    极值会被怠速振动之类的尖峰撑大，把整个圆盘的刻度压扁 —— 刻度一压扁，正常
    驾驶动作在盘上都只占中心一小块，反而看不出差别。
    这样定刻度之后，偶发的瞬时尖峰会超出量程，圆盘会把点**钉在圆周上**（见
    `render_hud_frame`），旁边的数字读数仍然是真实瞬时值。
    """
    tel = sa.lapset.telemetry
    if tel is None or tel.a_long is None or tel.a_lat is None or tel.t_imu is None:
        return 1.5, 1.0
    lat = float(np.nanmax(np.abs(imu.peak_g(tel.a_lat, tel.t_imu))))
    lon = float(np.nanmax(np.abs(imu.peak_g(tel.a_long, tel.t_imu))))
    return max(1.0, lat), max(0.5, lon)


def render_hud_frame(
    k: int,
    hud: dict[str, np.ndarray],
    sa: ana.SessionAnalysis,
    W: int,
    H: int,
    fonts: _Fonts,
    *,
    show_trace: bool = True,
    g_lat_max: float = 1.5,
    g_long_max: float = 1.0,
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
    # 顶部两个卡片的公共边距
    # ------------------------------------------------------------------
    # 四个距离取同一个值 M：计时卡到上边界 / 到左边界，曲线卡到上边界 / 到右边界。
    # 中间那道空当也取 M，整行就是均匀的节奏：
    #
    #     M + 566（计时卡） + M + 760（曲线卡） + M = 画面宽
    #
    # 所以 M 不是随便定的 —— 卡片大小不动的活，它只能从宽度里解出来：
    # 2.7K（2704 宽）下解出 M ≈ 38（1080p 基准），到 px(60) 两张卡就撞上了。
    # 右下时速和左下纵向 G 条也对着这个 M，画面四边的外沿连成一条线。
    M = px(38)

    # ------------------------------------------------------------------
    # 左上：圈号 / 本圈计时 / 与最快圈的差距
    # ------------------------------------------------------------------
    # 高度 264 是算出来的，不是拍脑袋：内容是 LAP 行 / 大字号计时 / 差值 /
    # “最快圈”四行，最后一行必须离下沿留出与顶部相当的空白 —— 原来是 238，
    # 底部那行的墨迹下沿正好顶到框底（留白≈0），看上去头轻脚重。
    panel_w, panel_h = px(566), px(264)
    base.alpha_composite(_rounded_panel((panel_w, panel_h), px(18), _PANEL), (M, M))

    x0, y0 = M + px(28), M + px(20)
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
        # 按**墨迹**定位，不写死像素偏移：中文字体的 ascent / descent 各不相同
        # （Hiragino 和微软雅黑的下沿就不一样），写死偏移换个字体就会贴到框底。
        # 目标是“字的下沿离面板下沿 = px(26)”，与顶部 LAP 那行的上留白相当。
        txt = f"最快圈  {laps.format_lap_time(best.duration)}"
        f_best = fonts.get(px(26))
        ink = d.textbbox((0, 0), txt, font=f_best)
        d.text((x0, M + panel_h - px(26) - ink[3]), txt, font=f_best, fill=_DIM)

    # ------------------------------------------------------------------
    # 右下：时速
    # ------------------------------------------------------------------
    # 用右对齐 + 底对齐（anchor="rb"）：数字位数变化（99 → 100）时往左长，
    # 右边不会被挤得跳来跳去。
    spd = float(hud["speed"][k])
    if np.isfinite(spd):
        d.text((W - M, H - px(150)), f"{spd * 3.6:.0f}",
               font=fonts.get(px(172), mono=True), fill=_WHITE, anchor="rb")
        d.text((W - M, H - px(62)), "km/h",
               font=fonts.get(px(34)), fill=_DIM, anchor="rb")

    # ------------------------------------------------------------------
    # 下方两个仪表的公共几何
    # ------------------------------------------------------------------
    # 纵向 G 条和 G-G 圆盘要看起来像一套：条高 = 圆盘直径，上下也和圆盘对齐；
    # 两个标签共用同一条基线，两个数值也统一字号。改这里两边一起变。
    r = px(96)
    # 圆盘连同它上面那组「横向」读数整块左移，靠近左边的纵向 G 读数。
    # 它不能随便选：横向那一列宽约 px(126)，要让它和纵向读数的右缘之间正好
    # 留一个 M 的空当，圆心就得落在 px(265) —— 再往右两组读数又离得远了。
    cx, cy = px(265), H - px(196)
    bar_w = px(60)
    bar_h = 2 * r                              # 与圆盘直径等高
    bx = M                                     # 和顶部卡片的左边线对齐
    by = cy - r                                # 条的上下与圆盘对齐
    mid = by + bar_h / 2
    half = bar_h / 2
    lab_sans = fonts.get(px(30))               # 标签字号（两处一致）
    lab_num = fonts.get(px(30), mono=True)     # 数值字号（两处一致）
    lab_y = cy - r - px(74)                    # 两个标签共用这条基线

    # ------------------------------------------------------------------
    # 左下：纵向 G 指示条
    # ------------------------------------------------------------------
    # 【这里原本是“油门 / 刹车”，为什么会失真】
    # GoPro 根本测不到油门和刹车开度。原先那两条 T/B 指示条是从纵向 G
    # 硬换算出来的，而车体系纵向加速度里混着一项 -v·ω·sinβ（卡丁车漂移时
    # 能到 ±0.8 g），与油门刹车毫无关系 —— 弯中明明在加油，读数却在乱跳。
    # 换成直接显示纵向 G 本身：这是真实测到的量，不会骗人。
    d.text((bx, lab_y), "纵向G", font=lab_sans, fill=_DIM, anchor="ls")
    # 数值落在「纵向G」和指示条之间的空当里，字号跟横向那个数值一致
    d.text((bx, lab_y + px(46)),
           f"{a_long:+.2f} g" if np.isfinite(a_long) else "-- g",
           font=lab_num,
           fill=(_GREEN if a_long >= 0 else _RED) if np.isfinite(a_long) else _DIM,
           anchor="ls")

    d.rounded_rectangle([bx, by, bx + bar_w, by + bar_h], radius=px(10),
                        fill=(255, 255, 255, 34))
    d.line([bx - px(8), mid, bx + bar_w + px(8), mid],
           fill=(255, 255, 255, 120), width=max(1, int(1.4 * s)))
    if np.isfinite(a_long):
        # 向上 = 加速（绿），向下 = 刹车（红），与圆盘纵轴方向一致
        frac = float(np.clip(a_long / g_long_max, -1.0, 1.0))
        hgt = abs(frac) * half
        if hgt > 1.0:
            top = mid - hgt if a_long >= 0 else mid
            d.rounded_rectangle([bx, top, bx + bar_w, top + hgt],
                                radius=px(10), fill=_GREEN if a_long >= 0 else _RED)

    # ------------------------------------------------------------------
    # 中下：G-G 圆盘（局部超采样，保证圆和点平滑）
    # ------------------------------------------------------------------
    # r / cx / cy / lab_* 都在上面的「公共几何」里定好了，跟纵向 G 那条共用
    #
    # 圆盘用两个方向的**共同**量程：摩擦圆的形状要求两轴等标尺，
    # 这是它存在的意义（看得出“刹车 + 转向”的合力能不能吃满抓地力）。
    g_max = max(g_lat_max, g_long_max)
    g_scale = r / (g_max * 1.15)

    dot_x = cx + (a_lat if np.isfinite(a_lat) else 0.0) * g_scale
    dot_y = cy - (a_long if np.isfinite(a_long) else 0.0) * g_scale

    # 刻度是按"平滑 1 秒的整场峰值"定的（见 _g_limits），所以偶发的瞬时尖峰
    # （振动、路面冲击）会超出量程。把点**钉在圆周上**，而不是让它飞到画面外面去。
    _dx, _dy = dot_x - cx, dot_y - cy
    _rr = float(np.hypot(_dx, _dy))
    if _rr > r > 0:
        dot_x, dot_y = cx + _dx / _rr * r, cy + _dy / _rr * r

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
    d.text((cx, cy + r + px(16)), "G-G", font=fonts.get(px(26), mono=True), fill=_DIM, anchor="ma")
    # 把横向数值明确写在圆盘上方 —— 只打一个数字很容易被误当成“总 G”，
    # 而纵向那一半藏着不看就丢了（用户反馈的正是这个问题）。
    # 「横向」用中文字体、数字用等宽字体分开画：等宽字体不含中文字形，
    # 混在一串里会把中文变成豆腐块；而数字用等宽才能在跳动时不左右飘。
    y_lab = lab_y
    # 和纵向 G 一个排法：标签在上、数值在下，两行都居中对着圆盘。
    # 位置不在这里管 —— 圆盘跟着这一块一起动，只要改上面的 cx 就行。
    d.text((cx, y_lab), "横向", font=lab_sans, fill=_DIM, anchor="ms")
    d.text((cx, y_lab + px(46)),
           f"{a_lat:+.2f} g" if np.isfinite(a_lat) else "-- g",
           font=lab_num, fill=_CYAN if np.isfinite(a_lat) else _DIM, anchor="ms")

    # ------------------------------------------------------------------
    # 右上：本圈速度曲线 + 最快圈参考 + 当前位置游标
    # ------------------------------------------------------------------
    if show_trace and best is not None and best.speed is not None and 1 <= lap_no <= len(ls.laps):
        lap = ls.laps[lap_no - 1]
        box_w, box_h = px(760), px(196)
        tb_x, tb_y = W - M - box_w, M
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
class Cancelled(RuntimeError):
    """HUD 生成被中途取消（半成品已经删掉）。"""


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
    progress: Callable[[float], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> Path:
    """
    把 HUD 烧进视频。

    progress
        每写完一批帧回调一次，参数是 0~1 的完成比例。
        本地服务（serve.py）用它驱动网页上的进度条。
    should_stop
        返回 True 就中断，并把已经写了半截的文件删掉。
    """
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
    g_lat_max, g_long_max = _g_limits(sa)

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

    proc = subproc.popen(cmd, stdin=subprocess.PIPE, stderr=None)
    assert proc.stdin is not None
    # 每 2% 回调一次就够了；每帧都回调反而会让进度条一直在抖
    step = max(1, n_frames // 50)
    cancelled = False
    try:
        for k in range(n_frames):
            if should_stop is not None and should_stop():
                cancelled = True
                break
            frame = render_hud_frame(k, hud, sa, W, H, fonts, show_trace=show_trace,
                                     g_lat_max=g_lat_max, g_long_max=g_long_max)
            proc.stdin.write(frame.tobytes())
            if progress is not None and k % step == 0:
                progress(k / n_frames)
            if verbose and k % max(1, n_frames // 10) == 0:
                pct = k / n_frames * 100
                sys.stderr.write(f"\r  渲染进度 {pct:5.1f}%")
                sys.stderr.flush()
        if cancelled:
            proc.kill()
    except BrokenPipeError:
        pass
    finally:
        try:
            if proc.stdin:
                proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        ret = proc.wait()

    if cancelled:
        # 删掉半截文件：留着的话用户下次点会以为生成成功了
        out_path.unlink(missing_ok=True)
        raise Cancelled("HUD 生成已取消。")

    if progress is not None:
        progress(1.0)
    if verbose:
        sys.stderr.write("\r  渲染进度 100.0%\n")

    if ret != 0:
        raise RuntimeError(f"ffmpeg 编码失败（退出码 {ret}）。")
    return out_path


__all__ = ["Cancelled", "burn", "render_hud_frame", "video_info"]
