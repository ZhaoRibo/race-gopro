#!/usr/bin/env python3
"""
应用图标生成器
===============

为什么用代码画、而不是在仓库里放一张图：

* 图标要同时供 macOS（`.icns`）和 Windows（`.ico`）用，两边的格式、尺寸要求
  都不一样；与其在仓库里塞一堆二进制，不如按代码生成，随时能改、能重现。
* 小尺寸（16×16）和大尺寸（1024×1024）用同一套画法缩放出来，比例不会走样。

设计：深色圆角方块 + 一面**方格旗**（旗杆用 HUD 那个琥珀色）。

为什么不用赛道环：第一版画的就是一个带起伏的青色环，结果看起来像**卷尺**
（一圈的形状太中性了）。方格旗是赛车的通用符号，而且格子是高对比度的几何图形，
缩到 16×16 也不会糊。

配色取自 HUD（`overlay.py` 里那几个常量），所以图标和出片看着是一套东西。

命令行：
    python packaging/icon.py          # 生成到 packaging/icon/
    python packaging/icon.py --preview  # 顺带拼一张预览图，方便肉眼过一遍
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
OUT = HERE / "icon"

_BG_TOP = (24, 37, 58)
_BG_BOTTOM = (8, 12, 18)
_CYAN = (34, 211, 238)          # 和 HUD 的速度曲线同色
_AMBER = (245, 158, 11)         # 和 HUD 的“出场圈”提示同色
_WHITE = (240, 245, 250)

_SS = 4                          # 超采样：先放大 4 倍画，再缩回去，边缘才干净
_ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)
# iconutil 要求的固定文件名 → 像素尺寸
_ICONSET = {
    "icon_16x16.png": 16, "icon_16x16@2x.png": 32,
    "icon_32x32.png": 32, "icon_32x32@2x.png": 64,
    "icon_128x128.png": 128, "icon_128x128@2x.png": 256,
    "icon_256x256.png": 256, "icon_256x256@2x.png": 512,
    "icon_512x512.png": 512, "icon_512x512@2x.png": 1024,
}


def _draw_flag(n: int) -> Image.Image:
    """单独一层：一根靖杆 + 一面带波浪的方格旗（透明底）。"""
    layer = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    # 旗面：4 列 × 3 行 的格子；深色格子用比底色略亮的颜色，旗的轮廓才看得见。
    # 每格画成四边形，左右两边按正弦错开 —— 于是整面旗是“波浪”的，
    # 比直挺挺的矩形生动，而且计算量还是零头。
    x0, x1 = n * 0.255, n * 0.815
    y_top = n * 0.265
    cols, rows = 4, 3
    cw = (x1 - x0) / cols
    ch = n * 0.128
    amp = n * 0.045
    wave = lambda x: amp * math.sin(2 * math.pi * (x - x0) / (x1 - x0))  # noqa: E731

    dark_cell = (38, 52, 76, 255)
    for i in range(cols):
        for j in range(rows):
            xa, xb = x0 + i * cw, x0 + (i + 1) * cw
            ya = y_top + j * ch + wave(xa)
            yb = y_top + j * ch + wave(xb)
            fill = _WHITE if (i + j) % 2 == 0 else dark_cell
            d.polygon([(xa, ya), (xb, yb), (xb, yb + ch), (xa, ya + ch)], fill=fill)

    # 旗杆：琥珀色，和 HUD 里的“出场圈”提示同色
    pole_w = max(2, n * 0.055)
    d.rounded_rectangle([n * 0.175, n * 0.235, n * 0.175 + pole_w, n * 0.735],
                        radius=pole_w / 2, fill=_AMBER)

    # 稍微斜一点，像是比赛中在飘
    return layer.rotate(-7, resample=Image.BICUBIC, center=(n * 0.5, n * 0.45))


def render(size: int) -> Image.Image:
    """画一张 size×size 的 RGBA 图标。"""
    n = max(8, size * _SS)
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))

    # ---- 圆角方块背景，带一点从上到下的渐变（纯色在浅色桌面上会显得很平）----
    radius = int(n * 0.22)
    mask = Image.new("L", (n, n), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, n - 1, n - 1], radius=radius, fill=255)
    grad = Image.new("RGBA", (n, n))
    gd = ImageDraw.Draw(grad)
    for y in range(n):
        f = y / max(1, n - 1)
        gd.line([(0, y), (n, y)], fill=tuple(
            int(_BG_TOP[i] + (_BG_BOTTOM[i] - _BG_TOP[i]) * f) for i in range(3)) + (255,))
    img.paste(grad, (0, 0), mask)

    img.alpha_composite(_draw_flag(n))
    return img.resize((size, size), Image.LANCZOS)


def save_ico(path: Path) -> Path:
    """.ico：一张 256 的图 + 多尺寸声明（Windows 自己挑合适的那张）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    render(256).save(path, format="ICO", sizes=[(s, s) for s in _ICO_SIZES])
    return path


def save_icns(path: Path) -> Path | None:
    """
    .icns：先拼出 .iconset 目录，再交给 iconutil 打包。

    iconutil 是 macOS 自带的，所以这个函数在别的平台上直接跳过 ——
    Windows 那边只需要 .ico。
    """
    if sys.platform != "darwin":
        return None
    iconset = path.parent / "race-gopro.iconset"
    if iconset.exists():
        for f in iconset.iterdir():
            f.unlink()
    iconset.mkdir(parents=True, exist_ok=True)
    for name, px in _ICONSET.items():
        render(px).save(iconset / name)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(path)],
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"iconutil 打包失败：{done.stderr.strip()}")
    for f in iconset.iterdir():
        f.unlink()
    iconset.rmdir()
    return path


def make_icons() -> list[Path]:
    """生成打包要用的图标，返回实际生成的文件。"""
    made = [save_ico(OUT / "race-gopro.ico")]
    icns = save_icns(OUT / "race-gopro.icns")
    if icns:
        made.append(icns)
    return made


def _preview(path: Path) -> None:
    """把几个尺寸并排拼一张图，方便肉眼检查小尺寸糊没糊。"""
    sizes = (16, 32, 64, 128, 256)
    pad = 16
    W = sum(sizes) + pad * (len(sizes) + 1)
    H = max(sizes) + pad * 2
    sheet = Image.new("RGBA", (W, H), (245, 246, 248, 255))
    x = pad
    for s in sizes:
        ic = render(s)
        sheet.alpha_composite(ic, (x, pad + (max(sizes) - s) // 2))
        x += s + pad
    sheet.convert("RGB").save(path)
    print(f"预览图：{path}")


if __name__ == "__main__":
    files = make_icons()
    for f in files:
        print(f"  ✓ {f.relative_to(HERE.parent)}（{f.stat().st_size / 1024:.0f} KB）")
    if "--preview" in sys.argv:
        _preview(OUT / "preview.png")
