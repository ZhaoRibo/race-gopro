"""
HUD 字体检查（别再悄悄退到那个 11 px 的位图字体）
================================================

为什么需要这个：HUD 文字的大小完全取决于**能不能加载到一个真正的字体文件**。
一旦 `ImageFont.truetype` 全都失败，PIL 会退回到内置位图字体 —— 固定 11 px、
不能缩放。画在 4K 画面上就是"字小到几乎看不见"。

真实事故：v0.1.4 的 Windows 包。候选字体表里全是 mac / Linux 的路径，
Windows 上一个都不存在 → 整块 HUD 的字都成了 11 px 的小点。

这个测试盯两件事：

1. **当前平台上真的找得到字体** —— 找不到就红，说明候选表漏了这个平台；
2. **把候选表清空，兜底字体也必须随字号缩放** —— 这条才是关键：出问题的从来
   不是"有没有字体"，而是"字悄悄变小了"。当初那句 `ImageFont.load_default()`
   不传 size，拿到的是位图字体，于是整块 HUD 都废了。

运行方式：
    .venv/bin/python -m tests.test_hud_fonts
    .venv/bin/python -m pytest tests/
"""

from __future__ import annotations

import sys

from rapp import overlay


def _tap_height(font) -> float:
    """量一下这串字实际占多高 —— 用它判断字号有没有真的生效。"""
    box = font.getbbox("0123")
    return float(box[3] - box[1])


def test_current_platform_has_fonts() -> None:
    """当前平台上候选表必须能找到字体。"""
    assert overlay._find_font(overlay._FONT_NAMES, overlay._SANS_HINTS), (
        f"在 {sys.platform} 上找不到任何正文字体 —— 候选表是不是又漏了本平台的路径？"
    )
    assert overlay._find_font(overlay._MONO_NAMES, overlay._MONO_HINTS), (
        "找不到等宽字体 —— 大号计时数字只能退回正文字体，数字跳动时会左右抖"
    )


def test_fallback_font_still_scales() -> None:
    """
    一个候选字体都找不到时，兜底字体也必须随字号缩放。

    直接对着 v0.1.4 那个 bug：Windows 上所有候选都不存在，而
    `ImageFont.load_default()`（不传 size）给的是固定 11 px 的位图字体。
    """
    saved_sans, saved_mono = overlay._FONT_NAMES, overlay._MONO_NAMES
    saved_dirs = overlay._font_dirs
    overlay._FONT_NAMES = []            # 模拟"一个候选都没有"（≈ Windows 当初的情况）
    overlay._MONO_NAMES = []
    overlay._font_dirs = lambda: []     # 连"扫字体目录"这条兜底也堵死
    try:
        fonts = overlay._Fonts()
        small = _tap_height(fonts.get(20, mono=True))
        big = _tap_height(fonts.get(100, mono=True))
    finally:
        overlay._FONT_NAMES, overlay._MONO_NAMES = saved_sans, saved_mono
        overlay._font_dirs = saved_dirs

    assert big > small * 3, (
        f"字号 100 的字只有字号 20 的 {big / max(small, 1e-6):.1f} 倍高 —— "
        "说明退回了不能缩放的位图字体，4K 画面上会看不清"
    )


def test_windows_candidates_are_present() -> None:
    """Windows 的字体必须留在候选表里 —— 当初就是漏了它。"""
    assert "msyhbd.ttc" in overlay._FONT_NAMES, "候选表里没有微软雅黑"
    assert "consolab.ttf" in overlay._MONO_NAMES, "候选表里没有 Consolas"


def main() -> int:
    failed = 0

    sans = overlay._find_font(overlay._FONT_NAMES, overlay._SANS_HINTS)
    mono = overlay._find_font(overlay._MONO_NAMES, overlay._MONO_HINTS)
    print(f"平台 {sys.platform}：")
    print(f"  正文字体：{sans or '（没找到 ✗）'}")
    print(f"  等宽字体：{mono or '（没找到 ✗）'}")
    if not sans or not mono:
        failed += 1

    # 兜底字体必须能缩放
    saved = (overlay._FONT_NAMES, overlay._MONO_NAMES, overlay._font_dirs)
    overlay._FONT_NAMES, overlay._MONO_NAMES = [], []
    overlay._font_dirs = lambda: []
    try:
        fonts = overlay._Fonts()
        small = _tap_height(fonts.get(20, mono=True))
        big = _tap_height(fonts.get(100, mono=True))
    finally:
        overlay._FONT_NAMES, overlay._MONO_NAMES, overlay._font_dirs = saved
    ratio = big / max(small, 1e-6)
    if big > small * 3:
        print(f"✓ 没字体时的兜底字体仍随字号缩放（20→{small:.0f}px，100→{big:.0f}px，"
              f"{ratio:.1f} 倍）")
    else:
        print(f"✗ 兜底字体不缩放（{ratio:.1f} 倍）—— 4K 上会看不清")
        failed += 1

    if "msyhbd.ttc" in overlay._FONT_NAMES and "consolab.ttf" in overlay._MONO_NAMES:
        print("✓ Windows 的字体候选还在表里")
    else:
        print("✗ Windows 的字体候选被删了")
        failed += 1

    print()
    print("全部通过 ✓" if not failed else f"有 {failed} 项失败 ✗")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
