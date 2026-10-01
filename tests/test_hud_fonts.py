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
from pathlib import Path

from rapp import overlay

ROOT = Path(__file__).resolve().parent.parent


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


def test_windows_prefers_regular_weight() -> None:
    """
    Windows 的候选必须**常规字重在前**。

    踩过的坑：这张表原本是 `msyhbd.ttc`（雅黑**粗体**）打头，于是 Windows 上的
    HUD 一直比 mac 粗一整档（mac 用的是 Hiragino 常规），用户一眼就看出来了。
    名字里带 `bd` / `-B` 的都是粗体，别再把它们排前面。
    """
    assert overlay._SANS_WINDOWS[0] == "msyh.ttc", (
        f"Windows 第一个正文字体是 {overlay._SANS_WINDOWS[0]}，应该用雅黑常规")
    assert overlay._MONO_WINDOWS[0] == "consola.ttf", (
        f"Windows 第一个等宽字体是 {overlay._MONO_WINDOWS[0]}，应该用 Consolas 常规")
    for names in (overlay._SANS_WINDOWS, overlay._MONO_WINDOWS):
        bold = [n for n in names if "bd" in n.lower() or "-b" in n.lower()]
        assert len(bold) < len(names), f"{names} 里全是粗体？"
        assert names[0] not in bold, f"{names[0]} 是粗体，不该排第一"


def test_bundled_fonts_dont_hijack_system_fonts() -> None:
    """自带的字体必须排在最后，不能抢系统字体的位置。

    平台自带的字更好看（macOS 的 Hiragino、Windows 的微软雅黑），而且 HUD 的
    版面当初就是按它们的度量调的。自带那份只在“系统里一个都没有”时才上场。
    （v0.1.6 第一版把它们排在了最前面，结果 mac 上的字也变了，观感明显变差。）
    """
    assert overlay._FONT_NAMES[-1] == "NotoSansSC-Regular.otf"
    assert overlay._MONO_NAMES[-1] == "NotoSansMono-Regular.ttf"


def test_bundled_font_is_the_last_resort() -> None:
    """
    系统字体一个都找不到时，必须轮到包内自带的那份 —— 这才是它存在的理由。

    做法：把字体目录换成“只有包内目录”（等价于系统里什么字体都没有），
    看能不能挑出 Noto —— 这正是 v0.1.4 的 Windows 机器上的情形。
    """
    bundled = overlay._bundled_font_dir()
    if bundled is None or not (bundled / "NotoSansSC-Regular.otf").exists():
        return              # 还没跑过构建、没下字体，跳过
    saved = overlay._font_dirs
    overlay._font_dirs = lambda: [bundled]
    try:
        got = overlay._find_font(overlay._FONT_NAMES, overlay._SANS_HINTS)
    finally:
        overlay._font_dirs = saved
    assert got == str(bundled / "NotoSansSC-Regular.otf"), f"兜底失败，选中了 {got}"


def test_bundled_dir_in_macos_app_layout(tmp_path: Path) -> None:
    """
    macOS 的 .app 里，字体的落点是 Contents/Resources/fonts，
    而 PyInstaller 6 的 sys._MEIPASS 指向 Contents/Frameworks。

    只认 _MEIPASS/fonts 的话，mac 上永远找不到自带字体 —— 而且**看不出来**：
    它会静默退回系统字体（Hiragino），画面很正常，只是保底是空的。
    Windows 的 onedir 布局则确实是 _MEIPASS/fonts。
    """
    contents = tmp_path / "X.app" / "Contents"
    fonts = contents / "Resources" / "fonts"
    fonts.mkdir(parents=True)
    (fonts / "NotoSansSC-Regular.otf").write_bytes(b"x")
    (contents / "MacOS").mkdir()
    (contents / "Frameworks").mkdir()

    saved_exe = sys.executable
    had_meipass = hasattr(sys, "_MEIPASS")
    saved_meipass = getattr(sys, "_MEIPASS", None)
    try:
        sys.executable = str(contents / "MacOS" / "race-gopro")
        sys._MEIPASS = str(contents / "Frameworks")      # noqa: SLF001
        got = overlay._bundled_font_dir()
    finally:
        sys.executable = saved_exe
        if had_meipass:
            sys._MEIPASS = saved_meipass                 # noqa: SLF001
        elif hasattr(sys, "_MEIPASS"):
            del sys._MEIPASS                             # noqa: SLF001
    assert got == fonts, f"没找到 .app 里的字体，得到 {got}"


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

    if "msyh.ttc" in overlay._FONT_NAMES and "consola.ttf" in overlay._MONO_NAMES:
        print(f"✓ Windows 候选在表里，且常规字重优先（{overlay._SANS_WINDOWS[0]}"
              f" / {overlay._MONO_WINDOWS[0]}）")
    else:
        print("✗ Windows 的字体候选被删了")
        failed += 1

    fonts_dir = ROOT / "packaging" / "fonts"
    if (fonts_dir / "NotoSansSC-Regular.otf").exists():
        saved = overlay._font_dirs
        overlay._font_dirs = lambda: [fonts_dir]
        try:
            got = overlay._find_font(overlay._FONT_NAMES, overlay._SANS_HINTS)
        finally:
            overlay._font_dirs = saved
        if got == str(fonts_dir / "NotoSansSC-Regular.otf"):
            print("✓ 系统里没有字体时，包内自带的能兜住")
        else:
            print(f"✗ 兜底失败，选中了 {got}")
            failed += 1
        if (fonts_dir / "LICENSE-OFL.txt").exists():
            print("✓ OFL 授权原文随字体一起在包里")
        else:
            print("✗ 缺 OFL 授权原文（发包前要补）")
            failed += 1
    else:
        print("· 包里还没下字体（构建时会下）")

    print()
    print("全部通过 ✓" if not failed else f"有 {failed} 项失败 ✗")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
