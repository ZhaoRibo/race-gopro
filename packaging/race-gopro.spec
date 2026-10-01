# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller 打包配置
====================

产物：
    macOS   → dist/race-gopro.app    （双击即用的应用包）
    Windows → dist/race-gopro/       （文件夹，里面有 race-gopro.exe）

两个要点：

1. **ffmpeg / ffprobe 一起打进去**（塞在包的 `bin/` 下）。用户机器上装没装、
   装的是几年前的版本，都不影响 —— 这是"下载下来就能用"的前提。
   这份静态包由 packaging/build.py 提前下好放在 packaging/ffmpeg/。
2. **不加控制台窗口**（console=False）。所以程序自己会把 stdout 接到日志文件上
   （见 rapp/app.py 的 _redirect_output），不然 Windows 上 sys.stdout 是 None，
   随便一句 print 就会把程序打死。

PyInstaller 会在执行本文件时预定义 SPECPATH（本文件所在目录）。
"""

import os
import sys
from pathlib import Path

ROOT = Path(SPECPATH).parent          # noqa: F821 — SPECPATH 由 PyInstaller 注入
FFMPEG_DIR = ROOT / "packaging" / "ffmpeg"

APP_NAME = "race-gopro"
VERSION = os.environ.get("RACEGOPRO_VERSION", "0.1.0")
SUFFIX = ".exe" if sys.platform == "win32" else ""

# ---- 把 ffmpeg / ffprobe 塞进包的 bin/ 目录 ----
binaries = []
for _tool in ("ffmpeg", "ffprobe"):
    _p = FFMPEG_DIR / (_tool + SUFFIX)
    if not _p.exists():
        raise SystemExit(f"缺少 {_p}\n先跑 packaging/build.py（它会自动下载），"
                         "或者手动把静态 ffmpeg/ffprobe 放进 packaging/ffmpeg/")
    binaries.append((str(_p), "bin"))

# ---- matplotlib 的字体缓存（构建时预建，见 build.py）----
# 不带的话，用户第一次启动会花十几秒扫系统字体，界面上一片空白。
datas = []
_MPL_CACHE = ROOT / "packaging" / "matplotlib"
if _MPL_CACHE.is_dir() and any(_MPL_CACHE.iterdir()):
    datas.append((str(_MPL_CACHE), "matplotlib"))

# ---- HUD 用的字体（build.py 下的开源字体，约 9 MB）----
# **必须带上**。不带的话，用户机器上缺字体时 HUD 的文字会静默变成 11 px 的位图，
# 小到看不见（v0.1.4 的 Windows 包就是这么翻车的），而且不给任何报错。
_FONT_DIR = ROOT / "packaging" / "fonts"
_FONT_NEEDED = ("NotoSansSC-Regular.otf", "NotoSansMono-Regular.ttf")
_missing = [n for n in _FONT_NEEDED if not (_FONT_DIR / n).exists()]
if _missing:
    raise SystemExit(f"缺少 HUD 字体 {_missing}（应在 {_FONT_DIR}）—— "
                     "先跑 packaging/build.py，它会自动下载")
datas.append((str(_FONT_DIR), "fonts"))
if not (_FONT_DIR / "LICENSE-OFL.txt").exists():
    print("⚠ 没有 OFL 授权原文，发包前补一份（fonts/LICENSE-OFL.txt）")

a = Analysis(                          # noqa: F821
    [str(ROOT / "packaging" / "entry.py")],
    pathex=[str(ROOT)],                # 让 PyInstaller 找得到仓库根的 analyze.py
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        # rapp/app.py 里是延迟 import 的（函数内部），列出来更保险
        "analyze", "tests", "tests.test_gpmf",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # 用不到的大件，排掉能省不少体积
        "tkinter", "PyQt5", "PyQt6", "PySide2", "PySide6", "wx",
        "IPython", "jupyter", "notebook", "pytest", "docutils",
        "matplotlib.backends._backend_tk", "matplotlib.backends.backend_qt5agg",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)                      # noqa: F821

exe = EXE(                             # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,                     # 不要控制台窗口：输出走日志文件
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(                        # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name=APP_NAME,
)

if sys.platform == "darwin":
    app = BUNDLE(                      # noqa: F821
        coll,
        name=f"{APP_NAME}.app",
        icon=None,
        bundle_identifier="local.race-gopro.launcher",
        info_plist={
            "CFBundleName": "卡丁车遥测分析",
            "CFBundleDisplayName": "卡丁车遥测分析",
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "CFBundleGetInfoString": f"race-gopro {VERSION}",
            "LSApplicationCategoryType": "public.app-category.sports",
            "LSMinimumSystemVersion": "11.0",
            "NSHighResolutionCapable": True,
        },
    )
