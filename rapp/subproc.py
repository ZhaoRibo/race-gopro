"""
起子进程的统一入口 —— 只为了一件事：Windows 上别弹控制台窗口
==========================================================

为什么需要这个：

打包成窗口程序（PyInstaller 的 `console=False`）之后，Windows 上**每 spawn 一个
子进程，系统都会给它开一个新的控制台窗口**。而这个程序里 ffmpeg / ffprobe 会被
反复调用 —— 取单帧、解码遥测、编码 HUD —— 于是看板上每点一下，屏幕上就闪一个
黑框又立刻消失。用户的反馈是"很烦"。

`CREATE_NO_WINDOW` 就是关掉这个窗口的开关。

为什么非要收口到这一个模块：**漏掉任何一处就漏一个黑框**。这类调用散在
gpmf / serve / overlay / demo / app / dashboard 里，将来再加新调用时，别再直接写
`subprocess.run(...)` —— 用这里的 `run()` / `popen()`。
`tests/test_subprocess_hygiene.py` 会盯着这条规矩。

（macOS / Linux 上这两个封装就是普通的 subprocess —— `creationflags=0` 在 POSIX
上完全合法，等于什么都没做。）
"""

from __future__ import annotations

import subprocess
import sys


def _no_window() -> int:
    """
    Windows 上返回 CREATE_NO_WINDOW，其他平台返回 0。

    刻意写成函数而不是模块级常量：这样测试里能临时把 `sys.platform` 换成
    "win32"，验证"在 Windows 上确实带了这个标志" —— 否则这条分支在 mac 上
    永远测不到，而它偏偏只会在 Windows 上出问题。
    """
    if sys.platform != "win32":
        return 0
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000))


def run(cmd, **kwargs):
    """`subprocess.run`，自动带上"不要控制台窗口"。参数原样透传。"""
    kwargs.setdefault("creationflags", _no_window())
    return subprocess.run(cmd, **kwargs)


def popen(cmd, **kwargs):
    """`subprocess.Popen`，同上。"""
    kwargs.setdefault("creationflags", _no_window())
    return subprocess.Popen(cmd, **kwargs)


__all__ = ["run", "popen"]
