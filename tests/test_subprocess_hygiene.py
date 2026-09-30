"""
子进程卫生检查（Windows 上不弹黑框）
====================================

为什么需要这个：打包成窗口程序（`console=False`）后，Windows 上**每起一个子进程
都会闪一个新的控制台窗口**。而这个程序一分析视频就反复调 ffmpeg（取单帧、解
遥测、编 HUD），于是看板上每点一下，屏幕上就闪一个黑框又立刻消失 —— 用户实测
反馈"很烦"。

解法是给每个子进程加 `CREATE_NO_WINDOW`，而它**只在 Windows 上生效，macOS 上
就算漏掉也完全看不出来**。所以规矩是：所有子进程都走 `rapp/subproc.py` 的
`run()` / `popen()`，不直接调 `subprocess`。这个测试就盯着这条规矩。

运行方式：
    .venv/bin/python -m tests.test_subprocess_hygiene
    .venv/bin/python -m pytest tests/
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAPP = ROOT / "rapp"

# 只有它自己可以直接调 subprocess —— 它就是那个包装层
ALLOWED = {"subproc.py"}

# 会真的起进程、因而需要"别弹控制台"的调用
_SPAWN = re.compile(r"subprocess\.(run|Popen|check_output|call)\s*\(")

_FAKE_FLAG = 0x08000000


def _offenders() -> list[str]:
    bad: list[str] = []
    for path in sorted(RAPP.glob("*.py")):
        if path.name in ALLOWED:
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for n, line in enumerate(lines, 1):
            if _SPAWN.search(line):
                bad.append(f"{path.name}:{n}  {line.strip()}")
    return bad


def test_all_spawns_go_through_subproc() -> None:
    bad = _offenders()
    assert not bad, ("这些地方直接调了 subprocess —— Windows 上每次都会闪一个黑框，"
                     "改用 rapp/subproc.py 的 run() / popen()：\n  " + "\n  ".join(bad))


def _capture_kwargs(platform: str) -> dict:
    """把 subprocess.run 换掉，抓 subproc.run 实际传下去的创建标志。"""
    import subprocess

    from rapp import subproc

    captured: dict = {}
    real_run = subprocess.run
    real_platform = sys.platform
    had_flag = hasattr(subprocess, "CREATE_NO_WINDOW")
    real_flag = getattr(subprocess, "CREATE_NO_WINDOW", None)

    def fake_run(cmd, **kwargs):
        captured.update(kwargs)
        return None

    try:
        subprocess.run = fake_run
        subprocess.CREATE_NO_WINDOW = _FAKE_FLAG      # macOS 上没这个常量，补一个
        sys.platform = platform
        subproc.run(["x"])
    finally:
        subprocess.run = real_run
        sys.platform = real_platform
        if had_flag:
            subprocess.CREATE_NO_WINDOW = real_flag
        else:
            del subprocess.CREATE_NO_WINDOW
    return captured


def test_windows_gets_the_no_window_flag() -> None:
    """Windows 上必须真的带上 CREATE_NO_WINDOW。"""
    assert _capture_kwargs("win32").get("creationflags") == _FAKE_FLAG


def test_posix_gets_a_harmless_zero() -> None:
    """其他平台传 0 —— 在 POSIX 上这是默认值，等于什么都没做。"""
    assert _capture_kwargs("darwin").get("creationflags") == 0


def test_file_dialog_has_an_owner_window() -> None:
    """
    Windows 的文件框必须挂在 owner 窗体上。

    实测踩过（v0.1.3）：不给 owner 就直接 `ShowDialog()`，对话框会**躲在浏览器
    后面** —— 因为那个进程没有控制台、也没有前台窗口可依附，Windows 不会把它
    激活到最前面。用户看到的就是"点了没反应"，而 ShowDialog 是阻塞的，请求也就
    一直挂着。这个检查只在代码里确实用了 ShowDialog 时才生效。
    """
    text = (RAPP / "app.py").read_text(encoding="utf-8")
    if "ShowDialog(" not in text:
        return
    assert "$d.ShowDialog($f)" in text and "TopMost" in text, (
        "app.py 里的 PowerShell 文件框又变回不带 owner 窗体的 ShowDialog() 了；"
        "那样对话框会躲在浏览器后面，用户会以为「点了没反应」"
    )


def main() -> int:
    bad = _offenders()
    print(f"扫了 {len(list(RAPP.glob('*.py')))} 个模块")
    if bad:
        print("\n✗ 这些地方绕过了 rapp/subproc.py：")
        for item in bad:
            print("  -", item)
        print("\nWindows 上每调一次就会闪一个黑框；改用 run() / popen()。")
        return 1
    print("✓ 所有子进程都走 rapp/subproc.py")

    win = _capture_kwargs("win32").get("creationflags")
    posix = _capture_kwargs("darwin").get("creationflags")
    print(f"✓ Windows 上带 CREATE_NO_WINDOW：0x{win:08X}" if win == _FAKE_FLAG
          else f"✗ Windows 上没有带上标志（拿到 {win}）")
    print(f"✓ 其他平台是 0（无害）：{posix}" if posix == 0
          else f"✗ 其他平台传了 {posix}，在 POSIX 上会抛 ValueError")

    ok = win == _FAKE_FLAG and posix == 0
    try:
        test_file_dialog_has_an_owner_window()
        print("✓ 文件框挂在 TopMost 的 owner 窗体上")
    except AssertionError as exc:
        print("✗", exc)
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
