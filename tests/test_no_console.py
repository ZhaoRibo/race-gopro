"""
无控制台启动的回归测试（Windows 打包版）
========================================

为什么需要这个：**这个问题在 macOS 上永远复现不出来。**

PyInstaller 用 `console=False`（窗口模式）打包后，Windows 上 `sys.stdout` /
`sys.stderr` 不是"空流"，而是 **None**。任何一句 print、任何 `.isatty()` 都会
当场抛 `AttributeError` 把程序打死 —— 而且可能发生在**导入阶段**：
`rapp/report.py` 模块级那句 `_COLOR = sys.stdout.isatty()` 就是，它比 `app.py`
里负责"把输出接到日志"的 `_redirect_output()` 还早，根本轮不到那道救援。

真实事故：v0.1.2 的 Windows 包一启动就弹窗

    AttributeError: 'NoneType' object has no attribute 'isatty'
    entry.py:16 → app.py:46 → serve.py:54 → report.py:24

（顺带一提：那个弹窗是 PyInstaller 的窗口模式兜底打印的。因为 `sys.stderr`
是 None，解释器自己**没法**把 traceback 打出来 —— 这也是它难查的原因之一。）

测试做法：在**子进程**里把两个流真置成 None，再导入打包版启动时要过的那串
模块，看它是死是活。之所以要另开进程，是因为导入是带缓存的，在当前进程里
测不出第二次。

运行方式：
    .venv/bin/python -m tests.test_no_console
    .venv/bin/python -m pytest tests/
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 子进程里跑的脚本。要点：出错时不能靠 sys.stderr 报告（它已经是 None 了），
# 所以把情况写进一个探针文件，让父进程读。
_CHILD = r"""
import os
import sys
import traceback

sys.stdout = None
sys.stderr = None
sys.__stdout__ = None
sys.__stderr__ = None

probe = open(os.environ["RACEGOPRO_PROBE"], "w", encoding="utf-8")
probe.write("流已置空: stdout=%r stderr=%r\n" % (sys.stdout, sys.stderr))

try:
    if os.environ["RACEGOPRO_CASE"] == "startup":
        # 真实场景：一上来就没有控制台，导入打包版启动要过的那串模块
        import rapp.app
        import rapp.overlay
    else:
        # 单独验 report.py 自己的空值安全：正常导入一遍（此时包里的总护栏
        # 已经跑完，流是好的），再把流掐掉并**重新加载**它 —— 模块级那行
        # 会再执行一次，这次没有护栏兜着。
        import importlib
        import rapp
        import rapp.report
        sys.stdout = None
        sys.stderr = None
        importlib.reload(rapp.report)
except BaseException:
    traceback.print_exc(file=probe)
    probe.write("结果: 炸了\n")
    probe.close()
    raise SystemExit(1)

probe.write("结果: 通过\n")
probe.close()
"""


def _run(case: str) -> tuple[int, str]:
    """在子进程里模拟无控制台启动，返回 (退出码, 探针内容)。"""
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "probe.txt"
        env = dict(os.environ, RACEGOPRO_PROBE=str(probe), RACEGOPRO_CASE=case)
        done = subprocess.run([sys.executable, "-c", _CHILD], cwd=ROOT, env=env,
                              capture_output=True, text=True)
        text = probe.read_text(encoding="utf-8") if probe.exists() else ""
    return done.returncode, text


def test_startup_without_console_works() -> None:
    """打包版启动路径：没有任何控制台也必须能导入、能跑起来。"""
    code, text = _run("startup")
    assert "stdout=None" in text, f"没能真的把流置空，测试失去意义：\n{text}"
    assert code == 0, f"无控制台导入失败（Windows 打包版就是这样一启动就弹窗）：\n{text}"


def test_report_module_is_none_safe() -> None:
    """即使没有总护栏，report.py 自己也不该因为 stdout 是 None 而崩。"""
    code, text = _run("reload-report")
    assert code == 0, f"report.py 不是空值安全的：\n{text}"


def main() -> int:
    failed = 0
    for name, case, why in (
        ("打包版启动路径（无控制台）", "startup",
         "导入 rapp.app / rapp.overlay 必须成功"),
        ("report.py 自身的空值安全", "reload-report",
         "重新加载时 stdout 已是 None，模块级那行必须扛得住"),
    ):
        code, text = _run(case)
        ok = code == 0 and "stdout=None" in text
        print(f"{'✓' if ok else '✗'} {name} —— {why}")
        if not ok:
            failed += 1
            print("   " + text.replace("\n", "\n   ").strip())
    print()
    print("全部通过 ✓" if not failed else f"有 {failed} 项失败 ✗")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
