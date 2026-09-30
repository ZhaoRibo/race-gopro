"""
仓库文件名的跨平台检查
======================

为什么需要这个：**macOS 上合法的文件名，Windows 上未必合法。**

真实事故（v0.1.1 那次）：给 Windows 打包分支写调试脚本时，为了骗过平台判断
把 `os.name` 改成了 `"nt"`，于是 `pathlib.Path("/tmp/xxx.zip")` 变成了
WindowsPath，字符串化后是 `\\tmp\\xxx.zip`。在 macOS 上这只是一个**名字里带
反斜杠**的普通文件，被创建在仓库根目录，接着被 `git add -A` 顺手提交了。

本地一切正常（mac 的任务照常通过），推标签触发的 CI 却在 Windows 任务上
**第一步就挂了** —— `actions/checkout` 直接失败，一行代码都没跑到：

    error: invalid path '\\tmp\\fake-gyan.zip'

这类问题本地无法复现，所以用脚本兜一下：把 git 跟踪的所有文件名拿出来，
检查里面有没有 Windows 不允许的东西。

运行方式：
    .venv/bin/python -m tests.test_paths       # 直接跑
    .venv/bin/python -m pytest tests/          # 用 pytest 跑
"""

from __future__ import annotations

import subprocess

# Windows 文件名里不允许出现的字符。反斜杠尤其阴险：它在 Windows 上是路径
# 分隔符，出现在文件名里就意味着"这其实是条路径"，checkout 时必然报错。
_BAD_CHARS = set('\\:*?"<>|')

# 保留设备名：Windows 上叫这些的文件根本创建不出来（不区分大小写，
# 连 con.txt 这种也照样不行）
_RESERVED_NAMES = {"con", "prn", "aux", "nul"}
for _i in range(1, 10):
    _RESERVED_NAMES.add(f"com{_i}")
    _RESERVED_NAMES.add(f"lpt{_i}")


def tracked_files() -> list[str]:
    """git 跟踪的所有文件路径（用 -z，免得中文名被转义成八进制）。"""
    done = subprocess.run(["git", "ls-files", "-z"], capture_output=True, check=True)
    text = done.stdout.decode("utf-8", "surrogateescape")
    return [p for p in text.split("\0") if p]


def problems(paths: list[str]) -> list[str]:
    """挑出在 Windows 上会出问题的路径，返回人话解释（没问题就是空列表）。"""
    out: list[str] = []
    for path in paths:
        hit = sorted(_BAD_CHARS & set(path))
        if hit:
            shown = " ".join(repr(c) for c in hit)
            out.append(f"{path!r} 含有 Windows 不允许的字符：{shown}")
        for part in path.split("/"):
            # Windows 会把结尾的空格/点悄悄吃掉，导致名字对不上
            if part != part.rstrip(" ."):
                out.append(f"{path!r} 的 “{part}” 末尾有空格或点，Windows 存不住")
            if part.split(".")[0].lower() in _RESERVED_NAMES:
                out.append(f"{path!r} 的 “{part}” 是 Windows 保留设备名")
    return out


def test_no_windows_hostile_paths() -> None:
    bad = problems(tracked_files())
    assert not bad, ("仓库里有 Windows 上用不了的文件名，CI 的 Windows 任务会在 "
                     "checkout 那一步直接挂掉（连代码都跑不到）：\n  "
                     + "\n  ".join(bad))


def test_checker_catches_the_real_case() -> None:
    """检查器本身得管用 —— 拿当初那个真实文件名试一下。"""
    assert problems(["\\tmp\\fake-gyan.zip"])


def main() -> int:
    paths = tracked_files()
    bad = problems(paths)
    print(f"检查了 {len(paths)} 个被 git 跟踪的文件")

    # 先证明这套判断不是摆设：拿实际闯过祸的名字试一次
    assert problems(["\\tmp\\fake-gyan.zip"]), "检查器失灵了"

    if bad:
        print("\n✗ 有问题：")
        for item in bad:
            print("  -", item)
        print("\n这种名字在 macOS 上完全正常，在 Windows 上会让 checkout 失败。")
        print("清掉的办法：git rm -- '<那个文件名>'，然后提交。")
        return 1

    print("✓ 全部符合 Windows 的文件名规则")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
