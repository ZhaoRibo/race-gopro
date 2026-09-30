"""
打包用的入口脚本
================

为什么需要这么一层：`rapp/app.py` 里用的是包内相对导入（`from . import serve`），
只有作为包的一部分被 import 时才成立。PyInstaller 需要一个"顶层脚本"来跑，
所以这里薄薄包一层。

从源码跑的时候用不到这个文件（`python -m rapp.app` 就够了）。
"""

from __future__ import annotations

import sys

# 注意：如果 Windows 上是以窗口模式（console=False）启动的，此刻 sys.stdout /
# sys.stderr 都是 None。救命的那道护栏在 rapp/__init__.py 里 —— 导入 rapp.app
# 必然先执行包的初始化，所以到这里已经安全了，不要把这个 import 提到上面去。
from rapp.app import main

if __name__ == "__main__":
    sys.exit(main())
