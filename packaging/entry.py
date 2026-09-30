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

from rapp.app import main

if __name__ == "__main__":
    sys.exit(main())
