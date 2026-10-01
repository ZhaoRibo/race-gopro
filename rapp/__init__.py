"""
rapp - Race Analysis for GoPro
==============================

从 GoPro 视频（HERO 8/9/10/11/12/13）的 MP4 里提取 GPMF 遥测数据，
并输出卡丁车/赛道练习所需的专业赛车信息：圈速、分段、G 值、弯道分析等。

模块划分（对照 Python 数据处理习惯）：
    gpmf.py      —— 二进制解析层。相当于 pandas.read_csv，但解析的是 GoPro 私有二进制格式
    telemetry.py —— 统一采样层。把散落的传感器流合成一张对齐的"表"（类似 DataFrame）
    geo.py       —— 地理计算层。经纬度 → 平面米制坐标（类似把球坐标投影到笛卡尔系）
    imu.py       —— 传感器层。用 GRAV 扣重力，再把加速度投影成纵向 / 横向 G
    laps.py      —— 圈速逻辑层。自动找起终点线、切圈、算分段
    analysis.py  —— 分析层。弯道识别、轮胎摩擦圆、理论最佳圈
    report.py    —— 输出层。终端报表 + CSV/JSON
    charts.py    —— 输出层。matplotlib 静态图
    dashboard.py —— 输出层。交互式网页看板
    overlay.py   —— 输出层。把 HUD 烧进原视频
"""

import os
import sys

# ---------------------------------------------------------------------------
# 没有控制台时，先把 stdout / stderr 补上
# ---------------------------------------------------------------------------
# 为什么必须放在这里、而且必须在任何子模块被导入之前：
#
# PyInstaller 用 console=False（窗口模式）打包后，Windows 上 sys.stdout 和
# sys.stderr 不是“空流”，而是 **None**。任何一句 print、任何 `.isatty()`
# 都会当场抛 AttributeError 把程序打死 —— 而且可能发生在**导入阶段**。
# report.py 模块级那句“终端支不支持颜色”就是例子，它比 app.py 里负责把输出
# 接到日志的 _redirect_output() 还早，根本轮不到那道救援。
#
# 真实事故：v0.1.2 的 Windows 包一启动就弹窗
#     AttributeError: 'NoneType' object has no attribute 'isatty'
#
# 放在包的 __init__ 里，是因为**所有入口都要先经它**：打包版
# （entry.py → rapp.app）、源码版（python -m rapp.app）、命令行
# （analyze.py → from rapp import ...）全都覆盖，一处管住全部。
# 先接到系统垃圾桶，稍后 app.py 的 _redirect_output() 会换成真正的日志文件。
if sys.stdout is None or sys.stderr is None:
    _sink = open(os.devnull, "w", encoding="utf-8", errors="replace")
    if sys.stdout is None:
        sys.stdout = _sink
    if sys.stderr is None:
        sys.stderr = _sink
    if sys.__stdout__ is None:
        sys.__stdout__ = _sink
    if sys.__stderr__ is None:
        sys.__stderr__ = _sink

__version__ = "0.1.7"
"""包版本。发版时和 `packaging/build.py` 里的 `VERSION` 保持一致。"""
