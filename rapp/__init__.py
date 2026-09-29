"""
rapp - Race Analysis for GoPro
==============================

从 GoPro 视频（HERO 8/9/10/11/12/13）的 MP4 里提取 GPMF 遥测数据，
并输出卡丁车/赛道练习所需的专业赛车信息：圈速、分段、G 值、弯道分析等。

模块划分（对照 Python 数据处理习惯）：
    gpmf.py      —— 二进制解析层。相当于 pandas.read_csv，但解析的是 GoPro 私有二进制格式
    telemetry.py —— 统一采样层。把散落的传感器流合成一张对齐的"表"（类似 DataFrame）
    geo.py       —— 地理计算层。经纬度 → 平面米制坐标（类似把球坐标投影到笛卡尔系）
    imu.py       —— 传感器标定层。加速度计姿态对齐 + 摄像头安装角标定
    laps.py      —— 圈速逻辑层。自动找起终点线、切圈、算分段
    analysis.py  —— 分析层。弯道识别、轮胎摩擦圆、理论最佳圈
    report.py    —— 输出层。终端报表 + CSV/JSON
    charts.py    —— 输出层。matplotlib 静态图
    dashboard.py —— 输出层。交互式网页看板
    overlay.py   —— 输出层。把 HUD 烧进原视频
"""

__version__ = "0.1.0"
