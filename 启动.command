#!/bin/sh
# 双击这个文件就能启动 —— 会弹出浏览器，选个视频就能分析。
#
# Finder 里双击 .command 会用「终端」执行它，所以这个窗口就是程序的日志：
# 分析进度、出片记录、报错都打在这里。关掉浏览器页面程序会自己退出，
# 也可以在这个窗口按 Ctrl+C 强制退出。
#
# 为什么不用 .app 包：.app 更"像应用"，但拖到别处就找不到仓库了；.command
# 用 $0 定位自己，放在仓库里哪个位置、被复制一份都还能跑。

cd "$(dirname "$0")" || exit 1

# 这一行是关键：用 $0 算出脚本自己所在的目录再进去，所以**从哪儿双击都能跑**
# —— Finder 双击给的是完整绝对路径，命令行里用相对路径 ./启动.command 也一样。

if [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
  echo "⚠ 没找到 .venv，用系统的 python3。依赖可能不全，建议先跑："
  echo "    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
else
  echo "找不到 Python。先装一个：https://www.python.org/downloads/"
  echo "装完在这个目录里跑一次：python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  read -r _
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "⚠ 没找到 ffmpeg（要它来读视频和出片）。装一下：brew install ffmpeg"
fi

# 让程序把浏览器地址打出来；用户按 Ctrl+C 或关掉页面都能结束。
# 用 exec：让 python 顶替这个 shell，退出码直接成为 shell 的退出码 ——
# 正常退出（0）时终端窗口会自己关掉，真出错时才留着让你看 traceback。
# "$@" 是为了能 ./启动.command --port 8766 这样临时换个端口。
exec "$PY" -m rapp.app "$@"
