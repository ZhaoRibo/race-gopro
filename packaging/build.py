#!/usr/bin/env python3
"""
一键打包：下 ffmpeg → PyInstaller → 压成可以直接发 Release 的 zip
==============================================================

在仓库根目录跑（Windows 上把 `.venv/bin/python` 换成 `.venv\\Scripts\\python`）：

    .venv/bin/python packaging/build.py

产物在 `dist/`：

| 平台    | 产物                                     | 用户拿到之后                       |
| ------- | ---------------------------------------- | ---------------------------------- |
| macOS   | `race-gopro-macos-arm64.zip`             | 解压 → 双击「卡丁车遥测分析.app」 |
| Windows | `race-gopro-windows-x64.zip`             | 解压 → 双击 `race-gopro.exe`    |

**Windows 的包必须在 Windows 上构建** —— PyInstaller 不做交叉编译。所以正式发布
走 `.github/workflows/build-app.yml`：推一个 `v*` 标签，两个平台各跑一次，
出来的包自动挂到 Release 上。

ffmpeg 从哪来：各平台的**静态**构建（自己带齐了所有库，拷过去就能跑）。
千万别用 Homebrew 装的那个 —— 它是动态链接的，拷进包里换台机器就废了。
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "packaging"
FFMPEG_DIR = PKG / "ffmpeg"
DIST = ROOT / "dist"

VERSION = "0.1.0"
"""版本号。改这里就够了（会写进 macOS 的 Info.plist 和产物文件名）。"""

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
"""下载时带的 User-Agent（有些站点会拦不认识客户端的连接）。"""

_CURL = shutil.which("curl") or "curl"
"""用 curl 下载，不用 urllib。

踩过的坑：urllib 连 osxexperts 的 TLS 握手直接被对端断掉
（`SSL: UNEXPECTED_EOF_WHILE_READING`，带浏览器 UA 也没用），而同一个地址
curl 一点事没有。构建机器上（macOS / Windows / Linux 的镜像）都自带 curl，
换过去最省事，顺带还白拿了重试、重定向、代理这些现成的好处。
"""


def _fetch_text(url: str) -> str:
    """取一段文本（比如网页），失败直接报出来。"""
    req = subprocess.run([_CURL, "-fsSL", "--retry", "3", "--retry-delay", "2",
                          "--connect-timeout", "30", "-A", _UA, url],
                         capture_output=True, text=True)
    if req.returncode != 0:
        raise RuntimeError(f"取不到 {url}\n{(req.stderr or '').strip()}")
    return req.stdout


# ==========================================================================
# 第一步：把 ffmpeg / ffprobe 弄到手
# ==========================================================================
def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"  ✓ 已经有 {dest.name}（{dest.stat().st_size / 1e6:.0f} MB），跳过下载")
        return dest
    print(f"  ↓ {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)   # curl 不会自己建目录
    tmp = dest.with_suffix(dest.suffix + ".part")
    # -sS 很关键：不加的话 curl 的进度条会和真正的错误一起汇到 stderr，
    # 报错时只截前 300 字就是一堆进度条，看不到原因。
    req = subprocess.run([_CURL, "-fLsS", "--retry", "3", "--retry-delay", "2",
                          "--connect-timeout", "30", "-A", _UA,
                          "-o", str(tmp), url], capture_output=True, text=True)
    if req.returncode != 0:
        raise RuntimeError(f"下载 {url} 失败\n{(req.stderr or '').strip()}")
    tmp.replace(dest)
    print(f"  ✓ {dest.name}（{dest.stat().st_size / 1e6:.0f} MB）")
    return dest


def _extract_from_zip(archive: Path, wanted: dict[str, Path]) -> None:
    """
    从 zip 里挑出 ffmpeg / ffprobe。

    不写死压缩包内部的路径：各家的目录结构不一样
    （有的在根目录、有的在 `ffmpeg-7.0/bin/` 下面），
    所以按**文件名**找，找到就停 —— 版本号变了也不会挂。
    """
    want = {k + ("." + "exe" if os.name == "nt" else ""): v for k, v in wanted.items()}
    remaining = dict(want)
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = Path(info.filename).name
            if name in remaining and "/__MACOSX/" not in info.filename:
                dest = remaining.pop(name)
                with zf.open(info) as src, dest.open("wb") as out:
                    shutil.copyfileobj(src, out)
                dest.chmod(0o755)
                print(f"  ✓ {name} → {dest.relative_to(ROOT)}"
                      f"（{dest.stat().st_size / 1e6:.0f} MB）")
            if not remaining:
                break
    if remaining:
        raise RuntimeError(f"{archive.name} 里没找到 {list(remaining)}，"
                           "下载源可能改结构了，去 packaging/build.py 里改一下")


def _osx_experts_sources() -> list[tuple[str, str]]:
    """
    Apple Silicon 的静态包。

    文件名带版本号（`ffmpeg9arm.zip`），所以去首页现找当前是哪个版本 ——
    写死的话人家发个新版我们就构建不出来了。
    """
    url = "https://www.osxexperts.net/"
    print(f"  查一下 osxexperts 现在提供哪个版本：{url}")
    html = _fetch_text(url)
    out = []
    for name in ("ffmpeg", "ffprobe"):
        vers = [int(v) for v in re.findall(rf"{name}(\d+)arm\.zip", html)]
        if not vers:
            raise RuntimeError(f"osxexperts 首页没找到 {name} 的 arm64 包；"
                               "去 packaging/build.py 里改下载来源")
        ver = max(vers)
        out.append((name, f"https://www.osxexperts.net/{name}{ver}arm.zip"))
        print(f"  → {name}：版本 {ver}")
    return out


def _evermeet_sources() -> list[tuple[str, str]]:
    """Intel Mac 的静态包。evermeet 的 getrelease 地址永远指向最新版，不用猜。"""
    return [
        ("ffmpeg", "https://evermeet.cx/ffmpeg/getrelease/zip"),
        ("ffprobe", "https://evermeet.cx/ffmpeg/getrelease/ffprobe/zip"),
    ]


def _windows_sources() -> list[tuple[str, str]]:
    """Windows 的静态包。gyan.dev 的 essentials 构建带 libx264（出片要用）。"""
    return [
        ("both", "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"),
    ]


def fetch_ffmpeg() -> None:
    FFMPEG_DIR.mkdir(parents=True, exist_ok=True)
    suffix = ".exe" if os.name == "nt" else ""
    have = all((FFMPEG_DIR / (n + suffix)).exists() for n in ("ffmpeg", "ffprobe"))
    if have:
        print("ffmpeg / ffprobe 已经下好了，跳过")
        return

    print("准备 ffmpeg（第一次要下 100 MB 左右，之后会缓存）")
    if os.name == "nt":
        srcs = _windows_sources()
    elif sys.platform == "darwin" and platform.machine() == "arm64":
        srcs = _osx_experts_sources()
    elif sys.platform == "darwin":
        srcs = _evermeet_sources()
    else:
        raise SystemExit("这个平台没有现成的静态 ffmpeg 来源，"
                         "去 packaging/build.py 里加一条，或者手动把 "
                         "ffmpeg / ffprobe 放进 packaging/ffmpeg/")

    for name, url in srcs:
        archive = _download(url, PKG / "cache" / (name + ".zip"))
        if name == "both":
            _extract_from_zip(archive, {"ffmpeg": FFMPEG_DIR / ("ffmpeg" + suffix),
                                        "ffprobe": FFMPEG_DIR / ("ffprobe" + suffix)})
        else:
            _extract_from_zip(archive, {name: FFMPEG_DIR / (name + suffix)})

    for n in ("ffmpeg", "ffprobe"):
        p = FFMPEG_DIR / (n + suffix)
        if not p.exists():
            raise SystemExit(f"少了 {p}")
        req = subprocess.run([str(p), "-version"], capture_output=True, text=True)
        if req.returncode != 0:
            raise SystemExit(f"{p} 跑不起来，包可能不匹配当前平台：{req.stderr[:200]}")
        line = (req.stdout or "").splitlines()[0]
        print(f"  ✓ {line}")
        if n == "ffmpeg" and "enable-libx264" not in req.stdout:
            print("  ⚠ 这份 ffmpeg 没有 libx264 —— HUD 出片会失败。"
                  "换一个带 GPL/x264 的构建。")


def warm_matplotlib_cache() -> None:
    """
    提前把 matplotlib 的字体缓存建好，打进包里。

    不这么做的话，用户第一次启动会卡十几秒：matplotlib 第一次 import 要扫一遍
    系统字体建缓存，而这段时间屏幕上什么都没有 —— 看起来就是"双击了没反应"。
    构建时建好、运行时拷到用户目录，这十几秒就没了。
    """
    out = PKG / "matplotlib"
    if out.is_dir() and any(out.iterdir()):
        print("matplotlib 字体缓存已经建好了，跳过")
        return
    out.mkdir(parents=True, exist_ok=True)
    print("预先建好 matplotlib 字体缓存（一次性，要十几秒）…")
    env = dict(os.environ, MPLCONFIGDIR=str(out))
    req = subprocess.run(
        [sys.executable, "-c",
         "import matplotlib; from matplotlib import font_manager as fm; "
         "fm.fontManager; print('字体数：', len(fm.fontManager.ttflist))"],
        env=env, capture_output=True, text=True)
    if req.returncode != 0:
        print(f"  ⚠ 没建成（不致命，用户首次启动会慢一次）：{req.stderr.strip()[:200]}")
        return
    print("  " + (req.stdout or "").strip())
    print(f"  ✓ {', '.join(sorted(p.name for p in out.iterdir()))}")


# ==========================================================================
# 第二步：PyInstaller
# ==========================================================================
def run_pyinstaller(*, clean: bool) -> None:
    if importlib.util.find_spec("PyInstaller") is None:
        raise SystemExit("先装打包工具：\n"
                         "  .venv/bin/pip install -r packaging/requirements.txt\n"
                         "（Windows：.venv\\Scripts\\pip install -r packaging\\requirements.txt）")

    if clean:
        shutil.rmtree(ROOT / "build", ignore_errors=True)
        if DIST.exists():
            # 只删自己那几样，别把别人放在 dist 里的东西一起扬了
            for item in list(DIST.iterdir()):
                if item.name.startswith(("race-gopro", "卡丁车遥测分析")):
                    shutil.rmtree(item, ignore_errors=True)
                    item.unlink(missing_ok=True)

    env = dict(os.environ, RACEGOPRO_VERSION=VERSION)
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm",
           "--distpath", str(DIST), "--workpath", str(ROOT / "build"),
           str(PKG / "race-gopro.spec")]
    print("\n" + " ".join(cmd))
    if subprocess.run(cmd, cwd=ROOT, env=env).returncode != 0:
        raise SystemExit("PyInstaller 失败了，往上翻它的输出")


# ==========================================================================
# 第三步：摆成一个可以直接发的目录，再压成 zip
# ==========================================================================
_README_TEXT = """卡丁车遥测分析 —— 怎么用
============================

【第一次打开】

  macOS：右键点「卡丁车遥测分析.app」→「打开」→ 再点一次「打开」。
         （应用没有签名，直接双击会被系统拦下来说"来自身份不明的开发者"。
           这样放行一次之后，以后就能直接双击了。）

  Windows：双击 race-gopro\\race-gopro.exe。
           如果弹出蓝色的"Windows 已保护你的电脑"，点「更多信息」→「仍要运行」。

【然后】

  1. 浏览器会自己弹出来（没弹就手动打开 http://127.0.0.1:8765/ ）
  2. 点「选择视频文件…」，挑一个 GoPro **直接导出**的 MP4（别用剪辑软件导出的，
     那些会把遥测数据丢掉）
  3. 等它跑完（一个 10 分钟的视频大概几十秒），会自动跳到看板

  结果写在**视频旁边**的 <视频名>_out/ 文件夹里：看板、图表、数据表、HUD 视频。

  看板里能做的事：勾选看哪几圈、拖动滑动条改起点线、给某一圈或整段生成
  带 HUD 的叠加视频。

【关掉】

  关掉浏览器页面，程序就退出了。出片正在跑的话会等它写完再退。

【出问题看日志】

  macOS  ：~/Library/Application Support/race-gopro/app.log
           （在访达里按 Command+Shift+G ，把上面这行粘进去）
  Windows：%APPDATA%\\race-gopro\\app.log

【这个包是没签名的】

  不是病毒，是作者没买代码签名证书（苹果一年 99 美元）。介意的话可以自己从
  源码跑：https://github.com/
"""


def _write_notices(stage: Path, readme_name: str) -> None:
    (stage / readme_name).write_text(_README_TEXT, encoding="utf-8")
    notice = PKG / "THIRD-PARTY.txt"
    if notice.exists():
        shutil.copy2(notice, stage / "THIRD-PARTY.txt")


def make_zip() -> Path:
    """
    把要发的东西摆进一个目录再压成 zip：

        macOS  ：卡丁车遥测分析-0.1.0/   ← 解压后就是一个文件夹，不会散落一地
        Windows：race-gopro-0.1.0/

    里面放着应用本体 + 使用说明 + 第三方声明。用户解压、双击、完事。
    """
    if sys.platform == "darwin":
        app = DIST / "race-gopro.app"
        if not app.exists():
            raise SystemExit(f"没看到 {app}")
        # 改个中文名：Finder 里显示的就是这个（包里的可执行文件仍是 ASCII 的，
        # 改外壳目录名不影响启动）
        pretty = DIST / "卡丁车遥测分析.app"
        if pretty.exists():
            shutil.rmtree(pretty)
        app.rename(pretty)
        shutil.rmtree(DIST / "race-gopro", ignore_errors=True)   # COLLECT 的中间产物

        stage = DIST / f"卡丁车遥测分析-{VERSION}"
        shutil.rmtree(stage, ignore_errors=True)
        stage.mkdir(parents=True)
        pretty.rename(stage / pretty.name)
        _write_notices(stage, "使用说明.txt")

        out = DIST / f"race-gopro-macos-{platform.machine()}.zip"
        out.unlink(missing_ok=True)
        # 必须用 ditto：.app 里有一堆符号链接和权限位，普通 zip 会把它们压坏
        subprocess.run(["ditto", "-c", "-k", "--sequesterRsrc", "--keepParent",
                        str(stage), str(out)], check=True)
        shutil.rmtree(stage, ignore_errors=True)
        return out

    folder = DIST / "race-gopro"
    if not folder.exists():
        raise SystemExit(f"没看到 {folder}")
    # Windows 这边目录名用 ASCII：有的解压工具不认 zip 里的 UTF-8 中文名
    stage = DIST / f"race-gopro-{VERSION}"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    folder.rename(stage / folder.name)
    _write_notices(stage, "README.txt")

    out = DIST / "race-gopro-windows-x64.zip"
    out.unlink(missing_ok=True)
    shutil.make_archive(str(out.with_suffix("")), "zip", root_dir=DIST,
                        base_dir=stage.name)
    shutil.rmtree(stage, ignore_errors=True)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="把 race-gopro 打包成独立应用")
    ap.add_argument("--skip-ffmpeg", action="store_true",
                    help="跳过下载 ffmpeg（用 packaging/ffmpeg/ 里已有的）")
    ap.add_argument("--no-clean", action="store_true", help="不清 build/dist")
    args = ap.parse_args(argv)

    print(f"平台：{sys.platform} / {platform.machine()}   版本：{VERSION}\n")
    if not args.skip_ffmpeg:
        fetch_ffmpeg()
    warm_matplotlib_cache()
    print()
    run_pyinstaller(clean=not args.no_clean)
    out = make_zip()
    print(f"\n✓ 打包完成：{out.relative_to(ROOT)}"
          f"（{out.stat().st_size / 1e6:.0f} MB）")
    print("  发 Release 时把这个文件传上去就行。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
