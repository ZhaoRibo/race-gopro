#!/usr/bin/env python3
"""
race-gopro —— GoPro 卡丁车遥测分析
================================

用法示例
--------
  # 完整分析（圈速 + 报表 + 图表 + 网页看板）
  python analyze.py GX010123.MP4

  # 再加上带 HUD 的叠加视频
  python analyze.py GX010123.MP4 --overlay

  # 只看前两圈的开视频，省时间
  python analyze.py GX010123.MP4 --overlay --overlay-range 0,90

  # 没有视频？先用合成数据看看输出长什么样
  python analyze.py --demo

  # 检查整条流水线是否正确（含标定精度自检）
  python analyze.py --selftest

  # 看看视频里到底有哪些遥测流（排查问题用）
  python analyze.py GX010123.MP4 --list-streams

结果默认放在**源视频旁边**的 `<视频名>_out/` 目录（用 `-o` 可改）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rapp import (
    analysis,
    charts,
    dashboard,
    demo,
    gpmf,
    laps,
    overlay,
    report,
    serve,
    telemetry,
)


def _parse_range(text: str) -> tuple[float, float]:
    try:
        a, b = text.split(",")
        return float(a), float(b)
    except Exception as exc:
        raise argparse.ArgumentTypeError("格式应为 开始秒,结束秒，例如 0,90") from exc


def _parse_latlon(text: str) -> tuple[float, float]:
    try:
        a, b = text.split(",")
        return float(a), float(b)
    except Exception as exc:
        raise argparse.ArgumentTypeError("格式应为 纬度,经度，例如 31.2304,121.4737") from exc


OUT_SUFFIX = "_out"
"""输出目录名 = 视频名 + 这个后缀，放在**源视频所在目录**里。

为什么不再固定放 `./out`：用户的视频往往散在不同拍摄日 / 不同移动硬盘里，
固定一个相对路径的话，处理第二个盘不是撞名就是得手动改参数。
放在视频旁边就永远找得到，也不会互相覆盖。

带后缀而不是直接用视频名：明确标出"这是生成的"，不会和用户自己建的同名
文件夹混在一起，也好一把删掉（`rm -rf *_out`）。
"""


def resolve_outdir(
    want: Path | None, video_path: Path | None, source_name: str
) -> Path:
    """
    决定结果放哪里，返回**这次分析的输出目录**（已保证存在且可写）。

    优先级：
        --out 给了       → <--out>/<视频名>
        有源视频         → <源视频所在目录>/<视频名>_out
        --demo（无视频） → ./out/<名字>
    """
    if want is not None:
        out = want / source_name
    elif video_path is None:
        out = Path("out") / source_name
    else:
        out = video_path.resolve().parent / f"{video_path.stem}{OUT_SUFFIX}"
        # 源目录未必可写：只读挂载、相机 SD 卡、别人的共享盘都可能拦下来。
        # 与其跑到一半才报错，不如现在就试一下写权限，不行就退回 ./out。
        try:
            out.mkdir(parents=True, exist_ok=True)
            probe = out / ".write-test"
            probe.touch()
            probe.unlink()
        except OSError:
            fallback = Path("out") / source_name
            print(
                f"  ⚠ 视频所在目录写不进去，结果改放 {fallback.resolve()}",
                file=sys.stderr,
            )
            out = fallback
    out.mkdir(parents=True, exist_ok=True)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="race-gopro",
        description="从 GoPro 视频中提取圈速、G 值等专业赛车遥测数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("video", nargs="*", type=Path, help="GoPro 导出的 MP4 文件（可给多个，但只取第一个）")
    p.add_argument("-o", "--out", type=Path, default=None,
                   help=f"输出根目录。默认放在源视频旁边的 <视频名>{OUT_SUFFIX}/，"
                        "这样视频散在不同盘 / 不同文件夹时结果会跟着走；--demo 时默认 ./out")
    p.add_argument("--demo", action="store_true", help="用合成数据演示，不需要真实视频")

    p.add_argument("--gate", type=_parse_latlon, metavar="纬,经",
                   help="手动指定起点线：直接给经纬度（从赛道图上读坐标用）")
    p.add_argument("--gate-time", type=float, metavar="秒",
                   help="手动指定起点线：视频第 N 秒车正好过线（最容易上手）")
    p.add_argument("--gate-index", type=int, metavar="N",
                   help="从 --list-gates 的候选表里挑第 N 条（编号从 1 开始）")
    p.add_argument("--sectors", type=int, default=3, help="分段数量（默认 3）")
    p.add_argument("--grid-step", type=float, default=1.0, help="圈间对比的距离网格步长，米（默认 1.0）")
    p.add_argument("--min-laps", type=int, default=2, help="至少要有多少圈才认为识别成功（默认 2）")

    p.add_argument("--no-charts", action="store_true", help="不生成 matplotlib 图表")
    p.add_argument("--no-dashboard", action="store_true", help="不生成网页看板")
    p.add_argument("--no-csv", action="store_true", help="不导出 CSV")

    p.add_argument("--overlay", action="store_true", help="生成带 HUD 叠加的视频（耗时较长）")
    p.add_argument("--overlay-range", type=_parse_range, metavar="开始,结束",
                   help="只给这段时间叠加 HUD，单位秒")
    p.add_argument("--overlay-fps", type=float, default=15.0, help="HUD 刷新率，默认 15（越低越快）")
    p.add_argument("--overlay-crf", type=int, default=20, help="HUD 视频画质，越小越清晰（默认 20）")
    p.add_argument("--overlay-preset", default="medium", help="x264 预设，默认 medium")
    p.add_argument("--no-trace", action="store_true", help="HUD 里不画速度曲线小图")

    p.add_argument("--serve", action="store_true",
                   help="跑完不退出，把看板挂到本地端口上：这样在看板里能选中某一圈"
                        "一键生成 HUD 视频（浏览器的沙箱里调不了 ffmpeg，必须有这个服务）")
    p.add_argument("--serve-port", type=int, default=8765, metavar="N",
                   help="本地服务端口，默认 8765")

    p.add_argument("--list-streams", action="store_true", help="只列出视频里的遥测流，不做分析")
    p.add_argument("--list-gates", action="store_true",
                   help="只列出候选起点线（按直道优先排序），不做分析")
    p.add_argument("--selftest", action="store_true", help="用合成数据自检整条流水线")
    p.add_argument("-q", "--quiet", action="store_true", help="少打印一些中间信息")

    return p


def list_streams(video: Path) -> int:
    """打印 MP4 里的轨道信息和解析出的遥测流，用于排查问题。"""
    print(f"文件：{video}")
    print("\n— 容器里的轨道 —")
    for s in gpmf.probe_streams(video):
        handler = (s.get("tags") or {}).get("handler_name", "")
        print(f"  #{s['index']:<3} {s.get('codec_type'):<8} {s.get('codec_name', ''):<10} {handler}")

    print("\n— 解析出的遥测流 —")
    streams = gpmf.read_streams(video)
    if not streams:
        print("  （空）该视频里没有可解析的 GPMF 遥测数据。")
        return 1
    for name, st in sorted(streams.items()):
        print(f"  {name:<8} {st.times.size:>8} 个采样点  "
              f"{st.rate:>7.2f} Hz  单位={st.units or '?':<10} "
              f"值域=[{st.values.min()}, {st.values.max()}]")
    return 0


def list_gates(video: Path) -> int:
    """列出所有合格的候选起点线，供用户挑选。"""
    tel = telemetry.load(video, verbose=False)
    try:
        cands = laps.search_gates(tel)
    except RuntimeError as exc:
        print(f"搜索失败：{exc}", file=sys.stderr)
        return 1

    if not cands:
        print("没有找到任何合格的候选起点线。可能是本次录制里有效圈数不足 2 圈，"
              "或者 GPS 信号太差。", file=sys.stderr)
        return 1

    print(f"文件：{video}")
    print(f"全程里程约 {float(tel.dist[-1]):.0f} m\n")
    report.print_gate_candidates(cands)

    # 顺便说明自动搜索会选哪一条，方便对比
    try:
        auto = laps.find_gate(tel)
    except RuntimeError:
        return 0
    hit = laps._nearest_candidate(auto, cands)
    if hit is not None:
        rank = hit[0]
        print(f"\n  ⚠ 自动搜索会选候选 #{rank}（策略：偏僻度达标的前提下**优先直道**、"
              f"再取过线最早的那条），不是排第一的 #1。")
    else:
        print("\n  ⚠ 自动搜索选中位置离候选表里每一条都超过 50 m。")
    print("  想换线：加 --gate-index N 选某一条，或加 --gate-time 秒 自己指定过线时刻。")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # ------------------------------------------------------------------
    # 自检模式
    # ------------------------------------------------------------------
    if args.selftest:
        from tests import test_gpmf

        ok_parser, lines_parser = test_gpmf.run()
        ok_demo, lines_demo = demo.selftest()

        print()
        print("\n".join(lines_parser))
        print()
        print("\n".join(lines_demo))
        print()
        print("─" * 60)
        print(f"  GPMF 二进制解析器 : {'通过 ✓' if ok_parser else '未通过 ✗'}")
        print(f"  G 值提取          : {'通过 ✓' if ok_demo else '未通过 ✗'}")
        print("─" * 60)
        return 0 if (ok_parser and ok_demo) else 1

    # ------------------------------------------------------------------
    # 只看遥测流
    # ------------------------------------------------------------------
    if args.list_streams:
        if not args.video:
            print("请指定要检查的视频文件。", file=sys.stderr)
            return 2
        return list_streams(args.video[0])

    # ------------------------------------------------------------------
    # 只看候选起点线
    # ------------------------------------------------------------------
    if args.list_gates:
        if not args.video:
            print("请指定要检查的视频文件。", file=sys.stderr)
            return 2
        return list_gates(args.video[0])

    # 四种指定起点线的方式只能用一个（都不给就自动搜索）
    chosen = [
        ("--gate", args.gate is not None),
        ("--gate-time", args.gate_time is not None),
        ("--gate-index", args.gate_index is not None),
    ]
    if sum(1 for _, on in chosen if on) > 1:
        names = "、".join(n for n, on in chosen if on)
        print(f"{names} 只能用其中一个，请去掉多余的。", file=sys.stderr)
        return 2

    # ------------------------------------------------------------------
    # 取数据
    # ------------------------------------------------------------------
    verbose = not args.quiet
    demo_session = None

    if args.demo:
        print("使用合成数据（演示赛道）—— 不需要真实视频。")
        demo_session = demo.make(n_laps=8, verbose=False)
        tel = demo_session.telemetry
        source_name = "demo"
        video_path: Path | None = None
        print(tel.summary())
        if tel.gfield:
            print("\n— G 值提取 —")
            print(tel.gfield.describe())
    else:
        if not args.video:
            print("请提供 GoPro 视频文件，或加 --demo 用合成数据体验。", file=sys.stderr)
            return 2
        if len(args.video) > 1:
            print(
                "注意：检测到多个视频文件。GoPro 长时间录制会把视频切成多段，"
                "每段的遥测时间轴都从 0 开始，无法直接拼在一起。\n"
                "      本工具只分析第一个文件。要分析完整场次，请先用 ffmpeg 合并：\n"
                "        printf \"file '%s'\\n\" GX01*.MP4 > list.txt\n"
                "        ffmpeg -f concat -safe 0 -i list.txt -c copy merged.MP4\n",
                file=sys.stderr,
            )
        video_path = args.video[0]
        if not video_path.exists():
            print(f"找不到文件：{video_path}", file=sys.stderr)
            return 2
        tel = telemetry.load(video_path, verbose=verbose)
        source_name = video_path.stem

    # ------------------------------------------------------------------
    # 分析
    # ------------------------------------------------------------------
    lapset = laps.compute_lapset(
        tel,
        gate_latlon=args.gate,
        gate_time=args.gate_time,
        gate_index=args.gate_index,
        sectors=args.sectors,
        grid_step=args.grid_step,
        verbose=verbose,
    )
    sa = analysis.analyze(lapset, verbose=verbose)

    outdir = resolve_outdir(args.out, video_path, source_name)
    outdir.mkdir(parents=True, exist_ok=True)
    # 早早就把落点报出来：HUD 那段可能要跑好几分钟，用户想知道东西会去哪儿
    if verbose:
        print(f"\n输出目录：{outdir}")

    # 输出按用途分类，避免十几个文件平铺在一层里找不着北：
    #   charts/   图表
    #   tables/   数据表（CSV + JSON）
    #   最外层    只留"入口文件" dashboard.html（以及可选的 HUD 视频）——
    #             用户打开输出目录第一眼就该看到它，而不是在一堆 CSV 里翻。
    charts_dir = outdir / "charts"
    tables_dir = outdir / "tables"

    report.print_report(sa)

    tables_written: list[Path] = []
    if not args.no_csv:
        tables_written = report.export_csv(sa, tables_dir)
        tables_written.append(report.export_json(sa, tables_dir / "analysis.json"))

    figs: list[Path] = []
    if not args.no_charts:
        figs = charts.make_all(sa, charts_dir)

    html: Path | None = None
    if not args.no_dashboard:
        # hud_pad 只是给页面显示用的，实际出片以 serve.py 的常量为准
        html = dashboard.build(sa, outdir / "dashboard.html", hud_pad=serve.PAD_SECONDS)

    # ------------------------------------------------------------------
    # HUD 叠加视频
    # ------------------------------------------------------------------
    out_video: Path | None = None
    if args.overlay:
        if video_path is None:
            # 演示模式下没有源视频，临时生成一段测试画面
            from rapp.demo import make_test_video

            video_path = outdir / "demo_source.mp4"
            print("\n演示模式下需要一段源视频，正在用 ffmpeg 生成测试画面…")
            make_test_video(video_path, duration=min(40.0, tel.duration), fps=30.0)
            print(f"  已生成 {video_path}")

        out_video = outdir / f"{source_name}_hud.mp4"
        overlay.burn(
            video_path,
            sa,
            out_video,
            fps=args.overlay_fps,
            t_range=args.overlay_range,
            crf=args.overlay_crf,
            preset=args.overlay_preset,
            show_trace=not args.no_trace,
            verbose=True,
        )

    # ------------------------------------------------------------------
    # 输出目录一览
    # ------------------------------------------------------------------
    # 只列"该看哪个"，不把十几个文件名全铺出来 —— 目录结构已经说明了一切。
    print(f"\n全部输出位于：{outdir}")
    if html is not None:
        print("  dashboard.html   ← 双击打开，圈速 / G 值 / 弯道分析全在里面")
    if figs:
        print(f"  charts/          {len(figs)} 张图（想单独看大图就用这些）")
    if tables_written:
        print(f"  tables/          {len(tables_written)} 个数据文件（Excel / pandas 可直接打开）")
    if out_video is not None:
        print(f"  {out_video.name:<17}带 HUD 的叠加视频")

    # ------------------------------------------------------------------
    # 本地服务（可选，最后跑）：让看板上的「生成 HUD 视频」按钮能用
    # ------------------------------------------------------------------
    if args.serve:
        if video_path is None:
            print("\n--serve 需要源视频（--demo 没有可叠加的视频，加 --overlay 也不行）。",
                  file=sys.stderr)
            return 2
        if html is None:
            print("\n--serve 需要看板文件，但加了 --no-dashboard。去掉它再试。", file=sys.stderr)
            return 2
        return serve.run(
            sa, outdir, video_path,
            port=args.serve_port,
            fps=args.overlay_fps,
            crf=args.overlay_crf,
            preset=args.overlay_preset,
            show_trace=not args.no_trace,
            # 看板里改计时线时要按同一套参数重算，所以这些也得传过去
            sectors=args.sectors,
            grid_step=args.grid_step,
            write_csv=not args.no_csv,
            write_charts=not args.no_charts,
            write_dashboard=not args.no_dashboard,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
