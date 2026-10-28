"""
单文件离线看板的检查
====================

离线版是给 iPad / 发给朋友看的，坏了不会报错：打开只会看到"图表整块空白"
或者"表格没出来"。而这些症状在 Mac 上又不容易复现 —— Mac 上可能联网、
charts/ 目录也在旁边，链接随便就能打开。

所以这个测试盯四件事：

1. 图表真的被 base64 **内嵌**（不是链接到 charts/）；
2. 离线版不请求 CDN，联网版照旧走 CDN；
3. 出片 / 改计时线这些依赖本地服务的区块在离线版里被藏掉；
4. 内嵌脚本能过 `node --check`（语法坏的话整页的表格都不会出来）。

运行方式：
    .venv/bin/python -m tests.test_offline_html
    .venv/bin/python -m pytest tests/
"""

from __future__ import annotations

import base64
import shutil
import tempfile
from pathlib import Path

from rapp import analysis, dashboard, demo, laps

# 1×1 透明 PNG：测试只需要"能当成图片内嵌"，不需要真图
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
_PNG_B64 = base64.b64encode(_PNG).decode("ascii")


def run() -> tuple[bool, list[str]]:
    ok = True
    lines: list[str] = []

    def check(cond, msg: str) -> None:
        nonlocal ok
        lines.append(("✓ " if cond else "✗ ") + msg)
        ok = ok and bool(cond)

    data = demo.make(n_laps=4, verbose=False)
    sa = analysis.analyze(laps.compute_lapset(data.telemetry, verbose=False), verbose=False)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        (out / "charts").mkdir()
        for name in ("lap_times.png", "track_map.png"):
            (out / "charts" / name).write_bytes(_PNG)

        online, offline = dashboard.build_pair(sa, out, hud_pad=5.0)
        off = offline.read_text(encoding="utf-8")
        on = online.read_text(encoding="utf-8")
        both_exist = offline.exists() and online.exists()

    check(both_exist, "两份看板都生成了")
    check(off.count("data:image/png;base64,") == 2,
          f"图表按数量内嵌（{off.count('data:image/png;base64,')} 张）")
    check(_PNG_B64 in off, "内嵌的确实是图片原始字节（base64 可还原）")
    check("cdn.jsdelivr.net" not in off, "离线版不请求 CDN")
    check("cdn.jsdelivr.net" in on, "联网版照旧走 CDN")
    check('class="offline"' in off and 'class=""' in on, "body 的 offline 标记正确")
    check("const OFFLINE_MODE = true;" in off and "const OFFLINE_MODE = false;" in on,
          "页面脚本里的离线标记正确（离线时不报 Chart.js 加载失败）")
    check(".offline .chartsec" in off, "交互图表区块在离线版里被藏掉")
    check("body.offline #hudSec" in off and "body.offline .gatebar" in off,
          "出片 / 改计时线的区块在离线版里被藏掉")
    check('href="charts/' not in off, "离线版不再链接 charts/ 目录（单文件自足）")
    check("dashboard_offline.html" in on, "联网版列出离线版的下载链接")

    if shutil.which("node"):
        check(dashboard._check_inline_js(off) is None, "离线版内嵌脚本通过 node --check")
        check(dashboard._check_inline_js(on) is None, "联网版内嵌脚本通过 node --check")
    else:
        lines.append("· 没装 node，跳过脚本语法检查")

    return ok, lines


def test_offline_html() -> None:
    """pytest 入口。"""
    ok, lines = run()
    assert ok, "\n".join(lines)


def main() -> int:
    passed, lines = run()
    print("\n".join(lines))
    print()
    print("单文件离线版：" + ("通过 ✓" if passed else "未通过 ✗"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
