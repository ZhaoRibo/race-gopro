"""
应用图标检查
============

图标是代码画出来的（`packaging/icon.py`），不是仓库里的一张图 —— 所以它能悄悄
画坏而不被发现：比如某个尺寸渲染成空白、或者 .ico 里少了几种尺寸。
Windows 任务栏、macOS 的「访达」对尺寸的要求各不相同，少一个就会出现
"图标糊成一团"或者干脆不显示。

这个测试盯三件事：
  1. 每个需要的尺寸都能画出来，而且**不是空白**（有足够多的不透明像素）
  2. 画出来的图案不是一片纯色（真的画了旗子，不是只填了个底色）
  3. .ico 里确实打包了多种尺寸（Windows 会按显示场景挑）

运行方式：
    .venv/bin/python -m tests.test_icon
    .venv/bin/python -m pytest tests/
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packaging"))

import icon  # noqa: E402  ← packaging/ 不在包里，得手动加进 sys.path


def test_every_size_renders_and_is_not_blank() -> None:
    for size in (16, 32, 64, 128, 256):
        img = icon.render(size)
        assert img.size == (size, size), f"{size} 渲出来的尺寸是 {img.size}"
        alpha = img.getchannel("A").tobytes()
        ratio = sum(1 for b in alpha if b > 200) / (size * size)
        assert ratio > 0.5, f"{size}×{size} 只有 {ratio:.0%} 是不透明的，画丢了？"


def test_icon_has_actual_content() -> None:
    """不能是一片纯色 —— 至少得有深浅两种像素（棋盘格 + 旗杆）。"""
    colors = icon.render(256).convert("RGB").getcolors(maxcolors=1 << 20)
    assert colors and len(colors) > 50, (
        f"只有 {len(colors) if colors else 0} 种颜色，看着像没画出东西")


def test_ico_contains_multiple_sizes(tmp_path: Path) -> None:
    """.ico 必须是多尺寸的，否则 Windows 缩放时会糊。"""
    from PIL import Image

    out = icon.save_ico(tmp_path / "t.ico")
    with Image.open(out) as ico:
        sizes = set(ico.info.get("sizes") or [])
    assert len(sizes) >= 4, f".ico 里只有 {len(sizes)} 种尺寸：{sizes}"
    assert (16, 16) in sizes and (256, 256) in sizes, f"缺少必要尺寸：{sizes}"


def main() -> int:
    failed = 0
    for size in (16, 32, 64, 128, 256):
        img = icon.render(size)
        alpha = img.getchannel("A").tobytes()
        ratio = sum(1 for b in alpha if b > 200) / (size * size)
        ok = img.size == (size, size) and ratio > 0.5
        print(f"{'✓' if ok else '✗'} {size}×{size}：不透明 {ratio:.0%}")
        failed += 0 if ok else 1

    got = icon.render(256).convert("RGB").getcolors(maxcolors=1 << 20)
    colors = len(got) if got else 0
    ok = colors > 50
    print(f"{'✓' if ok else '✗'} 颜色数 {colors}（纯色图标就是画丢了）")
    failed += 0 if ok else 1

    print()
    print("全部通过 ✓" if not failed else f"有 {failed} 项失败 ✗")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
