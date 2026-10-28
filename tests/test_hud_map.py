"""
HUD 实时轨迹小地图的回归测试
============================

小地图是**纯画出来的**：坐标算错了不会抛异常，只会「什么都没画」或者画到画面
外面去 —— 打开视频才发现，代价是几分钟的编码。所以这里用合成数据跑一遍，
盯四件事：

1. 每一帧都能定位到车（x / y 不是 NaN）；
2. 面板落在画面里，且不压住左上计时卡、右上速度曲线卡、右下时速；
3. 真的画出了轨迹：跑过的距离越长，面板里的青色轨迹像素越多；
4. 当前位置那个白点确实在面板里，而且随车移动。

运行方式：
    .venv/bin/python -m tests.test_hud_map
    .venv/bin/python -m pytest tests/
"""

from __future__ import annotations

import numpy as np

from rapp import analysis, demo, laps, overlay

# 16:9 是打包版默认会遇到的画幅；4:3 是样例素材 GX010047 的画幅（2704×2028）。
# 两种都要成立：小地图的位置是按「右侧一列」算的，不是按 16:9 的空当算的。
_SIZES = ((1920, 1080), (2704, 2028))


def _session() -> analysis.SessionAnalysis:
    """跑一遍合成数据，得到和真实分析同构的 SessionAnalysis。"""
    data = demo.make(n_laps=4, verbose=False)
    lapset = laps.compute_lapset(data.telemetry, verbose=False)
    return analysis.analyze(lapset, verbose=False)


def _stats(img, view: overlay._TrackView) -> tuple[int, tuple[float, float]]:
    """统计面板里的青色轨迹像素数、白点中心（面板局部坐标）。"""
    arr = np.asarray(img)
    sub = arr[view.y:view.y + view.h, view.x:view.x + view.w]
    cyan = (sub[..., 0] < 90) & (sub[..., 1] > 170) & (sub[..., 2] > 200)
    white = ((sub[..., 0] > 225) & (sub[..., 1] > 225)
             & (sub[..., 2] > 225) & (sub[..., 3] > 200))
    ys, xs = np.nonzero(white)
    center = (float(xs.mean()), float(ys.mean())) if xs.size else (float("nan"),) * 2
    return int(cyan.sum()), center


def run() -> tuple[bool, list[str]]:
    ok = True
    lines: list[str] = []

    def check(cond, msg: str) -> None:
        nonlocal ok
        lines.append(("✓ " if cond else "✗ ") + msg)
        ok = ok and bool(cond)

    sa = _session()
    # 第 2 圈：第 1 圈在合成数据里通常被标成出场圈，拿它测容易撞上边界
    lap = sa.lapset.laps[1]
    fonts = overlay._Fonts()

    view = overlay.make_track_view(sa, *_SIZES[0], fonts)
    check(view is not None, "合成数据上构造出了小地图")
    if view is None:                                  # pragma: no cover
        return False, lines

    hud = overlay._hud_table(sa, 15.0, lap.t_start, lap.t_end)
    inside = hud["lap_index"] > 0
    check(bool(np.all(np.isfinite(hud["x"][inside]))),
          "本圈每一帧都有车的平面位置（x / y 不是 NaN）")

    # ---- 版面：两种画幅下都要在画面里，且不压别的卡片 ----
    for w, h in _SIZES:
        v = view if (w, h) == _SIZES[0] else overlay.make_track_view(sa, w, h, fonts)
        assert v is not None
        s = h / 1080.0
        tag = f"{w}×{h}"
        check(v.x >= 0 and v.x + v.w <= w,
              f"{tag}：面板横向在画面以内（x={v.x}，宽 {v.w}）")
        check(v.y >= 0 and v.y + v.h <= h,
              f"{tag}：面板纵向在画面以内（y={v.y}，高 {v.h}）")
        check(v.y >= 38 * s + 196 * s,
              f"{tag}：面板在右上速度曲线卡下方")
        check(v.x >= 38 * s + 566 * s,
              f"{tag}：面板不压左上计时卡")
        check(v.y + v.h <= h - 322 * s,
              f"{tag}：面板不压右下时速（数字 + km/h 约占 322 px）")

    # ---- 内容：轨迹越跑越长、白点跟着走 ----
    n = hud["t"].size
    early_px, early_c = _stats(
        overlay.render_hud_frame(int(n * 0.10), hud, sa, *_SIZES[0], fonts, track=view),
        view)
    late_px, late_c = _stats(
        overlay.render_hud_frame(int(n * 0.85), hud, sa, *_SIZES[0], fonts, track=view),
        view)

    check(early_px > 20, f"起步阶段已经画出轨迹（{early_px} 个青色像素）")
    check(late_px > early_px * 2,
          f"跑过的距离越长，轨迹越长（{early_px} → {late_px} 个像素）")
    check(np.isfinite(early_c[0]) and np.isfinite(late_c[0]),
          "当前位置的白点画出来了（面板里有白点）")
    if np.isfinite(early_c[0]) and np.isfinite(late_c[0]):
        moved = float(np.hypot(late_c[0] - early_c[0], late_c[1] - early_c[1]))
        check(moved > 10.0, f"白点随车移动（{moved:.0f} px）")
        check(0 <= late_c[0] <= view.w and 0 <= late_c[1] <= view.h,
              "白点落在面板以内")

    return ok, lines


def test_hud_map() -> None:
    """pytest 入口。"""
    ok, lines = run()
    assert ok, "\n".join(lines)


def main() -> int:
    passed, lines = run()
    print("\n".join(lines))
    print()
    print("实时轨迹小地图：" + ("通过 ✓" if passed else "未通过 ✗"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
