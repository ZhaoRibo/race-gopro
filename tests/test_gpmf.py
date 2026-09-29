"""
GPMF 解析器的回归测试
====================

为什么需要这个：真实 GoPro 文件不是随时都有的，而 GPMF 是私有二进制格式，
解析器里最容易出错的地方（字节序、容器嵌套、缩放系数、时间戳回绕）恰恰是
看代码看不出来的 —— 必须拿真实字节去跑。

所以这里**手工构造**一段 GPMF 字节流：完全按照 GoPro 的格式规范拼出来，
再喂给 `gpmf.parse_gpmf()` / `gpmf.read_streams()`，然后逐项核对解出来的
数值、采样率、单位、缩放是否正确。

构造出来的结构长这样（和真实文件一致）：

    DEVC ─┬─ DVNM "Camera"
          ├─ STRM ─┬─ STNM "ACCL"
          │        ├─ SCAL [1, 1]          → 换算系数 1.0
          │        ├─ SIUN "m/s^2"
          │        ├─ STMP 1000000         → 微秒
          │        └─ ACCL [10 组 int16×3]
          └─ STRM ─┬─ STNM "GPS5"
                   ├─ SCAL [10000000, 1000, 1000, 1000, 1000]
                   ├─ SIUN "deg,deg,m,m/s,m/s"
                   ├─ STMP 1000000
                   └─ GPS5 [10 组 int32×5]
    （然后整个 DEVC 再重复一次，模拟第二段采样）

运行方式：
    .venv/bin/python -m tests.test_gpmf        # 直接跑
    .venv/bin/python -m pytest tests/          # 用 pytest 跑
"""

from __future__ import annotations

import struct

import numpy as np

from rapp import gpmf


# ==========================================================================
# 手工构造 GPMF 字节流
# ==========================================================================
def _pad4(blob: bytes) -> bytes:
    """GPMF 的每个元素都按 4 字节对齐（不足补 0）。真实文件里能看到
    "Accelerometer"（13 字节）、"ZXY"（3 字节）后面都跟着补位的 0 字节。"""
    return blob + b"\x00" * ((-len(blob)) % 4)


def _elem(key: str, type_char: bytes, unit_size: int, body: bytes, repeat: int | None = None) -> bytes:
    """拼一个 GPMF 元素：4 字节名字 + 类型 + 单元大小 + 重复次数 + 载荷，末尾补齐到 4 字节。"""
    if repeat is None:
        repeat = len(body) // unit_size if unit_size else len(body)
    head = key.encode("ascii") + type_char + bytes([unit_size]) + struct.pack(">H", repeat)
    return _pad4(head + body)


def _container(key: str, payload: bytes, unit_size: int = 1) -> bytes:
    """拼一个容器元素。unit_size 给 0 可以测试解析器的兼容分支。"""
    head = key.encode("ascii") + b"?" + bytes([unit_size]) + struct.pack(">H", len(payload))
    return _pad4(head + payload)


def _str_elem(key: str, text: str) -> bytes:
    """字符串型元素（类型 'c'，以 \\0 结尾）。"""
    raw = text.encode("utf-8") + b"\x00"
    return _elem(key, b"c", 1, raw, repeat=len(raw))


def _int_elem(key: str, values: list[int], fmt: str = "i") -> bytes:
    """
    整型元素，大端序。

    注意 GPMF 的"单元大小"指的是**一个采样点占多少字节**，不是单个标量的字节数。
    像 SCAL 是两个 int32（分子 + 分母），就得算一个采样点、8 字节；
    写错了解析器会从中间截断，后面所有元素跟着错位。
    """
    type_of = {"b": b"b", "B": b"B", "h": b"s", "i": b"l", "I": b"L"}
    if fmt not in type_of:
        raise ValueError(f"不支持的格式字符: {fmt}")
    body = struct.pack(">" + fmt * len(values), *values)
    return _elem(key, type_of[fmt], len(body), body, repeat=1)


def _make_strm(name: str, scal: list[tuple[int, int]] | None, siun: str | None,
               stmp_us: int, payload_key: str, payload_body: bytes,
               unit_size: int, type_char: bytes, repeat: int) -> bytes:
    """拼一条完整的 STRM 容器。"""
    parts = [_str_elem("STNM", name)]
    if scal:
        flat = [v for pair in scal for v in pair]
        parts.append(_int_elem("SCAL", flat, "i"))
    if siun:
        parts.append(_str_elem("SIUN", siun))
    parts.append(_int_elem("STMP", [stmp_us], "I"))
    parts.append(_elem(payload_key, type_char, unit_size, payload_body, repeat=repeat))
    return _container("STRM", b"".join(parts))


def build_sample_gpmf() -> tuple[bytes, dict]:
    """
    造两段 DEVC，每段含 ACCL 和 GPS5，返回 (字节流, 真值字典)。

    两段是必要的：真实文件里 GoPro 会把同一个 DEVC 容器重复成千上万次，
    每段只装一小段采样，解析器必须跨容器把数据累积起来。
    """
    rng = np.random.default_rng(0)

    acc_n = 10  # 每块 ACCL 的采样点数
    gps_n = 10
    lat0, lon0 = 31.2304, 121.4737
    gps_scale = np.array([1e7, 1e7, 1e3, 1e3, 1e3])  # 原始整数 → 物理量的除数

    acc_blocks, gps_blocks, devcs = [], [], []
    for b in range(2):
        acc_vals = rng.integers(-20000, 20000, size=(acc_n, 3)).astype(np.int16)

        lat = lat0 + rng.normal(0, 1e-5, gps_n)
        lon = lon0 + rng.normal(0, 1e-5, gps_n)
        spd = np.abs(rng.normal(15.0, 1.0, gps_n))
        gps_vals = np.column_stack([lat, lon, np.full(gps_n, 12.5), spd, spd])

        acc_blocks.append(acc_vals)
        gps_blocks.append(gps_vals)

        stmp = 1_000_000 + b * 500_000  # 微秒；相差 0.5 s，每块 10 点 → 20 Hz

        strm_acc = _make_strm(
            "ACCL", [(1, 1)], "m/s^2", stmp, "ACCL",
            acc_vals.astype(">i2").tobytes(), 6, b"s", acc_n,
        )
        strm_gps = _make_strm(
            "GPS5",
            [(10_000_000, 1), (10_000_000, 1), (1_000, 1), (1_000, 1), (1_000, 1)],
            "deg,deg,m,m/s,m/s", stmp, "GPS5",
            np.round(gps_vals * gps_scale).astype(np.int32).astype(">i4").tobytes(), 20, b"l", gps_n,
        )
        devcs.append(_container("DEVC", _str_elem("DVNM", "Camera") + strm_acc + strm_gps))

    truth = {
        "acc": np.concatenate(acc_blocks),
        "gps": np.concatenate(gps_blocks),
        "n_acc": acc_n * 2,
        "n_gps": gps_n * 2,
    }
    return b"".join(devcs), truth


# ==========================================================================
# 断言
# ==========================================================================
def _check(ok: bool, msg: str, out: list[str], strict: list[bool]) -> None:
    out.append(("  ✓ " if ok else "  ✗ ") + msg)
    strict.append(ok)


def run(verbose: bool = True) -> tuple[bool, list[str]]:
    """跑完整套解析器检查，返回 (是否全部通过, 输出行)。"""
    out: list[str] = ["【GPMF 解析器自检】"]
    checks: list[bool] = []

    blob, truth = build_sample_gpmf()
    out.append(f"  构造的字节流长度: {len(blob)} 字节")

    # ---- 1. 树结构 ----
    roots = gpmf.parse_gpmf(blob)
    _check(len(roots) == 2 and all(r.key == "DEVC" for r in roots),
           f"顶层解析出 2 个 DEVC 容器（实际 {len(roots)} 个）", out, checks)
    _check(roots[0].child("DVNM") is not None, "DEVC 里能取到 DVNM 子元素", out, checks)
    strms = [c for c in roots[0].children if c.key == "STRM"]
    _check(len(strms) == 2, f"第一个 DEVC 里有 2 个 STRM（实际 {len(strms)} 个）", out, checks)

    # ---- 2. 流级解析：字节序、缩放、单位、时间戳 ----
    streams = gpmf.read_streams("dummy.mp4", gpmd_blob=blob)
    _check("ACCL" in streams and "GPS5" in streams,
           f"解出 ACCL 与 GPS5 两条流（实际 {sorted(streams)}）", out, checks)

    acc = streams.get("ACCL")
    if acc is not None:
        _check(acc.values.shape == (truth["n_acc"], 3),
               f"ACCL 形状应为 ({truth['n_acc']}, 3)，实际 {acc.values.shape}", out, checks)
        # 大端序最容易错：解错了数值会是完全随机的
        _check(bool(np.array_equal(acc.values, truth["acc"])),
               "ACCL 数值逐点一致（验证大端序读取正确）", out, checks)
        _check(acc.units == "m/s^2", f"ACCL 单位 = {acc.units!r}", out, checks)
        _check(abs(acc.rate - 20.0) < 0.01,
               f"ACCL 采样率应为 20.00 Hz，实际 {acc.rate:.3f} Hz", out, checks)
        # 时间戳要连续、单调
        _check(bool(np.all(np.diff(acc.times) > 0)), "ACCL 时间戳严格单调递增", out, checks)
        # STMP 记的是块首时刻（1_000_000 µs = 1.0 s），而且**不做归零** ——
        # 各条流共用同一个以录制开始为原点的时间轴，归零会制造出相对偏移
        _check(abs(float(acc.times[0]) - 1.0) < 1e-9,
               f"首采样点时间应为 1.0 s，实际 {float(acc.times[0]):.6f} s", out, checks)
        _check(abs(float(acc.times[-1]) - 1.95) < 1e-6,
               f"末采样点时间应为 1.95 s，实际 {float(acc.times[-1]):.6f} s", out, checks)

    gps_st = streams.get("GPS5")
    if gps_st is not None:
        phys = gps_st.physical()
        _check(phys.shape == (truth["n_gps"], 5),
               f"GPS5 形状应为 ({truth['n_gps']}, 5)，实际 {phys.shape}", out, checks)
        # 缩放系数如果方向搞反（乘而不是除），经纬度会跑出地球范围
        err = float(np.max(np.abs(phys[:, :4] - truth["gps"][:, :4])))
        _check(err < 1e-3, f"GPS5 缩放还原误差 {err:.2e}（应 < 1e-3）", out, checks)
        _check(bool(np.all(np.abs(phys[:, 0]) <= 90) and np.all(np.abs(phys[:, 1]) <= 180)),
               "还原出的经纬度落在合法范围内", out, checks)

    # ---- 3. 容器 unit_size = 0 的兼容分支 ----
    inner = _str_elem("STNM", "TEST")
    zero_container = _container("STRM", inner, unit_size=0)
    parsed_zero = gpmf.parse_gpmf(zero_container)
    _check(len(parsed_zero) == 1 and parsed_zero[0].child("STNM") is not None,
           "容器 unit_size=0 时仍能正确解析（部分固件会这么写）", out, checks)

    # ---- 4. STMP 回绕 ----
    # uint32 微秒在 2^32 处回绕（约 71.6 分钟）。这里跨过回绕点，
    # 期望解卷绕后仍然单调递增，且总时长算得对。
    wrapped = np.array([4_000_000_000, 4_294_000_000, 100_000_000, 700_000_000], dtype=np.float64)
    unwrapped = gpmf._unwrap_stmp(wrapped)
    _check(bool(np.all(np.diff(unwrapped) > 0)),
           "STMP 跨过 2^32 回绕点时能正确解卷绕（仍单调递增）", out, checks)
    expect = (2**32 + 700_000_000 - 4_000_000_000) / 1e6  # 微秒 → 秒
    got = float(unwrapped[-1] - unwrapped[0]) / 1e6
    _check(abs(got - expect) < 1e-6,
           f"回绕后的总时长应为 {expect:.6f} s，实际 {got:.6f} s", out, checks)

    ok = all(checks)
    return ok, out


def test_gpmf_roundtrip() -> None:
    """pytest 入口。"""
    ok, lines = run(verbose=False)
    assert ok, "\n".join(lines)


def test_unwrap_stmp() -> None:
    v = gpmf._unwrap_stmp(np.array([10, 20, 30], dtype=np.float64))
    assert list(v) == [10.0, 20.0, 30.0]


if __name__ == "__main__":
    passed, lines = run()
    print("\n".join(lines))
    print()
    print("解析器自检：" + ("通过 ✓" if passed else "未通过 ✗"))
    raise SystemExit(0 if passed else 1)
