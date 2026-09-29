"""
GoPro GPMF 遥测流的抽取与解析
=============================

背景知识
--------
GoPro 录视频时，会往 MP4 里塞一条**独立的元数据轨道**，名叫 `gpmd`
（MP4 里 handler_name 是 "GoPro MET"）。它不是视频也不是音频，是一条纯二进制的
传感器记录流，里面包含：

    GPS5 / GPS9   经纬度、海拔、速度（10Hz 或 18Hz）
    ACCL          三轴加速度计（200Hz，单位 m/s²）
    GYRO          三轴陀螺仪（200Hz，单位 deg/s 或 rad/s）
    GRAV          重力方向向量（相机坐标系，用来做姿态对齐）
    CORI          相机姿态四元数
    STMP          时间戳（微秒）

RaceChrono、Telemetry Overlay 这类专业圈速软件读的就是这条流。

GPMF 二进制格式
---------------
格式非常规整，每一层都是同一个"信封"：

    ┌──────────┬────────┬──────────┬────────────┬──────────────┐
    │ 4 字节名 │ 1 字节 │ 1 字节   │ 2 字节     │ N 字节       │
    │ "ACCL"   │ 类型码 │ 单元大小 │ 重复次数   │ 数据载荷     │
    │ (ASCII)  │ 's'    │ 6        │ 200        │ 200 × 6 字节 │
    └──────────┴────────┴──────────┴────────────┴──────────────┘

    载荷长度 = 单元大小 × 重复次数
    类型码 '?' 表示"这是个容器，里面还嵌套着别的元素"，就像 dict 里套 dict。
    所有数值都是**大端序**（big-endian）—— 和 Python 默认的小端 x86 相反，
    所以 numpy 读取时必须用 '>i2' 这种带 '>' 前缀的 dtype。

对照 Python 概念：GPMF 就是一坨嵌套的 dict / list，字段名固定 4 字节，
值有类型和形状。本模块做的就是把它翻译成 numpy 数组。
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# 类型码表：GPMF 类型字符 → (numpy/struct 格式字符, 单个标量字节数)
# --------------------------------------------------------------------------
_SCALAR_TYPES: dict[bytes, tuple[str, int]] = {
    b"b": ("i1", 1),  # int8
    b"B": ("u1", 1),  # uint8
    b"s": ("i2", 2),  # int16
    b"S": ("u2", 2),  # uint16
    b"l": ("i4", 4),  # int32
    b"L": ("u4", 4),  # uint32
    b"j": ("i8", 8),  # int64
    b"J": ("u8", 8),  # uint64
    b"f": ("f4", 4),  # float32
    b"d": ("f8", 8),  # float64
}

# STRM 容器里这些名字属于"元数据"，剩下的那个才是真正的传感器载荷
_STREAM_META_KEYS = {
    "STNM", "SCAL", "SIUN", "STMP", "TSMP", "UNIT", "SHUT", "MEMP",
    "MTRX", "ORIN", "DVID", "DVNM", "STRM", "EMPT",
}

# 容器的类型字节。
# 很多第三方文档写的是 '?'，但 GoPro 固件实际写的是 **0x00**（
# 用 ffmpeg 抽出 gpmd 后看头几个字节就知道：`DEVC 00 01 1ed0`）。
# 两者都要认，否则真实文件会一个流都解不出来。
_CONTAINER_TYPES = (b"?", b"\x00")

_STMP_WRAP = 1 << 32  # STMP 是 uint32 微秒，约 71.6 分钟回绕一次，需要解卷绕


# ==========================================================================
# 数据结构
# ==========================================================================
@dataclass
class GpmfElement:
    """GPMF 树里的一个节点。"""

    key: str
    """4 字节元素名，例如 'ACCL' / 'GPS5' / 'DEVC'。"""

    type: str
    """类型字符，'?' 表示容器。"""

    values: np.ndarray | None = None
    """数值型元素的去偏数据，形状 (重复次数, 每个采样点的分量数)。"""

    text: str | None = None
    """字符串型元素（类型 'U' / 'c'）的内容。"""

    children: list["GpmfElement"] = field(default_factory=list)
    """容器元素的子节点。"""

    @property
    def is_container(self) -> bool:
        return self.type == "?"

    def child(self, key: str) -> "GpmfElement | None":
        """按名字找第一个子节点（类似 dict.get）。"""
        for c in self.children:
            if c.key == key:
                return c
        return None


@dataclass
class Stream:
    """一条已解析好、带时间轴的传感器流。"""

    name: str
    """流名称，例如 'GPS5' / 'ACCL'。"""

    times: np.ndarray
    """相对录制开始的秒数，shape (N,)，单调递增。"""

    values: np.ndarray
    """原始整数值，shape (N, K)。"""

    units: str | None = None
    """SIUN 声明的单位。"""

    scale: np.ndarray | None = None
    """SCAL 声明的缩放系数（原样保留）。可能的形态见 `physical()`。"""

    description: str | None = None
    """STNM 里的人类可读描述，例如 'GPS (Lat., Long., Alt., 2D speed, 3D speed)'。"""

    @property
    def rate(self) -> float:
        """平均采样率 Hz。"""
        if self.times.size < 2:
            return 0.0
        span = float(self.times[-1] - self.times[0])
        return (self.times.size - 1) / span if span > 0 else 0.0

    def physical(self) -> np.ndarray:
        """
        换算成物理量：物理量 = 原始值 / 除数。

        SCAL 的编码有好几种，在真实文件里都遇到过，所以不能只认一种：

            · 一个值          [417]                     → 所有分量共用（HERO11 的 ACCL/GYRO）
            · 每个分量一个    [1e7,1e7,1e3,1e3,100]     → 逐分量除数（HERO11 的 GPS）
            · 数对            [(分子,分母), …]         → 除数 = 分子/分母（部分老文档/固件）

        用"元素个数 vs 分量数"来区分就够了：数对形态的元素个数恰好是分量数的两倍。
        （网上不少实现只写数对一种，拿 HERO11 的文件跑就会全解错。）
        """
        v = self.values.astype(np.float64)
        s = self.scale
        if s is None or s.size == 0:
            return v
        s = np.asarray(s, dtype=np.float64).ravel()
        k = v.shape[1] if v.ndim > 1 else 1

        if s.size == k:
            return v / s
        if s.size == 2 * k:
            num, den = s[0::2], s[1::2]
            den = np.where(den == 0, 1.0, den)
            return v / (num / den)
        return v / np.resize(s, k)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<Stream {self.name} n={self.times.size} rate={self.rate:.1f}Hz units={self.units!r}>"


# ==========================================================================
# 一、从 MP4 里抠出 gpmd 原始字节流
# ==========================================================================
def _ffmpeg_bin(name: str) -> str:
    exe = shutil.which(name)
    if exe is None:
        raise RuntimeError(
            f"找不到 {name}。请先安装 FFmpeg：brew install ffmpeg"
        )
    return exe


def probe_streams(mp4_path: str | Path) -> list[dict]:
    """用 ffprobe 列出 MP4 里所有轨道，方便定位 gpmd。"""
    ffprobe = _ffmpeg_bin("ffprobe")
    cmd = [
        ffprobe,
        "-v", "error",
        "-show_entries", "stream=index,codec_name,codec_tag_string,codec_type:stream_tags=handler_name",
        "-of", "json",
        str(mp4_path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    import json

    return json.loads(out).get("streams", [])


def find_gpmd_stream_index(mp4_path: str | Path) -> int | None:
    """找到 handler_name 含 'GoPro MET' 的轨道索引。"""
    for s in probe_streams(mp4_path):
        handler = (s.get("tags") or {}).get("handler_name", "") or ""
        if "gopro met" in handler.strip().lower() or s.get("codec_tag_string") == "gpmd":
            return int(s["index"])
    return None


def extract_gpmd_raw(mp4_path: str | Path) -> bytes:
    """
    把 MP4 里的 gpmd 轨抽成一段连续的原始字节。

    原理：让 ffmpeg 把这条元数据轨原样 copy 出来，不做任何转码。
    这里 ffmpeg 只当"拆包裹"工具用，真正的解析由下面的 Python 代码完成。
    """
    ffmpeg = _ffmpeg_bin("ffmpeg")
    mp4_path = Path(mp4_path)

    # 优先按 handler_name 选轨；取不到就退回按索引选
    mappers: list[list[str]] = [["-map", '0:m:handler_name:"GoPro MET"']]
    idx = find_gpmd_stream_index(mp4_path)
    if idx is not None:
        mappers.append(["-map", f"0:{idx}"])

    last_err = ""
    for mapper in mappers:
        for fmt in ("rawvideo", "data"):
            with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
                tmp_path = tmp.name
            cmd = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(mp4_path),
                "-codec", "copy",
                *mapper,
                "-f", fmt,
                tmp_path,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            blob = Path(tmp_path).read_bytes() if Path(tmp_path).exists() else b""
            Path(tmp_path).unlink(missing_ok=True)
            # gpmd 流以 'DEVC' 容器开头，用这个特征确认抽取成功
            if proc.returncode == 0 and len(blob) > 0 and blob[:4] in (b"DEVC", b"STRM"):
                return blob
            last_err = proc.stderr.strip() or f"输出 {len(blob)} 字节但不像 GPMF 数据"

    raise RuntimeError(
        f"无法从 {mp4_path.name} 中提取 GPMF 遥测流。\n"
        f"可能原因：该视频未开启 GPS 录制，或已被剪辑/转码抹掉了元数据轨。\n"
        f"ffmpeg 报错：{last_err}"
    )


# ==========================================================================
# 二、解析 GPMF 二进制树
# ==========================================================================
def _decode_values(type_byte: bytes, unit_size: int, repeat: int, body: bytes) -> tuple[np.ndarray | None, str | None]:
    """把一段载荷字节解成 numpy 数组或字符串。"""
    if repeat <= 0 or unit_size <= 0:
        return None, None

    # 字符串类型：'U' 是 UTF-8 串，'c' 是纯字符
    if type_byte in (b"U", b"c"):
        raw = body[: unit_size * repeat]
        raw = raw.rstrip(b"\x00")
        # GoPro 的单位字符串不是严格 UTF-8 —— 比如 "m/s²" 里的上标 2
        # 是单个字节 0xB2（Latin-1），直接按 UTF-8 解会变成乱码
        try:
            return None, raw.decode("utf-8")
        except UnicodeDecodeError:
            return None, raw.decode("latin-1")

    # FourCC：4 字节 ASCII，例如 'N', 'P', 'L'
    if type_byte == b"F":
        return None, body[: unit_size * repeat].decode("ascii", "replace")

    info = _SCALAR_TYPES.get(type_byte)
    if info is None:
        return None, None
    fmt, scalar_size = info

    if unit_size % scalar_size != 0:
        return None, None
    n_comp = unit_size // scalar_size

    dt = np.dtype(f">{fmt}")  # '>' = 大端
    n_total = repeat * n_comp
    usable = n_total * scalar_size
    if usable > len(body):
        return None, None
    flat = np.frombuffer(body[:usable], dtype=dt)
    return flat.reshape(repeat, n_comp), None


def _parse_elements(data: bytes, start: int, end: int, depth: int = 0) -> list[GpmfElement]:
    """递归解析一段字节区间里的所有 GPMF 元素。"""
    out: list[GpmfElement] = []
    pos = start
    while pos + 8 <= end:
        key = data[pos : pos + 4]
        if key == b"\x00\x00\x00\x00":
            break
        # 名字必须是可打印 ASCII，否则说明已经越界/错位
        if not all(32 <= c < 127 for c in key):
            break

        type_byte = data[pos + 4 : pos + 5]
        unit_size = data[pos + 5]
        repeat = int.from_bytes(data[pos + 6 : pos + 8], "big")

        is_container = type_byte in _CONTAINER_TYPES
        # 容器的"单元大小"字段在个别固件里是 0，此时直接把重复次数当字节数
        payload_len = repeat if (is_container and unit_size == 0) else unit_size * repeat
        body = data[pos + 8 : pos + 8 + payload_len]
        if len(body) < payload_len:
            break

        el = GpmfElement(
            key=key.decode("ascii"),
            type="?" if is_container else type_byte.decode("ascii", "replace"),
        )
        if is_container:
            # 最多嵌套 6 层，防止畸形数据导致无限递归
            if depth < 6:
                el.children = _parse_elements(body, 0, len(body), depth + 1)
        else:
            el.values, el.text = _decode_values(type_byte, unit_size, repeat, body)

        out.append(el)
        # GPMF 的每个元素都按 **4 字节对齐**。
        # 真实文件里 "Accelerometer"（13 字节）、"ZXY"（3 字节）后面都补了
        # 若干 0 字节；忽略了这一步，从第二个元素开始就会全部错位，
        # 表现为"树只解出前几个元素"。
        pos += 8 + payload_len
        pos += (-pos) % 4
    return out


def parse_gpmf(data: bytes) -> list[GpmfElement]:
    """把 gpmd 原始字节解析成元素树（顶层通常是重复出现的 DEVC 容器）。"""
    return _parse_elements(data, 0, len(data))


# ==========================================================================
# 三、把元素树整理成带时间轴的传感器流
# ==========================================================================
def _unwrap_stmp(raw_us: np.ndarray) -> np.ndarray:
    """
    STMP 若是 uint32 微秒，会在 2^32 处（约 71.6 分钟）回绕，需要解开成单调递增。

    关键是**判据要严**：只有当跌幅接近整个 2^32 时才算真回绕。
    如果宽容地把任何负跳变都当回绕，碰上本身就不按时间排序的流
    （比如把每个块的 STMP 拼起来、而块的顺序并非时间序），
    会被加上一堆 2^32，把时间轴撑到天文数字。
    """
    if raw_us.size == 0:
        return raw_us.astype(np.float64)
    v = raw_us.astype(np.int64)
    jumps = np.where(np.diff(v) < -(_STMP_WRAP * 0.5))[0]
    if jumps.size == 0:
        return v.astype(np.float64)
    offset = 0
    out = v.copy()
    for j in jumps:
        offset += _STMP_WRAP
        out[j + 1 :] += offset
    return out.astype(np.float64)


def _strm_name(strm: GpmfElement) -> str | None:
    """取 STRM 的名字（优先 STNM），没有就用载荷元素的名字。"""
    el = strm.child("STNM")
    if el is not None:
        if el.text:
            return el.text.strip()
        if el.values is not None:
            s = "".join(chr(int(c)) for c in el.values.ravel() if 0 < int(c) < 128).strip()
            if s:
                return s
    for c in strm.children:
        if c.key not in _STREAM_META_KEYS and c.values is not None:
            return c.key
    return None


def _strm_payload(strm: GpmfElement) -> GpmfElement | None:
    """
    STRM 里除元数据之外的那个元素，才是真正的传感器读数。

    取**最后一个**，不能取第一个：GoPro 会在真正的载荷前面放同属"数据"的
    辅助量，比如加速度计那一串是
        STMP → TSMP → STNM → ORIN → SIUN → SCAL → TMPC(温度) → ACCL
    取第一个非元数据元素会拿到温度而不是加速度。
    """
    found: GpmfElement | None = None
    for c in strm.children:
        if c.key not in _STREAM_META_KEYS and c.values is not None and c.values.size:
            found = c
    return found


def _strm_stmp(strm: GpmfElement) -> int | None:
    """本块最后一个采样点的时刻（微秒）。"""
    el = strm.child("STMP")
    if el is not None and el.values is not None and el.values.size:
        return int(el.values.ravel()[-1])
    return None


def _iter_strm(roots: list[GpmfElement]):
    """
    遍历出所有 STRM 容器。

    注意：GoPro 会把同一个 DEVC 容器**重复成千上万次**，每个容器里只装一小段采样，
    所以调用方必须跨容器累积。
    """
    for root in roots:
        if root.key == "STRM":
            yield root
        elif root.key == "DEVC":
            for c in root.children:
                if c.key == "STRM":
                    yield c


def dump_tree(element: GpmfElement, indent: int = 0, max_items: int = 4) -> list[str]:
    """
    把 GPMF 元素树打印成可读的缩进文本。

    排查"视频里明明有遥测却解不出数据"这类问题时，这是最直接的工具 ——
    能一眼看出每个元素的类型字节、单元大小、重复次数和取值。
    """
    lines: list[str] = []
    pad = "  " * indent
    if element.is_container:
        lines.append(f"{pad}{element.key}  (容器, {len(element.children)} 个子元素)")
        for c in element.children:
            lines.extend(dump_tree(c, indent + 1, max_items))
    elif element.text is not None:
        lines.append(f"{pad}{element.key}  type={element.type!r}  text={element.text!r}")
    elif element.values is not None and element.values.size:
        flat = element.values.ravel()
        head = ", ".join(f"{v:g}" for v in flat[:max_items])
        more = f", …共 {flat.size} 个值" if flat.size > max_items else ""
        lines.append(
            f"{pad}{element.key}  type={element.type!r}  "
            f"shape={element.values.shape}  [{head}{more}]"
        )
    else:
        lines.append(f"{pad}{element.key}  type={element.type!r}  (无值)")
    return lines


@dataclass
class _Block:
    """传感器流的一个采样块（对应一个 STRM 容器）。"""

    stmp: int | None
    """本块末尾采样点的时刻，微秒。"""

    values: np.ndarray
    """本块所有采样点，形状 (n, k)。"""

    self_timed: bool = False
    """True 表示 values 本身就是时间戳（STMP 单独成流的情况）。"""


def _build_stream(
    name: str,
    blocks: list[_Block],
    scale: np.ndarray | None,
    units: str | None,
    description: str | None = None,
) -> Stream:
    """
    把同一传感器的所有块拼成一条连续流，并用 STMP 推算每个采样点的时间。

    **STMP 记的是本块第一个采样点的时刻**（不是最后一个）。
    这一点用视频时长核对过：按"块首"解释算出的遥测总时长与视频时长吻合到 12 毫秒，
    按"块末"解释会差 380 毫秒。搞反了整条时间轴会偏几百毫秒，
    HUD 叠加就会对不上画面。

    时间轴**不做归零**：各条流共用同一个以录制开始为原点的时间轴，
    这样加速度计和 GPS 之间不会有相对偏移（摄像头开始采集各个传感器有先后，
    比如加速度计从 0.024 s 开始、GPS 从 0.116 s 开始）。
    """
    if not blocks:
        return Stream(name, np.zeros(0), np.zeros((0, 0)), units, scale, description)

    times_parts: list[np.ndarray] = []
    values_parts: list[np.ndarray] = []

    # 特例：这条流本身就是时间戳（有些固件把 STMP 单独做成一条流）
    if blocks[0].self_timed:
        raw = np.concatenate([b.values.ravel() for b in blocks]).astype(np.float64)
        t_us = _unwrap_stmp(raw)
        return Stream(name, t_us / 1e6, raw.reshape(-1, 1).astype(np.int64), units, scale, description)

    collected: list[tuple[int | None, np.ndarray]] = [
        (b.stmp, b.values) for b in blocks if b.values is not None and b.values.size
    ]
    if not collected:
        return Stream(name, np.zeros(0), np.zeros((0, 0)), units, scale, description)

    # 只有部分块带 STMP 时，用线性插值补全块级时间
    have = np.array([i for i, (s, _) in enumerate(collected) if s is not None], dtype=np.int64)
    if have.size == 0:
        t_blocks = np.arange(len(collected), dtype=np.float64) * 1000.0
    else:
        known_t = _unwrap_stmp(np.array([collected[i][0] for i in have], dtype=np.float64))
        if have.size == 1:
            t_blocks = np.full(len(collected), known_t[0], dtype=np.float64)
        else:
            t_blocks = np.interp(
                np.arange(len(collected), dtype=np.float64), have.astype(np.float64), known_t
            )

    n_blocks = len(collected)
    for bi, (_, vals) in enumerate(collected):
        n = vals.shape[0]
        # 本块内相邻采样点的间隔：
        # 相邻两块的首采样点间隔 = n × dt，所以 dt 直接用块间隔除以本块点数
        if bi + 1 < n_blocks:
            dt = (t_blocks[bi + 1] - t_blocks[bi]) / max(n, 1)
        elif bi > 0:
            dt = (t_blocks[bi] - t_blocks[bi - 1]) / max(collected[bi - 1][1].shape[0], 1)
        else:
            dt = 1000.0  # 单块且无参照，先按 1kHz 占位
        if not np.isfinite(dt) or dt <= 0:
            dt = 1000.0
        times_parts.append(t_blocks[bi] + np.arange(n, dtype=np.float64) * dt)
        values_parts.append(vals)

    times = np.concatenate(times_parts) / 1e6  # 微秒 → 秒
    values = np.concatenate(values_parts, axis=0)
    # 跨块边界可能有 ±1 微秒抖动，强制单调
    times = np.maximum.accumulate(times)
    return Stream(name, times, values, units, scale, description)


def _decode_scal(el: GpmfElement | None) -> np.ndarray | None:
    """
    取出 SCAL（缩放系数）的原始取值，**不做解释**。

    解释放在 `Stream.physical()` 里 —— 因为要判断 SCAL 属于哪种形态，
    必须知道载荷有多少个分量，而那是 Stream 才知道的信息。
    """
    if el is None or el.values is None or el.values.size == 0:
        return None
    return el.values.astype(np.float64).ravel()


def read_streams(mp4_path: str | Path, gpmd_blob: bytes | None = None) -> dict[str, Stream]:
    """
    一步到位：MP4 → {流名: Stream}。

    返回的字典里通常包含 GPS5 / GPS9 / ACCL / GYRO / GRAV / CORI / STMP 等键。
    """
    blob = gpmd_blob if gpmd_blob is not None else extract_gpmd_raw(mp4_path)
    roots = parse_gpmf(blob)

    # 一遍遍历同时收集数据块和 SCAL/SIUN
    acc: dict[str, list[_Block]] = {}
    scal_map: dict[str, np.ndarray | None] = {}
    unit_map: dict[str, str | None] = {}
    desc_map: dict[str, str] = {}

    for strm in _iter_strm(roots):
        payload = _strm_payload(strm)
        if payload is None:
            # 没有载荷的 STRM 大多数是"只声明不采样"的（比如本文件里的 GPS9 ——
            # STNM 写了 'GPS (Lat., Long., Alt., 2D, 3D, days, secs, DOP, fix)'，
            # 但根本没有 GPS9 载荷元素）。直接跳过。
            # 只有明确叫 stmp 的流才当成独立时间轴。
            hint = (_strm_name(strm) or "").strip().lower()
            if hint == "stmp":
                stmp_el = strm.child("STMP")
                if stmp_el is not None and stmp_el.values is not None:
                    acc.setdefault("STMP", []).append(_Block(None, stmp_el.values, self_timed=True))
            continue

        # 流名用**载荷的 4CC**（ACCL/GYRO/GPS5/GPS9/…），
        # 不能用 STNM —— 那里面装的是描述文字，比如
        # 'GPS (Lat., Long., Alt., 2D speed, 3D speed)'，拿来当键就找不到流了。
        name = payload.key
        desc = _strm_name(strm)
        if desc and desc != name and desc_map.get(name) is None:
            desc_map[name] = desc

        acc.setdefault(name, []).append(_Block(_strm_stmp(strm), payload.values))

        if scal_map.get(name) is None:
            s = _decode_scal(strm.child("SCAL"))
            if s is not None:
                scal_map[name] = s
        if not unit_map.get(name):
            siun = strm.child("SIUN")
            if siun is not None and siun.text:
                unit_map[name] = siun.text.strip()

    streams: dict[str, Stream] = {}
    for name, blocks in acc.items():
        st = _build_stream(
            name, blocks, scal_map.get(name), unit_map.get(name), desc_map.get(name)
        )
        if st.times.size:
            streams[name] = st
    return streams


__all__ = [
    "GpmfElement",
    "Stream",
    "dump_tree",
    "extract_gpmd_raw",
    "find_gpmd_stream_index",
    "parse_gpmf",
    "probe_streams",
    "read_streams",
]
