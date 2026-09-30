"""
本地小服务 —— 让看板上的「生成 HUD 视频」按钮真的能干活
======================================================

为什么需要它：看板是单个 HTML，双击打开时跑在浏览器沙箱里，**没有任何办法
调用 ffmpeg**。所以"点一下就出片"必须有人在浏览器外面接活，这个模块就是那个
接活的 —— 一个只监听 127.0.0.1 的小 HTTP 服务，不联网、不对外。

用法是让 analyze.py 顺便把服务挂起来：

    .venv/bin/python analyze.py video/GX010047.MP4 --serve

分析照常跑完、文件照常写完之后进程不退出，继续监听端口。然后用
**http://127.0.0.1:8765/** 打开看板（而不是双击 html 文件），按钮就活了。

为什么必须走 http 而不是双击文件：这样看板和接口是**同一个源**，不用趟
跨源（file:// 的源是 null，连预检都过不去）。顺带还能让浏览器直接播放生成好
的视频，charts/ 和 tables/ 那些链接也由同一个服务提供，行为不变。

接口（都在 /api/ 下）：
    GET  /api/ping               探活。看板用它判断按钮能不能用
    POST /api/hud   {"lap": 8}    开始生成第 8 圈的 HUD，返回任务号
    POST /api/hud   {"full": true} 整段视频
    GET  /api/hud?id=xxx           查进度
    POST /api/hud/cancel {"id"}    取消
    POST /api/gate  {"x": 12.3, "y": -40.5}   用这个位置当计时线重算整场
    POST /api/gate  {"auto": true}             恢复成自动搜索的计时线

刻意不做的事：不排队、不并发。同一时刻只允许一个出片任务 —— 编码本来就吃满
所有核心，同时跑两个只会让两个都变慢。
"""

from __future__ import annotations

import json
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import analysis as ana
from . import charts
from . import dashboard
from . import laps
from . import overlay
from . import report

PAD_SECONDS = 5.0
"""出片时在每圈前后各留几秒 —— 不然过线那一瞬间的内容会被切掉。"""

_MS_PER_FRAME = {
    # 编一帧要多少毫秒。2704×2028 实测：veryfast 约 31 ms、medium 约 51 ms，
    # 只用来给网页一个量级上的预期，不追求准。
    "ultrafast": 22.0, "superfast": 26.0, "veryfast": 31.0,
    "faster": 38.0, "fast": 44.0, "medium": 51.0, "slow": 80.0,
}
_DEFAULT_MS_PER_FRAME = 60.0
_MB_PER_SECOND = 7.0
"""输出视频每秒大约多少 MB（crf 20 附近的实测值，同样只是粗估）。"""

_HOST = "127.0.0.1"
"""只监听本地回环。这个服务能启动 ffmpeg，绝不能暴露到局域网。"""

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".csv": "text/csv; charset=utf-8",
    ".png": "image/png",
    ".mp4": "video/mp4",
}


class _Job:
    """一次出片任务。跑在自己的线程里，状态由网页轮询读取。"""

    def __init__(self, label: str, lap_index: int | None, t0: float, t1: float,
                 out_path: Path) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.label = label
        """给网页看的名字：「第 8 圈」或「整段」。"""
        self.lap = lap_index
        self.t0 = t0
        self.t1 = t1
        self.out = out_path
        self.state = "running"          # running / done / error / cancelled
        self.pct = 0.0
        self.msg = ""
        self.stop = threading.Event()
        self.started = time.time()
        self.elapsed = 0.0


class _State:
    """服务端共享状态：这次分析的结果 + 出片任务表。"""

    def __init__(self, sa: ana.SessionAnalysis, outdir: Path, video: Path, *,
                 fps: float, crf: int, preset: str, show_trace: bool,
                 sectors: int = 3, grid_step: float = 1.0,
                 write_csv: bool = True, write_charts: bool = True,
                 write_dashboard: bool = True) -> None:
        self.sa = sa
        self.tel = sa.lapset.telemetry
        self.outdir = outdir.resolve()
        self.video = video.resolve()
        self.fps = fps
        self.crf = crf
        self.preset = preset
        self.show_trace = show_trace
        # 重算计时线时要沿用的分析参数（和命令行那一次保持一致）
        self.sectors = sectors
        self.grid_step = grid_step
        self.write_csv = write_csv
        self.write_charts = write_charts
        self.write_dashboard = write_dashboard
        self.jobs: dict[str, _Job] = {}
        self.lock = threading.Lock()
        self._rebuild_lock = threading.Lock()
        # 出片耗时的粗估要用到源视频帧率（输出帧率跟的是源视频）
        try:
            self.src_fps = float(overlay.video_info(self.video)["fps"])
        except Exception:                      # noqa: BLE001 — 估不准也不该影响出片
            self.src_fps = 30.0

    def estimate(self, seconds: float) -> dict:
        """这段时间的片子大约要跑多久、多大。纯粗估，给用户一个心理准备。"""
        ms = _MS_PER_FRAME.get(self.preset, _DEFAULT_MS_PER_FRAME)
        return {
            "sec": round(seconds * self.src_fps * ms / 1000.0, 1),
            "mb": round(seconds * _MB_PER_SECOND, 1),
        }

    # ---- 圈的起止时间 ----
    def lap_range(self, lap_index: int) -> tuple[float, float]:
        """这一圈的出片范围：起止各留 PAD_SECONDS 秒，再截到视频时长以内。"""
        lap = next((l for l in self.sa.lapset.laps if l.index == lap_index), None)
        if lap is None:
            raise KeyError(f"没有第 {lap_index} 圈")
        tele = self.sa.lapset.telemetry
        duration = tele.duration if tele is not None else lap.t_end + PAD_SECONDS
        t0 = max(0.0, lap.t_start - PAD_SECONDS)
        t1 = min(duration, lap.t_end + PAD_SECONDS)
        if t1 <= t0:
            raise ValueError("这一圈的时间范围不合法")
        return t0, t1

    def running(self) -> _Job | None:
        return next((j for j in self.jobs.values() if j.state == "running"), None)

    # ---- 换计时线重算 ----
    def rebuild(self, gate: laps.Gate | None) -> dict:
        """
        用新的计时线重算整场，并把产物全部重写一遍。gate 给 None 就是回到自动搜索。

        为什么不先预览再应用：重算本身只要几秒（解析过的遥测直接复用），
        而产物必须整体重写（圈号变了，图表/看板/CSV 全都跟着变）。
        不满意就再点一下，或者在页上点「恢复自动」。
        """
        if self.tel is None:
            raise RuntimeError("这份数据里没有遥测，改不了计时线。")
        if self.running() is not None:
            raise RuntimeError(
                "有出片任务正在跑，等它完成再改计时线 —— 不然出来的片段和重算后的圈号对不上。"
            )
        if not self._rebuild_lock.acquire(blocking=False):
            raise RuntimeError("上一次重算还没结束，稍等一下。")
        try:
            lapset = laps.compute_lapset(
                self.tel, gate=gate,
                sectors=self.sectors, grid_step=self.grid_step, verbose=False,
            )
            if not lapset.laps:
                raise RuntimeError("这条线切不出完整的圈，换个位置试试。")
            sa = ana.analyze(lapset, verbose=False)

            if self.write_csv:
                report.export_csv(sa, self.outdir / "tables")
                report.export_json(sa, self.outdir / "tables" / "analysis.json")
            if self.write_charts:
                charts.make_all(sa, self.outdir / "charts")
            if self.write_dashboard:
                dashboard.build(sa, self.outdir / "dashboard.html", hud_pad=PAD_SECONDS)

            self.sa = sa
            best = sa.lapset.best_lap
            g = sa.lapset.gate
            print(f"[计时线] 已重算：{len(sa.lapset.laps)} 圈，"
                  f"最快 {laps.format_lap_time(best.duration) if best else '—'}"
                  f"（{'手动指定' if gate is not None else '自动搜索'}）", flush=True)
            return {
                "ok": True,
                "n_laps": len(sa.lapset.laps),
                "best_lap": laps.format_lap_time(best.duration) if best else "—",
                "best_lap_no": best.index if best else 0,
                "mean_lap": laps.format_lap_time(sa.lapset.mean_lap),
                "std_lap": round(float(sa.lapset.std_lap), 3),
                "gate": {"x": round(g.x, 1), "y": round(g.y, 1),
                         "lat": round(g.lat, 6), "lon": round(g.lon, 6)},
                "manual": gate is not None,
            }
        finally:
            self._rebuild_lock.release()

    def start(self, lap_index: int | None) -> _Job:
        """lap_index 给 None 就是整段视频（不加前后余量，就是全片）。"""
        with self.lock:
            busy = self.running()
            if busy is not None:
                raise RuntimeError(
                    f"{busy.label}还在生成中（{busy.pct * 100:.0f}%）。"
                    "编码会吃满所有核心，同时跑两个只会两个都变慢。"
                )
            if lap_index is None:
                tele = self.sa.lapset.telemetry
                duration = tele.duration if tele is not None else 0.0
                if duration <= 0:
                    raise ValueError("读不到视频时长，出不了整段。")
                t0, t1 = 0.0, duration
                label = "整段"
                out = self.outdir / f"{self.video.stem}_hud.mp4"
            else:
                t0, t1 = self.lap_range(lap_index)
                label = f"第 {lap_index} 圈"
                out = self.outdir / f"{self.video.stem}_hud_lap{lap_index}.mp4"
            job = _Job(label, lap_index, t0, t1, out)
            self.jobs[job.id] = job
            # 顺手清掉两小时前的旧任务，免得一直堆在内存里
            cutoff = time.time() - 7200
            for k in [k for k, j in self.jobs.items()
                      if j.state != "running" and j.started < cutoff]:
                del self.jobs[k]
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None or job.state != "running":
            return False
        job.stop.set()
        return True

    def _run(self, job: _Job) -> None:
        # 这些 print 都要 flush：日志被重定向到文件时 stdout 是块缓冲的，
        # 不 flush 的话中途 Ctrl+C 会把出片记录全丢掉
        print(f"\n[HUD] {job.label} {job.t0:.1f}~{job.t1:.1f} s 开始生成…", flush=True)
        try:
            overlay.burn(
                self.video, self.sa, job.out,
                fps=self.fps, t_range=(job.t0, job.t1),
                crf=self.crf, preset=self.preset, show_trace=self.show_trace,
                verbose=False,
                progress=lambda p: setattr(job, "pct", p),
                should_stop=job.stop.is_set,
            )
            job.state, job.pct = "done", 1.0
        except overlay.Cancelled:
            job.state, job.msg = "cancelled", "已取消"
        except Exception as exc:                       # noqa: BLE001 — 要把原因回给网页
            job.state, job.msg = "error", f"{type(exc).__name__}: {exc}"
        finally:
            job.elapsed = time.time() - job.started
            print(f"[HUD] {job.label} {job.state}，用时 {job.elapsed:.0f} s", flush=True)
            if job.state == "done":
                print(f"       {job.out}", flush=True)

    # ---- 转成 JSON 回给网页 ----
    def as_json(self, job: _Job) -> dict:
        return {
            "id": job.id,
            "lap": job.lap,
            "label": job.label,
            "state": job.state,
            "pct": round(job.pct, 4),
            "elapsed": round(job.elapsed or (time.time() - job.started), 1),
            "msg": job.msg,
            "file": job.out.name if job.state == "done" else "",
            "url": f"/hud/{job.out.name}" if job.state == "done" else "",
            "path": str(job.out) if job.state == "done" else "",
            "range": [round(job.t0, 3), round(job.t1, 3)],
        }


def _make_handler(state: _State) -> type[BaseHTTPRequestHandler]:
    """把 state 闭包进 handler —— 比挂类属性干净，也不用担心多实例互相踩。"""

    class Handler(BaseHTTPRequestHandler):
        server_version = "race-gopro-hud"

        # 默认会把每个请求打到 stderr，太吵；出片的进度我们自己打
        def log_message(self, fmt, *args):    # noqa: A002 — 覆盖父类签名
            pass

        # ---------------- 基础工具 ----------------
        def _json(self, obj, code: int = 200) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                return {}
            try:
                return json.loads(self.rfile.read(n) or b"{}")
            except (ValueError, TypeError):
                return {}

        def _path_for(self, url_path: str) -> Path | None:
            """把 URL 映射到输出目录里的文件。"""
            if url_path in ("/", "/dashboard.html", "/index.html"):
                return state.outdir / "dashboard.html"
            for folder in ("charts", "tables"):
                if url_path.startswith(f"/{folder}/"):
                    # 只保留最后一段文件名：'../' 之类的目录成分直接被丢掉
                    return state.outdir / folder / Path(url_path).name
            if url_path.startswith("/hud/"):
                return state.outdir / Path(url_path).name
            return None

        def _send_file(self, path: Path, head_only: bool = False) -> None:
            try:
                size = path.stat().st_size
            except OSError:
                self._json({"error": f"文件不存在：{path.name}"}, 404)
                return

            ctype = _CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
            start, end, partial = 0, size - 1, False
            rng = (self.headers.get("Range") or "").strip()
            if rng.startswith("bytes="):
                lo, _, hi = rng[6:].strip().partition("-")
                try:
                    if lo:
                        start = int(lo)
                        end = int(hi) if hi else size - 1
                    elif hi:                     # "bytes=-500" = 最后 500 字节
                        start = max(0, size - int(hi))
                        end = size - 1
                    else:
                        raise ValueError
                except ValueError:
                    start, end = 0, size - 1
                else:
                    # 支持 Range 是为了让浏览器能拖动视频进度条；
                    # 请求越界就老实退回整个文件
                    end = min(end, size - 1)
                    partial = 0 <= start <= end

            length = end - start + 1
            self.send_response(206 if partial else 200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if head_only:
                return

            with path.open("rb") as fh:
                fh.seek(start)
                left = length
                while left > 0:
                    chunk = fh.read(min(1 << 18, left))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        # 浏览器拖动进度条时会直接掐断连接，属于正常现象
                        return
                    left -= len(chunk)

        # ---------------- 路由 ----------------
        def do_GET(self):                    # noqa: N802 — 父类要求的命名
            self._route(head_only=False)

        def do_HEAD(self):                   # noqa: N802
            self._route(head_only=True)

        def _route(self, head_only: bool) -> None:
            url = urlparse(self.path)
            path = url.path

            if path == "/api/ping":
                tele = state.sa.lapset.telemetry
                duration = float(tele.duration) if tele is not None else 0.0
                self._json({
                    "ok": True,
                    "source": tele.source if tele is not None else "",
                    "n_laps": len(state.sa.lapset.laps),
                    "pad": PAD_SECONDS,
                    "busy": state.running() is not None,
                    "duration": round(duration, 3),
                    "full_est": state.estimate(duration),
                })
                return

            if path == "/api/hud":
                job_id = (parse_qs(url.query).get("id") or [""])[0]
                job = state.jobs.get(job_id)
                if job is None:
                    self._json({"error": "任务不存在（服务重启过？）"}, 404)
                else:
                    self._json(state.as_json(job))
                return

            target = self._path_for(path)
            if target is None:
                self._json({"error": "没有这个路径"}, 404)
            else:
                self._send_file(target, head_only=head_only)

        def do_POST(self):                   # noqa: N802
            path = urlparse(self.path).path
            body = self._body()
            if path == "/api/hud":
                try:
                    job = state.start(None if body.get("full") else int(body.get("lap")))
                except (KeyError, ValueError, TypeError, RuntimeError) as exc:
                    self._json({"error": str(exc)}, 400)
                else:
                    self._json({"id": job.id, "lap": job.lap, "label": job.label,
                                "range": [round(job.t0, 3), round(job.t1, 3)]})
                return
            if path == "/api/hud/cancel":
                ok = state.cancel(str(body.get("id") or ""))
                self._json({"ok": ok})
                return
            if path == "/api/gate":
                try:
                    if body.get("auto"):
                        summary = state.rebuild(None)
                    else:
                        try:
                            gx, gy = float(body["x"]), float(body["y"])
                        except (KeyError, TypeError, ValueError) as exc:
                            raise ValueError(
                                "要给出赛道图上的坐标：{\"x\": 米, \"y\": 米}，"
                                "或者 {\"auto\": true} 回到自动搜索。"
                            ) from exc
                        summary = state.rebuild(laps.gate_by_xy(state.tel, gx, gy))
                except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                    self._json({"error": str(exc)}, 400)
                else:
                    self._json(summary)
                return
            self._json({"error": "没有这个接口"}, 404)

    return Handler


def run(sa: ana.SessionAnalysis, outdir: str | Path, video: str | Path, *,
        port: int = 8765, fps: float = 15.0, crf: int = 20,
        preset: str = "medium", show_trace: bool = True,
        sectors: int = 3, grid_step: float = 1.0,
        write_csv: bool = True, write_charts: bool = True,
        write_dashboard: bool = True) -> int:
    """起服务并一直阻塞到 Ctrl+C。返回进程退出码。"""
    state = _State(sa, Path(outdir), Path(video), fps=fps, crf=crf,
                   preset=preset, show_trace=show_trace,
                   sectors=sectors, grid_step=grid_step,
                   write_csv=write_csv, write_charts=write_charts,
                   write_dashboard=write_dashboard)
    try:
        httpd = ThreadingHTTPServer((_HOST, port), _make_handler(state))
    except OSError as exc:
        print(f"\n端口 {port} 起不来：{exc}", file=sys.stderr)
        print("换一个：--serve-port 8766", file=sys.stderr)
        return 2

    url = f"http://{_HOST}:{port}/"
    print("\n" + "─" * 56)
    print("  本地服务已启动，用浏览器打开：")
    print(f"    {url}")
    print("  只有从这个地址打开的看板，「生成 HUD 视频」和「改计时线」才管用")
    print("  （双击 dashboard.html 不行 —— 浏览器不允许它调用 ffmpeg）")
    print(f"  源视频：{state.video}")
    print(f"  出片参数：{fps:g} fps / crf {crf} / preset {preset}"
          f" / 前后各留 {PAD_SECONDS:g} s")
    print("  按 Ctrl+C 停止")
    print("─" * 56, flush=True)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止。")
    finally:
        httpd.server_close()
    return 0


__all__ = ["PAD_SECONDS", "run"]
