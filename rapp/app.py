"""
网页应用 —— 双击就能用，不需要命令行
====================================

命令行那条路（`python analyze.py 视频.MP4 --serve`）对熟手没问题，但要一个
不懂终端的人记住一串参数、还要先 cd 到仓库目录，太劝退了。这个模块把它包成
一个"应用"：

    双击「启动.command」
      → 弹出浏览器，第一屏让你挑视频
      → 挑完自动分析，结果写在**视频旁边**的 <视频名>_out/ 里
      → 自动跳到看板（就是平时那个 dashboard.html，交给本地服务托管）
      → 服务继续开着，出片 / 改起点线这些按钮都能用
      → 关掉浏览器页面，程序自己退出

它和 `--serve` 用的是**同一套** HTTP 处理器（serve._make_handler），区别只有
两点：多了一个"还没载入视频"的中间状态（选视频那一屏），以及页面关掉就退出。

为什么非得有个本地服务：浏览器沙箱里既不能读任意路径、也不能调 ffmpeg，
"选视频"和"出片"这两件事都只能由浏览器外面的进程来做。所以这里的做法是
让**服务先起来**，选视频这一屏本身就是服务提供的一个网页。

安全：只监听 127.0.0.1。但请注意，它能列目录、能启动 ffmpeg，所以**只在
自己电脑上跑**，别改 _HOST。
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import serve, telemetry

_ROOT = Path(__file__).resolve().parent.parent
"""仓库根目录。analyze.py 在这里，它是脚本不是包，所以得手动加进 sys.path。"""

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_RECENT = _ROOT / ".race-gopro-recent.json"
"""最近打开过的视频。放仓库根、已 gitignore —— 换台机器不该带着别人的路径。"""

_VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".mts", ".mpg"}

_IDLE_GRACE = 4.0
"""最后一个页面关掉之后，等几秒再退出 —— 刷新页面也会触发一次"关闭"，
留一点时间让新页面报到，否则一刷新程序就自杀了。"""

_IDLE_LIMIT = 20 * 60
"""完全没有任何请求多久就认为页面已经没了（浏览器崩了、拔了电源之类，
pagehide 没机会发出去）。有出片任务时不计时。"""


# ==========================================================================
class _Tee:
    """
    把分析过程的输出**同时**送到终端和网页。

    为什么用重定向这种土办法而不是在流水线里埋进度回调：分析本来就会打印
    "— G 值提取 —" 这种分节信息，重定向等于白捡一份进度；而埋回调要改遍
    rapp/ 下面好几个模块，改动面大，两边打印的内容还容易对不上。
    """

    def __init__(self, stream, sink: collections.deque, lock: threading.Lock) -> None:
        self._stream = stream
        self._sink = sink
        self._lock = lock

    def write(self, text: str) -> int:
        if text:
            with self._lock:
                self._sink.append(text)
        return self._stream.write(text)

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        # 让流水线里的进度条/彩色输出走"非终端"分支，免得日志里全是转义码
        return False

    def __getattr__(self, name):
        return getattr(self._stream, name)


class _Loader:
    """管"挑视频 → 分析 → 建好服务状态"这条线。同一时刻只允许一次。"""

    def __init__(self, session: serve.Session, hud_args: dict) -> None:
        self.session = session
        self.hud_args = hud_args
        self.lock = threading.Lock()
        self.chunks: collections.deque[str] = collections.deque(maxlen=2000)
        self.phase = "idle"          # idle / running / done / error
        self.video: Path | None = None
        self.outdir: Path | None = None
        self.error = ""
        self.started = 0.0

    # ---- 给网页看的状态 ----
    def status(self) -> dict:
        with self.lock:
            chunks = list(self.chunks)
        text = "".join(chunks)
        lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
        return {
            "phase": self.phase,
            "video": str(self.video) if self.video else "",
            "outdir": str(self.outdir) if self.outdir else "",
            "error": self.error,
            "elapsed": round(time.time() - self.started, 1) if self.started else 0.0,
            "log": lines[-80:],
        }

    def start(self, raw_path: str) -> None:
        path = Path(os.path.expanduser(raw_path.strip().strip('"').strip("'")))
        if not path.exists():
            raise ValueError(f"找不到这个文件：{path}")
        if path.is_dir():
            raise ValueError("这是个文件夹，要选里面的视频文件。")
        with self.lock:
            if self.phase == "running":
                raise RuntimeError("上一次分析还没跑完，稍等一下。")
            self.phase, self.video = "running", path
            self.error, self.started = "", time.time()
            self.chunks.clear()
        threading.Thread(target=self._run, args=(path,), daemon=True).start()

    # ---- 真正的分析 ----
    def _run(self, video: Path) -> None:
        import analyze                     # 延迟导入：它在仓库根，不在包里

        try:
            args = analyze.build_parser().parse_args([])   # 全默认：图表 CSV 看板都出
            args.quiet = False
            real_stdout = sys.stdout
            sys.stdout = _Tee(real_stdout, self.chunks, self.lock)
            try:
                print(f"开始分析：{video}")
                print("（第一次跑要读几 GB 的原始文件，慢一些；同一个视频再跑会快很多）\n")
                tel = telemetry.load(video, verbose=True)
                res = analyze.run_analysis(tel, video, video.stem, args)
            finally:
                sys.stdout = real_stdout

            # 分析结果换成了服务端状态，看板的出片 / 改计时线才有数据可用。
            # app_mode=True 会让看板知道「服务是跟着页面活的」，从而上报自己的
            # 生死（见 dashboard.py 里的 lifeLink）。
            self.session.state = serve.build_state(
                res.sa, res.outdir, video, app_mode=True, **self.hud_args)
            self.outdir = res.outdir
            self.phase = "done"
            best = res.sa.lapset.best_lap
            _recent_add(video, f"{len(res.sa.lapset.laps)} 圈"
                        + (f" · 最快 {best.duration:.3f}s" if best else ""))
            print(f"\n完成。结果在：{res.outdir}", flush=True)
        except Exception as exc:                       # noqa: BLE001 — 要把原因回给网页
            self.phase = "error"
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"\n出错了：{self.error}", file=sys.stderr, flush=True)
            traceback.print_exc()


def _recent_load() -> list[dict]:
    try:
        data = json.loads(_RECENT.read_text(encoding="utf-8"))
        return [d for d in data if isinstance(d, dict) and d.get("path")]
    except (OSError, ValueError):
        return []


def _recent_add(path: Path, summary: str) -> None:
    items = [d for d in _recent_load() if d.get("path") != str(path)]
    items.insert(0, {"path": str(path), "at": time.strftime("%Y-%m-%d %H:%M"),
                     "note": summary})
    try:
        _RECENT.write_text(json.dumps(items[:8], ensure_ascii=False, indent=2),
                           encoding="utf-8")
    except OSError:
        pass                       # 记不住最近记录不影响分析，不打扰用户


# ==========================================================================
def choose_file() -> tuple[str | None, str]:
    """
    弹一个系统自带的文件选择框，返回 (路径, 错误说明)。

    用户点了取消返回 (None, "")，这不算错误。用系统对话框而不是纯网页方案，
    是因为浏览器的 <input type="file"> **只会给文件名、拿不到真实路径**，
    而分析必须知道文件在哪（结果要写它旁边）。
    """
    if sys.platform == "darwin" and shutil.which("osascript"):
        script = ('POSIX path of (choose file with prompt "选择 GoPro 拍的视频" '
                  'of type {"mp4", "MP4", "mov", "MOV", "m4v"})')
        req = subprocess.run(["osascript", "-e", script],
                             capture_output=True, text=True)
        if req.returncode == 0 and req.stdout.strip():
            return req.stdout.strip(), ""
        err = " ".join((req.stderr or "").split())
        if "User canceled" in err or "-128" in err:
            return None, ""              # 点了取消，不是错误
        return None, (f"系统对话框没能用起来（osascript 退出码 {req.returncode}）：{err}"
                      if err else
                      "系统对话框被中断了，再点一次试试；或者用下面的「浏览文件夹」。")

    if shutil.which("zenity"):                       # Linux 的常见选择
        req = subprocess.run(
            ["zenity", "--file-selection", "--title=选择 GoPro 拍的视频",
             "--file-filter=视频 | *.mp4 *.MP4 *.mov *.MOV *.m4v"],
            capture_output=True, text=True)
        if req.returncode == 0 and req.stdout.strip():
            return req.stdout.strip(), ""
        return None, ""

    return None, "这个系统上没有可用的文件选择框，请用下面的「浏览文件夹」或直接粘贴路径。"


def _browse(directory: str, last: str) -> dict:
    """列一个目录。只给文件夹和视频文件 —— 这是选视频用的，不是文件管理器。"""
    if directory:
        d = Path(os.path.expanduser(directory))
    elif last:
        d = Path(last).parent
    else:
        movies = Path.home() / "Movies"
        d = movies if movies.is_dir() else Path.home()
    if not d.is_dir():
        d = Path.home()
    d = d.resolve()

    dirs, files = [], []
    try:
        for entry in sorted(d.iterdir(), key=lambda p: p.name.lower()):
            if entry.name.startswith("."):
                continue
            try:
                if entry.is_dir():
                    # 生成出来的 *_out 目录不用列，免得和视频混在一起
                    if not entry.name.endswith("_out"):
                        dirs.append({"name": entry.name, "path": str(entry)})
                elif entry.suffix.lower() in _VIDEO_EXT:
                    files.append({
                        "name": entry.name,
                        "path": str(entry),
                        "mb": round(entry.stat().st_size / 1e6),
                    })
            except OSError:
                continue               # 没权限的条目直接跳过
    except OSError as exc:
        return {"error": f"读不了这个目录：{exc}", "dir": str(d), "dirs": [], "files": []}

    parent = str(d.parent) if d.parent != d else ""
    return {"dir": str(d), "parent": parent, "dirs": dirs[:300], "files": files[:300]}


# ==========================================================================
_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>卡丁车遥测分析</title>
<style>
  :root{--bg:#0d1117;--panel:#161b22;--line:#2a313a;--fg:#e6edf3;--dim:#8b949e;
        --accent:#1f6feb;--ok:#3fb950;--warn:#f0c674;}
  *{box-sizing:border-box;}
  body{margin:0;background:var(--bg);color:var(--fg);
       font-family:-apple-system,"PingFang SC","Heiti TC",sans-serif;}
  .wrap{max-width:720px;margin:0 auto;padding:52px 22px 60px;}
  h1{margin:0 0 6px;font-size:23px;letter-spacing:.3px;}
  h1 span{font-size:13px;font-weight:400;color:var(--dim);margin-left:10px;}
  .sub{color:var(--dim);font-size:13.5px;line-height:1.75;margin-bottom:30px;}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:13px;
        padding:20px;margin-bottom:16px;}
  .big{display:block;width:100%;padding:15px;border-radius:10px;border:0;
       background:var(--accent);color:#fff;font-size:15px;font-weight:600;
       font-family:inherit;cursor:pointer;}
  .big:hover{background:#2a7bf0;}
  .big:disabled{opacity:.5;cursor:not-allowed;}
  .row{display:flex;gap:9px;margin-top:12px;flex-wrap:wrap;}
  .row input{flex:1 1 260px;min-width:0;background:#0d1117;border:1px solid var(--line);
             color:var(--fg);border-radius:8px;padding:9px 11px;font-size:13px;
             font-family:ui-monospace,Menlo,monospace;}
  .row button,.mini{background:#1c2430;border:1px solid var(--line);color:var(--fg);
        border-radius:8px;padding:9px 15px;font-size:13px;font-family:inherit;
        cursor:pointer;white-space:nowrap;}
  .row button:hover,.mini:hover{background:#222c3a;}
  h3{margin:0 0 11px;font-size:13px;color:var(--dim);font-weight:600;}
  .hint{color:var(--dim);font-size:12px;line-height:1.7;margin:11px 0 0;}
  .recent{list-style:none;margin:0;padding:0;}
  .recent li{margin-bottom:2px;}
  .recent button{display:block;width:100%;text-align:left;background:transparent;
        border:1px solid transparent;color:var(--fg);border-radius:8px;
        padding:8px 10px;font-size:12.5px;font-family:inherit;cursor:pointer;}
  .recent button:hover{background:#1c2430;border-color:var(--line);}
  .recent em{color:var(--dim);font-style:normal;margin-left:8px;font-size:11.5px;}
  .crumbs{display:flex;gap:8px;align-items:center;margin-bottom:9px;}
  .crumbs code{flex:1;font-size:11.5px;color:var(--dim);overflow:hidden;
        text-overflow:ellipsis;white-space:nowrap;
        font-family:ui-monospace,Menlo,monospace;}
  .listing{max-height:290px;overflow:auto;border:1px solid var(--line);
        border-radius:9px;}
  .listing button{display:block;width:100%;text-align:left;background:transparent;
        border:0;border-bottom:1px solid #1c2229;color:var(--fg);padding:8px 11px;
        font-size:12.5px;font-family:inherit;cursor:pointer;}
  .listing button:hover{background:#1c2430;}
  .listing button.em{color:var(--dim);}
  .listing em{color:var(--dim);font-style:normal;float:right;font-size:11.5px;}
  .msg{margin-top:12px;font-size:12.5px;line-height:1.7;}
  .msg .bad{color:var(--warn);}
  .log{background:#0a0e13;border:1px solid var(--line);border-radius:9px;padding:11px 13px;
       margin-top:13px;max-height:330px;overflow:auto;font-size:11.5px;line-height:1.55;
       font-family:ui-monospace,Menlo,monospace;color:#a9b4c0;white-space:pre-wrap;}
  .bar{height:5px;background:#232a33;border-radius:3px;overflow:hidden;margin:14px 0 0;}
  .bar i{display:block;height:100%;width:35%;background:var(--accent);border-radius:3px;
         animation:slide 1.25s ease-in-out infinite;margin-left:-35%;}
  @keyframes slide{0%{margin-left:-35%}100%{margin-left:100%}}
  .spin{display:inline-block;width:11px;height:11px;border:2px solid #2f3a46;
        border-top-color:var(--accent);border-radius:50%;margin-right:7px;
        animation:spin .8s linear infinite;vertical-align:-1px;}
  @keyframes spin{to{transform:rotate(360deg)}}
  footer{color:#5c6672;font-size:11.5px;line-height:1.8;margin-top:26px;}
</style>
</head>
<body>
<div class="wrap">
  <h1>卡丁车遥测分析 <span>选个视频就能开始</span></h1>
  <div class="sub">挑一个 GoPro 直接导出的 MP4。分析结果会放在<b>这个视频旁边</b>的
    <code>&lt;视频名&gt;_out/</code> 里，不用你指定目录。</div>

  <div class="card" id="pick">
    <button class="big" id="pickBtn" type="button">选择视频文件…</button>
    <div class="row">
      <input id="pathInput" type="text" placeholder="也可以把视频路径粘在这里，例如 /Volumes/SD/DCIM/100GOPRO/GX010123.MP4"
             spellcheck="false">
      <button id="goBtn" type="button">开始分析</button>
    </div>
    <p class="hint" id="pickHint">在 Finder 里想复制路径：右键文件 → 按住 Option 键 →
      「拷贝…为路径名称」。</p>
    <div class="msg" id="pickMsg"></div>
  </div>

  <div class="card" id="recentCard" hidden>
    <h3>最近打开过</h3>
    <ul class="recent" id="recentList"></ul>
  </div>

  <div class="card" id="browseCard" hidden>
    <h3>浏览文件夹</h3>
    <div class="crumbs">
      <button class="mini" id="upBtn" type="button">↑ 上一层</button>
      <code id="curDir">—</code>
    </div>
    <div class="listing" id="listing"></div>
    <p class="hint">只列文件夹和视频文件。</p>
  </div>

  <div class="card" id="browseToggleCard">
    <button class="mini" id="browseToggle" type="button">找不到文件？浏览文件夹 ›</button>
  </div>

  <div class="card" id="busyCard" hidden>
    <h3><span class="spin"></span><span id="busyTitle">正在分析…</span></h3>
    <div class="hint" id="busyFile"></div>
    <div class="bar" id="busyBar"></div>
    <div class="log" id="logBox"></div>
    <p class="hint">分析完会自动跳到看板。这个页面别关 —— 关掉程序就退出了。</p>
  </div>

  <footer>分析完成后会打开看板，出片、改起点线都在那里点。<br>
    关掉浏览器页面，程序会自动退出（也可以回到这个终端窗口按 Ctrl+C）。</footer>
</div>

<script>
// ---------- 页面生命周期 ----------
// 服务和这个页面绑在一起：页面还在，服务就活着；页面关掉，服务自己结束。
// 用 sendBeacon 而不是 fetch —— 关闭页面时普通请求会被浏览器掐断，
// sendBeacon 是专为"最后一句话"设计的，一定会发出去。
let pageId = null;
async function hello(){
  try {
    const j = await (await fetch("api/hello", {cache: "no-store"})).json();
    pageId = j.id;
  } catch (e) { /* 服务没起来时不用管，反正页面也用不了 */ }
}
window.addEventListener("pagehide", () => {
  if (pageId) navigator.sendBeacon("api/bye?id=" + encodeURIComponent(pageId));
});
setInterval(() => {
  if (pageId) fetch("api/alive?id=" + encodeURIComponent(pageId), {method: "POST"})
    .catch(() => {});
}, 10000);
hello();

// ---------- 选视频 ----------
const $ = id => document.getElementById(id);
const msg = html => { $("pickMsg").innerHTML = html; };

function fail(text){ msg('<span class="bad">' + text + '</span>'); }

$("pickBtn").addEventListener("click", async () => {
  msg("正在等系统对话框…");
  $("pickBtn").disabled = true;
  try {
    const r = await fetch("api/dialog", {method: "POST"});
    const j = await r.json();
    if (j.error) { fail(j.error); return; }
    if (!j.path) { msg(""); return; }              // 用户点了取消
    $("pathInput").value = j.path;
    start(j.path);
  } catch (e) {
    fail("请求没发出去，服务还在吗？");
  } finally {
    $("pickBtn").disabled = false;
  }
});

$("goBtn").addEventListener("click", () => start($("pathInput").value));
$("pathInput").addEventListener("keydown", e => {
  if (e.key === "Enter") start($("pathInput").value);
});

async function start(path){
  if (!path || !path.trim()) { fail("先把视频路径填上。"); return; }
  msg("");
  try {
    const r = await fetch("api/load", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({path: path})
    });
    const j = await r.json();
    if (j.error) { fail(j.error); return; }
    $("pick").hidden = true;
    $("browseCard").hidden = true;
    $("browseToggleCard").hidden = true;
    $("recentCard").hidden = true;
    $("busyCard").hidden = false;
    $("busyFile").textContent = path;
    poll();
  } catch (e) {
    fail("请求没发出去，服务还在吗？");
  }
}

// ---------- 分析进度 ----------
let polling = false;
async function poll(){
  if (polling) return;
  polling = true;
  let errs = 0;
  while (true) {
    await new Promise(r => setTimeout(r, 800));
    let j = null;
    try {
      j = await (await fetch("api/state", {cache: "no-store"})).json();
      errs = 0;
    } catch (e) {
      if (++errs > 6) { fail("连不上本地服务了，可能已经退出。"); polling = false; return; }
      continue;
    }
    $("logBox").textContent = (j.log || []).join("\\n");
    $("logBox").scrollTop = $("logBox").scrollHeight;
    if (j.phase === "done") {
      $("busyBar").style.display = "none";
      $("busyTitle").textContent = "分析完成，正在打开看板…";
      $("logBox").textContent += "\\n\\n结果目录：" + j.outdir;
      setTimeout(() => { location.href = "dashboard.html"; }, 700);
      polling = false;
      return;
    }
    if (j.phase === "error") {
      $("busyBar").style.display = "none";
      $("busyTitle").textContent = "出问题了";
      $("logBox").textContent += "\\n\\n" + (j.error || "未知错误");
      $("pick").hidden = false;
      $("browseToggleCard").hidden = false;
      fail("分析没成功，上面是原因。修好后可以再试一次。");
      polling = false;
      return;
    }
  }
}

// ---------- 浏览文件夹 ----------
$("browseToggle").addEventListener("click", () => {
  const c = $("browseCard");
  c.hidden = !c.hidden;
  $("browseToggle").textContent = c.hidden ? "找不到文件？浏览文件夹 ›" : "收起文件夹 ‹";
  if (!c.hidden && !$("listing").dataset.loaded) browse("");
});

async function browse(dir){
  let j = null;
  try {
    j = await (await fetch("api/browse?dir=" + encodeURIComponent(dir || ""),
                           {cache: "no-store"})).json();
  } catch (e) { return; }
  if (j.error) { fail(j.error); }
  $("listing").dataset.loaded = "1";
  $("curDir").textContent = j.dir || "—";
  $("curDir").title = j.dir || "";
  $("upBtn").disabled = !j.parent;
  const rows = [];
  if (j.parent) rows.push('<button data-dir="' + esc(j.parent) + '">📁 ..</button>');
  for (const d of j.dirs)
    rows.push('<button data-dir="' + esc(d.path) + '">📁 ' + esc(d.name) + '</button>');
  for (const f of j.files)
    rows.push('<button class="em" data-file="' + esc(f.path) + '">🎬 ' + esc(f.name)
              + '<em>' + f.mb + ' MB</em></button>');
  $("listing").innerHTML = rows.join("") || '<button class="em">这个文件夹里没有视频</button>';
}

function esc(s){
  return String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
}

$("listing").addEventListener("click", e => {
  const b = e.target.closest("button");
  if (!b) return;
  if (b.dataset.dir) { browse(b.dataset.dir); return; }
  if (b.dataset.file) { $("pathInput").value = b.dataset.file; start(b.dataset.file); }
});
$("upBtn").addEventListener("click", () => browse($("curDir").title));

// ---------- 最近打开 ----------
(async () => {
  try {
    const j = await (await fetch("api/recent", {cache: "no-store"})).json();
    if (!j.items || !j.items.length) return;
    $("recentList").innerHTML = j.items.map((it, i) =>
      '<li><button data-i="' + i + '">' + esc(it.name || it.path)
      + '<em>' + esc(it.note || "") + ' · ' + esc(it.at || "") + '</em></button></li>'
    ).join("");
    $("recentCard").hidden = false;
    $("recentList").addEventListener("click", e => {
      const b = e.target.closest("button");
      if (b) start(j.items[+b.dataset.i].path);
    });
  } catch (e) { /* 没有最近记录就算了 */ }
})();
</script>
</body>
</html>
"""


def _launcher_html() -> bytes:
    return _TEMPLATE.encode("utf-8")


# ==========================================================================
class _App:
    """整个应用的状态：会话 + 载入器 + 页面登记表 + 退出计时。"""

    def __init__(self, hud_args: dict, open_browser: bool = True) -> None:
        self.session = serve.Session()
        self.loader = _Loader(self.session, hud_args)
        self.open_browser = open_browser
        self.pages: dict[str, float] = {}
        self.lock = threading.Lock()
        self.last_request = time.time()
        self.idle_since: float | None = None
        self.httpd = None
        self.quitting = False

    # ---- 页面生死 ----
    def hello(self) -> str:
        pid = uuid.uuid4().hex[:12]
        with self.lock:
            self.pages[pid] = time.time()
            self.idle_since = None
        return pid

    def alive(self, pid: str) -> None:
        with self.lock:
            self.pages[pid] = time.time()
            self.idle_since = None

    def bye(self, pid: str) -> None:
        with self.lock:
            self.pages.pop(pid, None)
            if not self.pages:
                self.idle_since = time.time()

    def touch(self) -> None:
        self.last_request = time.time()

    # ---- 退出 ----
    def quit_soon(self, why: str, delay: float = 0.0) -> None:
        if self.quitting:
            return
        self.quitting = True
        threading.Thread(target=self._quit, args=(why, delay), daemon=True).start()

    def _quit(self, why: str, delay: float) -> None:
        time.sleep(delay)
        state = self.session.state
        # 出片正在跑就先等它 —— 用户多半是"点了出片就去干别的"，不该半路掐掉
        waited = False
        while state is not None and state.running() is not None:
            if not waited:
                print(f"\n[退出] {why}，但出片还在跑，等它结束再退…", flush=True)
                waited = True
            time.sleep(5)
        print(f"\n[退出] {why}。服务已停止。", flush=True)
        if self.httpd is not None:
            self.httpd.shutdown()

    def watchdog(self, httpd) -> None:
        """兜底清理：页面已经没了（浏览器崩了之类）也别让服务一直挂着占端口。"""
        while not self.quitting:
            time.sleep(2)
            now = time.time()
            with self.lock:
                stale = [p for p, t in self.pages.items() if now - t > 300]
                for p in stale:
                    self.pages.pop(p, None)
                idle = self.idle_since
            state = self.session.state
            busy = ((state is not None and state.running() is not None)
                    or self.loader.phase == "running")
            if idle is not None and not busy and now - idle > _IDLE_GRACE:
                self.quit_soon("浏览器页面已关闭")
                return
            if not busy and now - self.last_request > _IDLE_LIMIT:
                self.quit_soon(f"{_IDLE_LIMIT // 60} 分钟没有任何操作")
                return


def _make_handler(app: _App) -> type[BaseHTTPRequestHandler]:
    """
    在 serve 的处理器上**加一层**：选视频阶段的几个接口 + 页面生死。

    载入视频之后所有请求都交回给 serve 那套 —— 看板、图表、出片、改计时线
    一行都不用改，这也是把应用做成"同一套服务多一个中间状态"的原因。
    """
    Base = serve.make_handler(app.session)

    class Handler(Base):
        server_version = "race-gopro-app"

        # ---- 小工具 ----
        def _ok(self, obj=None):
            self._json(obj if obj is not None else {"ok": True})

        def _post_json(self):
            return self._body()

        # ---- 选视频阶段的路由 ----
        def _app_get(self, path: str, query: dict) -> bool:
            """返回 True 表示这个请求我已经处理掉了。"""
            if path == "/api/hello":
                self._ok({"id": app.hello()})
                return True
            if path == "/api/state":
                st = app.loader.status()
                # 还开着的页面数。调试"关页面没退出"这种问题时，一眼就能看出
                # 是页面没报到、还是报到了没销号
                with app.lock:
                    st["pages"] = len(app.pages)
                self._ok(st)
                return True
            if path == "/api/recent":
                items = []
                for d in _recent_load():
                    p = Path(d["path"])
                    items.append({
                        "path": d["path"], "name": p.name,
                        "at": d.get("at", ""), "note": d.get("note", ""),
                        "exists": p.exists(),
                    })
                self._ok({"items": items})
                return True
            if path == "/api/browse":
                last = app.loader.video
                self._ok(_browse((query.get("dir") or [""])[0], str(last) if last else ""))
                return True
            return False

        def _app_post(self, path: str, query: dict) -> bool:
            if path == "/api/alive":
                app.alive((query.get("id") or [""])[0])
                self._ok()
                return True
            if path == "/api/bye":
                app.bye((query.get("id") or [""])[0])
                self._ok()
                return True
            if path == "/api/quit":
                self._ok()
                app.quit_soon("收到退出请求", delay=0.2)
                return True
            if path == "/api/dialog":
                p, err = choose_file()
                if err:
                    self._json({"error": err}, 500)
                else:
                    self._ok({"path": p or ""})
                return True
            if path == "/api/load":
                raw = (self._post_json().get("path") or "")
                try:
                    app.loader.start(raw)
                except (ValueError, RuntimeError) as exc:
                    self._json({"error": str(exc)}, 400)
                else:
                    self._ok({"started": True})
                return True
            return False

        # ---- 入口 ----
        def do_GET(self):                     # noqa: N802
            app.touch()
            url = urlparse(self.path)
            query = parse_qs(url.query)
            if self._app_get(url.path, query):
                return
            if app.session.state is None:
                if url.path in ("/", "/index.html", "/launcher"):
                    body = _launcher_html()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self._json({"error": "还没有载入视频"}, 409)
                return
            if url.path in ("/", "/index.html", "/launcher"):
                # 已经有结果了，回首页就直接去看板
                self.send_response(302)
                self.send_header("Location", "/dashboard.html")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            super().do_GET()

        def do_HEAD(self):                    # noqa: N802
            app.touch()
            if app.session.state is None:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            super().do_HEAD()

        def do_POST(self):                    # noqa: N802
            app.touch()
            url = urlparse(self.path)
            query = parse_qs(url.query)
            if self._app_post(url.path, query):
                return
            if app.session.state is None:
                self._json({"error": "还没有载入视频"}, 409)
                return
            super().do_POST()

    return Handler


def _fix_tool_path() -> None:
    """
    把 Homebrew 那几个目录补进 PATH。

    为什么需要：双击 .app 时，程序拿到的是系统默认 PATH
    （/usr/bin:/bin:/usr/sbin:/sbin）—— 里面**没有 Homebrew**，因为那是 shell
    的 rc 文件加的，而 .app 根本不经过 shell。结果就是 ffmpeg 明明装在
    /opt/homebrew/bin，`shutil.which("ffmpeg")` 却找不到，整个程序直接罢工。

    只在真的找不到时才补，不改动用户自己的环境；命令行启动时 PATH 本来就是
    全的，这个函数会直接返回。
    """
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    parts = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    for d in ("/opt/homebrew/bin", "/usr/local/bin", "/opt/local/bin"):
        if os.path.isdir(d) and d not in parts:
            parts.append(d)
    os.environ["PATH"] = os.pathsep.join(parts)
    if shutil.which("ffmpeg"):
        print(f"[提示] PATH 里原来没有 ffmpeg，已补上 {os.environ['PATH']}", flush=True)


# ==========================================================================
def run(port: int = 8765, open_browser: bool = True) -> int:
    """起应用并阻塞到退出。返回进程退出码。"""
    _fix_tool_path()
    hud_args = dict(fps=15.0, crf=20, preset="veryfast", show_trace=True,
                    sectors=3, grid_step=1.0)
    app = _App(hud_args, open_browser=open_browser)
    handler = _make_handler(app)
    url = f"http://127.0.0.1:{port}/"
    try:
        httpd = serve.bind(handler, port)
    except OSError:
        # 端口被占：多半是上一次没退干净。与其报错，不如直接把已经跑着的那个打开
        print(f"\n端口 {port} 已经被占用 —— 大概率是本程序已经开着一个了。")
        print(f"直接给你打开那个：{url}")
        if open_browser:
            webbrowser.open(url)
        return 0

    app.httpd = httpd
    threading.Thread(target=app.watchdog, args=(httpd,), daemon=True).start()

    print("\n" + "─" * 58)
    print("  卡丁车遥测分析")
    print(f"  浏览器地址：{url}")
    if open_browser:
        print("  （已经在浏览器里打开了）")
    print("  选视频 → 自动分析 → 跳看板；出片、改起点线都在看板里点。")
    print("  关掉浏览器页面程序会自己退出；也可以在这个窗口按 Ctrl+C。")
    print("─" * 58, flush=True)
    if open_browser:
        webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，退出。")
    finally:
        httpd.server_close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m rapp.app",
        description="卡丁车遥测分析 —— 网页应用（双击 启动.command 也可以）")
    p.add_argument("--port", type=int, default=8765, help="本地服务端口，默认 8765")
    p.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(port=args.port, open_browser=not args.no_open)


if __name__ == "__main__":
    sys.exit(main())
