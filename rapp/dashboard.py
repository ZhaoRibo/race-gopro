"""
输出层：交互式网页看板
====================

生成一个**单文件 HTML**，双击就能在浏览器里打开：

    · 圈速柱状图，最快圈高亮
    · 速度—距离曲线，可点击图例开关每一条圈
    · 相对最快圈的 delta 曲线
    · G-G 摩擦圆散点图
    · 赛道俯视图（按速度着色）
    · 逐弯顶点速度明细表

数据直接以 JSON 内嵌在 HTML 里，不需要服务器，也不怕文件被挪走。
Chart.js 走 CDN，联网即可；离线时页面会给出提示（其余表格内容仍然可看）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import analysis as ana
from . import laps

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root{
    --bg:#0e1117; --panel:#161b22; --line:#232a33;
    --fg:#e6edf3; --dim:#8b949e; --best:#a855f7; --accent:#22d3ee;
  }
  *{box-sizing:border-box;}
  body{margin:0;background:var(--bg);color:var(--fg);
       font-family:"PingFang SC","Heiti TC",-apple-system,"Segoe UI",Roboto,"Microsoft YaHei",sans-serif;}
  header{padding:22px 26px 12px;border-bottom:1px solid var(--line);
         background:linear-gradient(180deg,#161b22 0%,#0e1117 100%);}
  h1{margin:0 0 4px;font-size:20px;letter-spacing:.5px;}
  h1 span{color:var(--best);}
  .sub{color:var(--dim);font-size:12.5px;}
  .wrap{padding:18px 26px 60px;max-width:1500px;margin:0 auto;}
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:22px;}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:13px 15px;}
  .card .k{color:var(--dim);font-size:11.5px;margin-bottom:6px;letter-spacing:.3px;}
  .card .v{font-size:21px;font-weight:600;font-variant-numeric:tabular-nums;}
  .card .v small{font-size:12px;color:var(--dim);font-weight:400;margin-left:3px;}
  .card.best .v{color:var(--best);}
  .card.accent .v{color:var(--accent);}
  section{background:var(--panel);border:1px solid var(--line);border-radius:12px;
          padding:16px 18px;margin-bottom:18px;}
  section h2{margin:0 0 4px;font-size:14.5px;font-weight:600;}
  section p.hint{margin:0 0 14px;color:var(--dim);font-size:12px;line-height:1.6;}
  .chartbox{position:relative;height:340px;}
  .chartbox.tall{height:430px;}
  /* 需要 x/y 等比例的图（赛道图、G-G 图）用正方形容器。
     光靠设置坐标轴范围还不够 —— 容器本身不是正方形的话，
     1 米横向和 1 米纵向占的像素数依然不同，赛道形状照样会被拉变形。 */
  .chartbox.square{position:relative;height:auto;width:100%;
                   max-width:620px;margin:0 auto;aspect-ratio:1/1;}
  .chartbox.wide-square{position:relative;height:auto;width:100%;
                        max-width:760px;margin:0 auto;aspect-ratio:1/1;}
  table{width:100%;border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums;}
  th,td{padding:7px 9px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap;}
  th{color:var(--dim);font-weight:500;position:sticky;top:0;background:var(--panel);}
  th:first-child,td:first-child{text-align:left;}
  td.best{color:var(--best);font-weight:600;}
  tr:hover td{background:#1c2430;}
  .scroll{max-height:460px;overflow:auto;}
  .warn{background:#3d2a10;border:1px solid #6b4a15;color:#f0c674;
        padding:11px 14px;border-radius:8px;font-size:12.5px;margin-bottom:16px;line-height:1.7;}
  .pill{display:inline-block;padding:1px 7px;border-radius:20px;font-size:11px;
        background:#232a33;color:var(--dim);margin-left:6px;}

  /* 圈选择器：吸顶，滚到哪张图都能随手勾 */
  .picker{position:sticky;top:0;z-index:20;background:var(--bg);
          border:1px solid var(--line);border-radius:12px;padding:12px 16px 11px;
          margin-bottom:18px;box-shadow:0 10px 18px -14px #000;}
  .picker .row{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;}
  .picker h2{margin:0;font-size:14px;font-weight:600;}
  .picker .count{color:var(--dim);font-size:12px;font-variant-numeric:tabular-nums;}
  .lapchips{display:flex;flex-wrap:wrap;gap:7px;margin:10px 0 9px;}
  .lapchip{display:inline-flex;align-items:center;gap:6px;cursor:pointer;
           background:var(--panel);border:1px solid var(--line);border-radius:8px;
           padding:4px 10px 4px 7px;font-size:12.5px;user-select:none;
           font-variant-numeric:tabular-nums;color:var(--dim);}
  .lapchip:hover{border-color:#39414d;}
  .lapchip.on{color:var(--fg);border-color:#39414d;background:#1c2430;}
  .lapchip input{accent-color:var(--accent);margin:0;cursor:pointer;}
  .lapchip em{font-style:normal;color:var(--dim);font-size:11.5px;}
  .lapchip.on em{color:#a9b4c0;}
  .lapchip.best span{color:var(--best);font-weight:600;}
  .pickerbtns{display:flex;gap:7px;flex-wrap:wrap;}
  .pickerbtns button{background:var(--panel);color:var(--dim);border:1px solid var(--line);
                     border-radius:7px;padding:4px 11px;font-size:12px;cursor:pointer;
                     font-family:inherit;}
  .pickerbtns button:hover{color:var(--fg);border-color:#39414d;background:#1c2430;}
  .picker .hint{margin:9px 0 0;color:var(--dim);font-size:11.5px;line-height:1.6;}

  /* “其他文件”区块：指向 charts/ 与 tables/ 里的产物 */
  section h3{margin:15px 0 7px;font-size:12px;font-weight:600;color:var(--dim);letter-spacing:.4px;}
  section h3:first-of-type{margin-top:4px;}
  .files{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:5px 18px;}
  .files a{display:block;color:var(--fg);text-decoration:none;font-size:12.5px;
           padding:5px 9px;border-radius:7px;border:1px solid transparent;}
  .files a:hover{background:#1c2430;border-color:var(--line);}
  .files a span{color:var(--dim);margin-left:8px;font-size:11.5px;}
</style>
</head>
<body>
<header>
  <h1>卡丁车遥测分析 <span>__BESTLAP__</span></h1>
  <div class="sub">__SOURCE__ · 共 __NLAPS__ 圈 · 里程 __DIST__ m · __DATE__</div>
</header>
<div class="wrap">

  <div id="nodata"></div>

  <div class="cards">
    __CARDS__
  </div>

  <section class="picker" id="picker" hidden>
    <div class="row">
      <h2>选择要显示的圈</h2>
      <span class="count" id="pickerCount"></span>
    </div>
    <div class="lapchips" id="lapPicker"></div>
    <div class="pickerbtns" id="pickerBtns">
      <button type="button" data-act="all">全选</button>
      <button type="button" data-act="none">全不选</button>
      <button type="button" data-act="best">只看最快圈</button>
      <button type="button" data-act="top3">最快 3 圈</button>
      <button type="button" data-act="top5">最快 5 圈</button>
    </div>
    <p class="hint">勾选会同时作用到下面所有「每圈一条线」的图表：速度—距离曲线、时间差、每圈走线、每圈走线偏差。<b>点任意一张图的图例效果完全一样</b>，两边是同步的（圈速分布、G-G 图、赛道俯视图不受影响 —— 它们不是按圈拆的）。</p>
  </section>

  <section>
    <h2>圈速分布</h2>
    <p class="hint">紫柱是最快圈。虚线是平均圈、理论最佳和连续分段最佳。非正常圈（如出场圈）已从图中移除，避免把纵轴拉开、其余柱子被挤成一条线；它们在下面的表格里仍完整保留。</p>
    <div class="chartbox"><canvas id="cLapTime"></canvas></div>
  </section>

  <section>
    <h2>速度—距离曲线</h2>
    <p class="hint">点击下方图例可以隐藏/显示某几圈。灰色竖带是识别出的弯道区间。曲线越重合，说明走线越一致。</p>
    <div class="chartbox tall"><canvas id="cSpeed"></canvas></div>
  </section>

  <section>
    <h2>相对最快圈的时间差</h2>
    <p class="hint">曲线上升 = 从这一点开始正在丢时间；下降 = 正在把时间追回来。看得最清楚的一栏。</p>
    <div class="chartbox tall"><canvas id="cDelta"></canvas></div>
  </section>

  <section>
    <h2>G-G 图（摩擦圆）</h2>
    <p class="hint">横轴 = 横向 G（左转为正），纵轴 = 纵向 G（加速为正）。两轴等比例，散点围出的外沿就是这条轮胎的抓地力极限。</p>
    <div class="chartbox square"><canvas id="cGG"></canvas></div>
  </section>

  <section>
    <h2>赛道俯视图</h2>
    <p class="hint">颜色代表速度。白色的短直线是起点线，标注是识别出的弯道。x/y 轴等比例，所以赛道形状没有变形。</p>
    <div class="chartbox wide-square"><canvas id="cMap"></canvas></div>
  </section>

  <section>
    <h2>每圈走线</h2>
    <p class="hint">把每一圈的实测轨迹叠在一起（各圈按"距起点线的距离"对齐）。点击图例可以隐藏/显示某几圈。<b>线条几乎重合是正常的</b> —— GPS 单点噪声就有 1~3 米，真实走线差异往往只有零点几米，光看形状分不出差别，要看下面那张偏差图。x/y 轴等比例。</p>
    <div class="chartbox square"><canvas id="cLines"></canvas></div>
  </section>

  <section>
    <h2>每圈走线偏差</h2>
    <p class="hint">每一圈相对<b>平均走线</b>的横向偏移：正 = 在平均线左侧，负 = 右侧。这是看走线差异真正管用的一张 —— 噪声被抵消掉了，差异直接量化成米。</p>
    <div class="chartbox tall"><canvas id="cDev"></canvas></div>
  </section>

  <section>
    <h2>逐弯顶点速度</h2>
    <p class="hint">每个弯里速度最低的那一点。同一弯各圈差别越大，说明这个弯还没形成稳定跑法。</p>
    <div class="scroll"><table id="tCorner"></table></div>
  </section>

  <section>
    <h2>逐圈明细</h2>
    <p class="hint">标有 <b>*</b> 的圈（灰色）是出场圈之类的非正常圈，不计入稳定性统计，也不出现在上面的图表里。</p>
    <div class="scroll"><table id="tLap"></table></div>
  </section>

__FILES__</div>

<script>
const DATA = __DATA__;
// 图表只用有效圈：出场圈通常慢十几秒，画进来会把其他圈压成一堆
const LAPS = DATA.laps.filter(l => l.valid);
const SKIPPED = DATA.laps.filter(l => !l.valid).map(l => "#" + l.index).join(", ");
const SUFFIX = SKIPPED ? "（已排除非正常圈 " + SKIPPED + "）" : "";
const BEST = LAPS.length ? LAPS.reduce((a,b)=>b.duration_s<a.duration_s?b:a) : null;
const isBest = l => BEST && l.index === BEST.index;

if (typeof Chart === "undefined") {
  document.getElementById("nodata").innerHTML =
    '<div class="warn">图表库（Chart.js）加载失败 —— 当前处于离线状态。' +
    '页面下方的两张表格仍然可以正常查看。联网后刷新即可恢复图表。</div>';
} else {

const PURPLE = "#a855f7";
const PALETTE = ["#3b82f6","#f97316","#22c55e","#ef4444","#a855f7","#14b8a6",
                 "#eab308","#ec4899","#64748b","#8b5cf6","#06b6d4","#f43f5e",
                 "#84cc16","#6366f1"];
const color = (i, l) => isBest(l) ? PURPLE : PALETTE[i % PALETTE.length];

Chart.defaults.color = "#8b949e";
Chart.defaults.borderColor = "#232a33";
Chart.defaults.font.family = '"PingFang SC","Heiti TC",sans-serif';

// ---------- 弯道区间底纹：写成一个内联插件，所有图表共用 ----------
const bandPlugin = {
  id: "bands",
  beforeDatasetsDraw(chart, args, opts) {
    const bands = (opts && opts.bands) || [];
    if (!bands.length) return;
    const {ctx, chartArea: a, scales} = chart;
    if (!a || !scales.x) return;
    ctx.save();
    ctx.fillStyle = "rgba(255,255,255,0.045)";
    for (const [s, e] of bands) {
      const x0 = scales.x.getPixelForValue(s), x1 = scales.x.getPixelForValue(e);
      ctx.fillRect(x0, a.top, Math.max(x1 - x0, 1), a.bottom - a.top);
    }
    ctx.restore();
  }
};
Chart.register(bandPlugin);

// ---------- 水平参考线：平均圈 / 理论最佳 / 连续分段最佳 ----------
const refLinePlugin = {
  id: "refLines",
  afterDatasetsDraw(chart, args, opts) {
    const lines = (opts && opts.lines) || [];
    if (!lines.length) return;
    const {ctx, chartArea: a, scales} = chart;
    if (!a || !scales.y) return;
    ctx.save();
    ctx.font = "11px 'PingFang SC',sans-serif";
    ctx.textBaseline = "bottom";
    for (const ln of lines) {
      if (ln.y == null || !isFinite(ln.y)) continue;
      const y = scales.y.getPixelForValue(ln.y);
      if (y < a.top || y > a.bottom) continue;
      ctx.strokeStyle = ln.color;
      ctx.lineWidth = 1.4;
      ctx.setLineDash(ln.dash || []);
      ctx.beginPath(); ctx.moveTo(a.left, y); ctx.lineTo(a.right, y); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = ln.color;
      ctx.fillText(ln.text, a.right - ctx.measureText(ln.text).width - 6, y - 3);
    }
    ctx.restore();
  }
};
Chart.register(refLinePlugin);

// ---------- 圈选择开关 ----------
// 下面所有「每圈一条线」的图表共用这一份状态。点勾选框和点图例是同一件事：
// 都改这一份状态，然后统一刷所有注册过的图表 —— 两边永远不会不同步。
const lapOn = new Map();
LAPS.forEach(l => lapOn.set(l.index, true));
const LAP_CHARTS = [];   // [{chart, laps: [圈号, ...]}]，laps[i] 是第 i 个 dataset 对应的圈号

function applyLapSelection() {
  for (const item of LAP_CHARTS) {
    item.chart.data.datasets.forEach((ds, i) => {
      const idx = item.laps[i];
      ds.hidden = idx === undefined ? false : !lapOn.get(idx);
    });
    item.chart.update("none");
  }
  syncPicker();
}

function toggleLap(idx, force) {
  lapOn.set(idx, force === undefined ? !lapOn.get(idx) : !!force);
  applyLapSelection();
}

// 注册一张"按圈拆"的图表：接管它的图例点击（改成切全局选择），
// 并记下 dataset 序号 → 圈号 的映射（各图表的映射并不一样：
// Delta 图就刻意不含最快圈，因为自己减自己是恒为 0 的一条直线）。
function registerLapChart(chart, lapIdx) {
  if (chart.options.plugins && chart.options.plugins.legend) {
    chart.options.plugins.legend.onClick = (evt, item) => {
      const idx = lapIdx[item.datasetIndex];
      if (idx !== undefined) toggleLap(idx);
    };
  }
  LAP_CHARTS.push({chart, laps: lapIdx});
  return chart;
}

const pickerEl = document.getElementById("lapPicker");
function syncPicker() {
  pickerEl.querySelectorAll("input[data-lap]").forEach(inp => {
    const on = !!lapOn.get(+inp.dataset.lap);
    inp.checked = on;
    inp.parentElement.classList.toggle("on", on);
  });
  const n = [...lapOn.values()].filter(Boolean).length;
  document.getElementById("pickerCount").textContent =
    "已选 " + n + " / " + LAPS.length + " 圈" + (n ? "" : "（图表会是空的）");
}

(function initPicker(){
  if (!LAPS.length) return;
  document.getElementById("picker").hidden = false;
  pickerEl.innerHTML = LAPS.map(l =>
    '<label class="lapchip' + (isBest(l) ? " best" : "") + '">' +
      '<input type="checkbox" data-lap="' + l.index + '" checked>' +
      "<span>#" + l.index + (isBest(l) ? " ★" : "") + "</span>" +
      "<em>" + l.time + "</em>" +
    "</label>").join("");
  pickerEl.addEventListener("change", e => {
    if (e.target.matches("input[data-lap]")) toggleLap(+e.target.dataset.lap, e.target.checked);
  });
  document.getElementById("pickerBtns").addEventListener("click", e => {
    const act = e.target.dataset.act;
    if (!act) return;
    const byTime = [...LAPS].sort((a, b) => a.duration_s - b.duration_s);
    const pick = k => new Set(byTime.slice(0, k).map(l => l.index));
    if (act === "all") LAPS.forEach(l => lapOn.set(l.index, true));
    else if (act === "none") LAPS.forEach(l => lapOn.set(l.index, false));
    else if (act === "best") LAPS.forEach(l => lapOn.set(l.index, isBest(l)));
    else if (act === "top3" || act === "top5") {
      const keep = pick(act === "top3" ? 3 : 5);
      LAPS.forEach(l => lapOn.set(l.index, keep.has(l.index)));
    }
    applyLapSelection();
  });
  syncPicker();
})();

// ---------- 1. 圈速柱状图 ----------
const dur = LAPS.map(l => l.duration_s);
const refs = dur.concat([DATA.theoretical_best, DATA.rolling_best]);
const yLo = Math.min(...refs) - 0.6;
const yMax = Math.max(...refs) + 1.2;
new Chart(document.getElementById("cLapTime"), {
  type: "bar",
  data: {
    labels: LAPS.map(l => "#" + l.index),
    datasets: [{
      data: dur,
      backgroundColor: LAPS.map((l,i) => color(i,l)),
      borderRadius: 4,
      barPercentage: 0.72
    }]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: {display: false},
      tooltip: {callbacks: {label: c => LAPS[c.dataIndex].time + "  (" + c.parsed.y.toFixed(3) + " s)"}},
      refLines: {lines: [
        {y: DATA.mean_lap, color: "#8b949e", dash: [6,4], text: "平均 " + DATA.mean_lap_text},
        {y: DATA.theoretical_best, color: "#ef4444", dash: [2,3], text: "理论最佳 " + DATA.theoretical_best_text},
        {y: DATA.rolling_best, color: "#f59e0b", dash: [8,3,2,3], text: "连续分段最佳 " + DATA.rolling_best_text}
      ]}
    },
    scales: {
      y: {min: yLo, max: yMax, title: {display: true, text: "圈时 (s)"}}
    }
  }
});

// ---------- 2. 速度曲线 ----------
const speedSets = LAPS.map((l,i) => ({
  label: "#" + l.index + "  " + l.time,
  data: l.speed_kmh.map((v,j) => ({x: DATA.grid[j], y: v})),
  borderColor: color(i,l),
  backgroundColor: color(i,l),
  borderWidth: isBest(l) ? 2.6 : 1.2,
  pointRadius: 0, tension: 0.15, spanGaps: true
}));
const cSpeed = new Chart(document.getElementById("cSpeed"), {
  type: "line",
  data: {datasets: speedSets},
  options: {
    responsive: true, maintainAspectRatio: false,
    interaction: {mode: "index", intersect: false},
    plugins: {
      legend: {labels: {boxWidth: 14, boxHeight: 3, font: {size: 11}}},
      bands: {bands: DATA.corner_bands}
    },
    scales: {
      x: {type: "linear", title: {display: true, text: "距起点线距离 (m)"}},
      y: {title: {display: true, text: "速度 (km/h)"}}
    }
  }
});
registerLapChart(cSpeed, LAPS.map(l => l.index));

// ---------- 3. Delta 曲线 ----------
const deltaSets = LAPS.filter(l => !isBest(l)).map(l => {
  const i = LAPS.indexOf(l);
  return {
    label: "#" + l.index,
    data: l.delta_s.map((v,j) => ({x: DATA.grid[j], y: v})),
    borderColor: color(i,l), backgroundColor: color(i,l),
    borderWidth: 1.4, pointRadius: 0, tension: 0.15, spanGaps: true
  };
});
const cDelta = new Chart(document.getElementById("cDelta"), {
  type: "line",
  data: {datasets: deltaSets},
  options: {
    responsive: true, maintainAspectRatio: false,
    interaction: {mode: "nearest", intersect: false},
    plugins: {
      legend: {labels: {boxWidth: 14, boxHeight: 3, font: {size: 11}}},
      bands: {bands: DATA.corner_bands},
      tooltip: {callbacks: {label: c => c.dataset.label + "  " + (c.parsed.y>=0?"+":"") + c.parsed.y.toFixed(3) + " s"}}
    },
    scales: {
      x: {type: "linear", title: {display: true, text: "距起点线距离 (m)"}},
      y: {title: {display: true, text: "时间差 (s)"}}
    }
  }
});
// Delta 图刻意不含最快圈（自己减自己恒为 0），所以映射和别的图表不一样，
// 必须显式传它自己的圈号列表，不能假设跟 LAPS 的序号一一对应。
registerLapChart(cDelta, LAPS.filter(l => !isBest(l)).map(l => l.index));

// ---------- 4. G-G 图 ----------
const gg = DATA.gg;
new Chart(document.getElementById("cGG"), {
  type: "scatter",
  data: {datasets: [{
    label: "采样点",
    data: gg.lat.map((v,i) => ({x: v, y: gg.lon[i]})),
    pointRadius: 1.4,
    pointBackgroundColor: "rgba(59,130,246,0.35)",
    pointBorderWidth: 0
  }, {
    label: "抓地力包线",
    type: "line",
    data: gg.env_lat.map((v,i) => ({x: v, y: gg.env_lon[i]})),
    borderColor: "#ef4444", borderWidth: 2.4, pointRadius: 0, tension: 0.3, fill: false
  }]},
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: {legend: {labels: {boxWidth: 14, boxHeight: 3, font: {size: 11}}}},
    scales: {
      x: {title: {display: true, text: "横向 G   (正值 = 左转)"},
          min: -Math.ceil(gg.max), max: Math.ceil(gg.max)},
      y: {title: {display: true, text: "纵向 G   (正值 = 加速)"},
          min: -Math.ceil(gg.max), max: Math.ceil(gg.max)}
    }
  }
});

// ---------- 5. 赛道俯视图 ----------
const map = DATA.track_map;

// x/y 必须等比例，否则赛道形状会被拉变形（卡丁车场是细长形，拉伸后完全认不出来）。
// Chart.js 没有内置的"等比例坐标轴"选项，所以要两手都做：
//   · 数据窗口取成正方形：x、y 用**相同的半径**，取两者跨度里的较大值
//   · 容器也做成正方形（CSS aspect-ratio:1/1）
// 两者同时成立，1 米横向和 1 米纵向占的像素数才相同。
const mapX = map.x, mapY = map.y;
const mapXLo = Math.min(...mapX), mapXHi = Math.max(...mapX);
const mapYLo = Math.min(...mapY), mapYHi = Math.max(...mapY);
const mapCx = (mapXLo + mapXHi) / 2, mapCy = (mapYLo + mapYHi) / 2;
const mapHalf = Math.max(mapXHi - mapXLo, mapYHi - mapYLo) / 2 * 1.06;  // 留 6% 边距

new Chart(document.getElementById("cMap"), {
  type: "scatter",
  data: {datasets: [{
    label: "赛道",
    data: map.x.map((v,i) => ({x: v, y: map.y[i], s: map.speed_kmh[i]})),
    pointRadius: 3.2,
    pointBorderWidth: 0,
    pointBackgroundColor: map.speed_kmh.map(s => speedColor(s, map.vmin, map.vmax))
  }, {
    label: "起点线",
    type: "line",
    data: map.gate.map(p => ({x: p[0], y: p[1]})),
    borderColor: "#ffffff", borderWidth: 2.5, pointRadius: 0, fill: false
  }]},
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: {labels: {boxWidth: 14, boxHeight: 3, font: {size: 11}}},
      tooltip: {callbacks: {label: c => c.datasetIndex === 0
        ? c.raw.s.toFixed(1) + " km/h"
        : "起点线"}}
    },
    scales: {
      x: {min: mapCx - mapHalf, max: mapCx + mapHalf,
          title: {display: true, text: "东向 (m)"}},
      y: {min: mapCy - mapHalf, max: mapCy + mapHalf,
          title: {display: true, text: "北向 (m)"}}
    }
  }
});

function speedColor(s, lo, hi) {
  const t = Math.max(0, Math.min(1, (s - lo) / Math.max(hi - lo, 1e-6)));
  // turbo 色带的简化版：蓝 → 青 → 绿 → 黄 → 红
  const stops = [[48,88,200],[34,180,200],[60,200,110],[230,200,60],[225,60,45]];
  const f = t * (stops.length - 1);
  const i = Math.min(Math.floor(f), stops.length - 2);
  const k = f - i;
  const c = stops[i].map((v,j) => Math.round(v + (stops[i+1][j] - v) * k));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

// ---------- 每圈走线：把各圈的实测轨迹叠在一起 ----------
// 各圈的轨迹已经在同一张距离网格上对齐（同一下标 = 距起点线同样的距离），
// 所以直接画就是从同一个起跑点出发的一束线。
const LINES = DATA.laps.filter(l => l.valid && l.path && l.path.length);
if (LINES.length && DATA.line_extent) {
  const ext = DATA.line_extent;
  const cLines = new Chart(document.getElementById("cLines"), {
    type: "line",
    data: {datasets: LINES.map((l, i) => ({
      label: "#" + l.index + " " + l.time + (isBest(l) ? " ★" : ""),
      data: l.path.map(p => ({x: p[0], y: p[1]})),
      borderColor: color(i, l),
      borderWidth: isBest(l) ? 3 : 1.4,
      pointRadius: 0, pointHitRadius: 0, fill: false, tension: 0
    }))},
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: {mode: "nearest", intersect: false},
      plugins: {
        legend: {labels: {boxWidth: 14, boxHeight: 3, font: {size: 11}}},
        tooltip: {callbacks: {label: c => c.dataset.label}}
      },
      // 等比例：x、y 用同一个半径（见 DATA.line_extent），容器再来一个正方形。
      // x 必须显式声明 type:"linear" —— line 类型的 x 轴**默认是 category**，
      // 而我们的 x 是 559xxx 这样的数值坐标，当成类别会变成几十万个刻度，
      // 曲线全挤到最左边（踩过）。scatter 默认就是 linear，所以赛道俯视图不用写。
      scales: {
        x: {type: "linear", min: ext.cx - ext.half, max: ext.cx + ext.half,
            title: {display: true, text: "东向 (m)"}},
        y: {min: ext.cy - ext.half, max: ext.cy + ext.half,
            title: {display: true, text: "北向 (m)"}}
      }
    }
  });
  registerLapChart(cLines, LINES.map(l => l.index));
}

// ---------- 每圈走线偏差：相对平均线的横向偏移 ----------
// 为什么单给一张：GPS 单点噪声 1~3 米，而真实走线差异往往只有零点几米 ——
// 在叠加图上噪声会把差异盖住，看起来"所有圈都跑在同一条线上"。
// 把相对平均线的偏移单独画出来，差异才量化得出来。
if (LINES.length) {
  const cDev = new Chart(document.getElementById("cDev"), {
    type: "line",
    data: {datasets: LINES.map((l, i) => ({
      label: "#" + l.index + " " + l.time + (isBest(l) ? " ★" : ""),
      data: (l.dev || []).map((v, k) => ({x: DATA.grid[k], y: v})),
      borderColor: color(i, l),
      borderWidth: isBest(l) ? 2.6 : 1.2,
      pointRadius: 0, pointHitRadius: 0, fill: false, tension: 0
    }))},
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: {mode: "nearest", intersect: false},
      plugins: {
        legend: {labels: {boxWidth: 14, boxHeight: 3, font: {size: 11}}},
        bands: {bands: DATA.corner_bands},
        tooltip: {callbacks: {
          label: c => c.dataset.label + "：" + (c.raw.y >= 0 ? "偏左 " : "偏右 ")
                      + Math.abs(c.raw.y).toFixed(2) + " m"
        }}
      },
      scales: {
        x: {type: "linear", title: {display: true, text: "距起点线 (m)"}},
        y: {title: {display: true, text: "相对平均走线的横向偏移 (m)"},
            ticks: {callback: v => v === 0 ? "0（平均线）" : v}}
      }
    }
  });
  registerLapChart(cDev, LINES.map(l => l.index));
}
}  // end Chart available

// ---------- 表格（不依赖 Chart.js） ----------
(function renderTables(){
  const fmt = v => (v===null||v===undefined) ? "—" : v;
  const td = (v, cls) => "<td" + (cls ? ' class="' + cls + '"' : "") + ">" + v + "</td>";

  // ---- 弯道表 ----
  const cHead = ["弯","方向","起点","顶点","长度","半径","顶点速度"].concat(
    DATA.laps.map(l => "#" + l.index)
  ).concat(["速度波动","刹车点离散"]);
  let html = "<thead><tr>" + cHead.map(h => "<th>" + h + "</th>").join("") + "</tr></thead><tbody>";
  for (const c of DATA.corners) {
    let row = "<tr>";
    row += td(c.index) + td(c.direction === "L" ? "左" : "右");
    row += td(c.d_start_m.toFixed(0)) + td(c.d_apex_m.toFixed(0));
    row += td((c.d_end_m - c.d_start_m).toFixed(0)) + td(fmt(c.radius_m));
    row += td(c.apex_avg_kmh.toFixed(1));
    c.apex_speed_kmh.forEach((v, i) => {
      row += td(v.toFixed(1), i === c.best_lap_idx ? "best" : "");
    });
    row += td(fmt(c.spread_kmh)) + td(fmt(c.brake_spread_m));
    html += row + "</tr>";
  }
  document.getElementById("tCorner").innerHTML = html + "</tbody>";

  // ---- 逐圈明细表 ----
  const lHead = ["圈号","圈时","Δ最快"].concat(DATA.sector_labels)
    .concat(["里程","极速","最慢","平均速","峰值横G","峰值刹G","峰值加速","速度增量","加速占比"]);
  let h2 = "<thead><tr>" + lHead.map(h => "<th>" + h + "</th>").join("") + "</tr></thead><tbody>";
  for (const l of DATA.laps) {
    const delta = l.duration_s - BEST.duration_s;
    const dTxt = isBest(l) ? "—" : (delta >= 0 ? "+" : "") + delta.toFixed(3);
    const mark = l.valid ? "" : " *";
    const cls = !l.valid ? "dim" : (isBest(l) ? "best" : "");
    let row = "<tr>";
    row += td("#" + l.index + mark, cls);
    row += td(l.time, cls);
    row += td(dTxt, cls);
    l.sectors_s.forEach(s => row += td(s.toFixed(3), cls));
    row += td(l.length_m.toFixed(0), cls) + td(l.max_speed_kmh.toFixed(1), cls);
    row += td(l.min_speed_kmh.toFixed(1), cls) + td(l.avg_speed_kmh.toFixed(1), cls);
    row += td(l.peak_lat_g.toFixed(2), cls) + td(l.peak_brake_g.toFixed(2), cls);
    row += td(l.peak_accel_g.toFixed(2), cls) + td(l.speed_gain_ms.toFixed(0), cls);
    row += td(l.accel_time_pct.toFixed(0) + "%", cls);
    h2 += row + "</tr>";
  }
  document.getElementById("tLap").innerHTML = h2 + "</tbody>";
})();
</script>
</body>
</html>
"""


def _subsample(arr: np.ndarray, n: int) -> np.ndarray:
    """把长数组抽稀到 n 个点，控制 HTML 体积。"""
    m = arr.size
    if m <= n:
        return arr
    idx = np.linspace(0, m - 1, n).astype(int)
    return arr[idx]


# 输出目录里图表 / 数据表的文件名 → 一句话说明。
# 看板只放"结论"，细节大图和原始数据留在这些文件里，所以页尾要给出入口。
_CHART_FILES = [
    ("lap_times.png", "圈速分布"),
    ("speed_trace.png", "速度—距离曲线"),
    ("delta.png", "相对最快圈的时间差"),
    ("gg_diagram.png", "G-G 摩擦圆"),
    ("track_map.png", "赛道俯视图"),
    ("lap_lines.png", "每圈走线对比"),
    ("corner_apex.png", "逐弯顶点速度"),
]
_TABLE_FILES = [
    ("laps.csv", "每圈汇总"),
    ("corners.csv", "弯道 × 圈 明细"),
    ("laps_aligned_speed.csv", "距离对齐的速度"),
    ("laps_aligned_glat.csv", "距离对齐的横向 G"),
    ("telemetry.csv", "全场高频遥测"),
    ("analysis.json", "结构化结果，喂给别的工具"),
]


def _file_index(outdir: Path) -> str:
    """
    生成页尾的"其他文件"区块。

    按磁盘上**实际存在**的文件来列，所以加了 --no-charts / --no-csv
    之后不会留下点不开的死链接；一个都没有时整块不输出。
    """
    groups: list[str] = []
    for folder, title, items in (
        ("charts", "图表（点开看大图）", _CHART_FILES),
        ("tables", "数据表（Excel / pandas 可直接打开）", _TABLE_FILES),
    ):
        links = "".join(
            f'<a href="{folder}/{name}">{name}<span>{desc}</span></a>'
            for name, desc in items
            if (outdir / folder / name).exists()
        )
        if links:
            groups.append(f"<h3>{title}</h3><div class=\"files\">{links}</div>")
    if not groups:
        return ""
    return (
        "  <section>\n"
        "    <h2>其他文件</h2>\n"
        '    <p class="hint">这一页只放结论。想要单张大图发朋友圈，或者拿原始数据自己做表，'
        "从下面拿 —— 它们都在本文件旁边的子目录里。</p>\n"
        "    " + "\n    ".join(groups) + "\n"
        "  </section>\n\n"
    )


def _cards(sa: ana.SessionAnalysis) -> str:
    ls = sa.lapset
    t = ls.telemetry
    best = ls.best_lap
    max_lat = max((l.peak_lat_g for l in ls.laps), default=0.0)
    max_brk = max((l.peak_brake_g for l in ls.laps), default=0.0)
    max_spd = max((l.max_speed for l in ls.laps), default=0.0)

    items = [
        ("最快圈", laps.format_lap_time(best.duration) if best else "—",
         f"第 {best.index} 圈" if best else "", "best"),
        ("理论最佳圈", laps.format_lap_time(ls.theoretical_best), "各分段最好之和", ""),
        ("连续分段最佳", laps.format_lap_time(sa.rolling_best), "真实可达极限", "accent"),
        ("平均圈", laps.format_lap_time(ls.mean_lap), f"标准差 {ls.std_lap:.3f}s", ""),
        ("稳定性", f"{ls.std_lap / ls.mean_lap * 100:.2f}%" if ls.mean_lap else "—", "变异系数，越小越稳", ""),
        ("最高速度", f"{max_spd * 3.6:.1f}", "km/h", ""),
        ("最大横向G", f"{max_lat:.2f}", "g", ""),
        ("最大刹车G", f"{max_brk:.2f}", "g", ""),
    ]
    if t is not None and t.gfield is not None:
        items.append(("G 值可信度", f"{t.gfield.quality:.2f}", "与 GPS 参考的一致性 (0~1)", "accent"))

    out = []
    for k, v, sub, cls in items:
        small = f"<small>{sub}</small>" if sub else ""
        out.append(
            f'<div class="card {cls}"><div class="k">{k}</div>'
            f'<div class="v">{v}{small}</div></div>'
        )
    return "\n    ".join(out)


def _check_inline_js(html: str) -> str | None:
    """
    用 node 检查内嵌脚本的语法，返回错误信息（没问题则返回 None）。

    为什么要专门做这一步：看板的图表代码全是内嵌 JS，一旦有语法错误，
    整段脚本都不会执行，页面会变成"标题正常、表格能看、图表全白"。
    这种故障从 Python 侧完全看不出来 —— 文件生成成功了，HTML 也是合法的，
    必须真的把 JS 解析一遍才能发现。（踩过：两个 `const` 重名，
    `Identifier 'yLo' has already been declared`，五个图表全部不显示）
    """
    import re
    import shutil
    import subprocess
    import tempfile

    node = shutil.which("node")
    if not node:
        return None  # 没装 node 就跳过，不影响出结果
    blocks = re.findall(r"<script>(.*?)</script>", html, re.S)
    if not blocks:
        return None

    with tempfile.NamedTemporaryFile(
        "w", suffix=".js", delete=False, encoding="utf-8"
    ) as f:
        f.write(blocks[-1])
        tmp = f.name
    try:
        proc = subprocess.run([node, "--check", tmp], capture_output=True, text=True)
        if proc.returncode == 0:
            return None
        lines = [ln.strip() for ln in (proc.stderr or "").splitlines() if ln.strip()]
        # node 的输出顺序是「临时文件路径:行号」→ 源码 → 箭头 → 「SyntaxError: 说明」，
        # 真正有用的是最后那行，别把临时文件路径当成错误信息报给用户
        for ln in lines:
            if "Error" in ln:
                return ln
        return lines[0] if lines else "未知语法错误"
    except OSError:
        return None
    finally:
        Path(tmp).unlink(missing_ok=True)


def build(sa: ana.SessionAnalysis, path: str | Path, *, keep_laps: int = 16, points_per_lap: int = 260) -> Path:
    """
    生成单文件 HTML 看板。

    keep_laps / points_per_lap 用来控制文件体积 —— 圈数太多或采样太密会让
    HTML 变得很大，浏览器渲染也会卡。默认值对 10~20 分钟的练习刚好。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ls = sa.lapset

    laps_used = ls.laps[:keep_laps]
    grid = _subsample(ls.grid, points_per_lap)
    gi = np.linspace(0, ls.grid.size - 1, grid.size).astype(int)

    lap_json = []
    lap_objs: list[laps.Lap] = []
    for k, lap in enumerate(ls.laps):
        if lap not in laps_used:
            continue
        dl = sa.deltas[k]
        lap_objs.append(lap)
        lap_json.append({
            "index": lap.index,
            "valid": lap.valid,
            "duration_s": round(lap.duration, 3),
            "time": laps.format_lap_time(lap.duration),
            "sectors_s": [round(s, 3) for s in lap.sectors],
            "length_m": round(lap.length, 1),
            "max_speed_kmh": round(lap.max_speed * 3.6, 1),
            "min_speed_kmh": round(lap.min_speed * 3.6, 1),
            "avg_speed_kmh": round(float(np.mean(lap.speed)) * 3.6, 1),
            "peak_lat_g": round(lap.peak_lat_g, 2),
            "peak_brake_g": round(lap.peak_brake_g, 2),
            "peak_accel_g": round(lap.peak_accel_g, 2),
            "speed_gain_ms": round(lap.speed_gain_ms, 1),
            "accel_time_pct": round(lap.accel_time_pct, 1),
            "speed_kmh": [round(float(v), 2) for v in lap.speed[gi] * 3.6],
            "delta_s": [round(float(v), 3) for v in dl[gi]],
        })

    # ---- 每圈走线（交互图用）----
    # 「偏差」必须在这里算，而不能在图表 JS 里算：它是"各圈相对**平均走线**"的量，
    # 单独一圈算不出来。法向取 +落在行进方向左手边（正 = 偏左）。
    geo_laps = [l for l in lap_objs if l.valid and l.gx is not None]
    if geo_laps:
        mx = np.mean([l.gx for l in geo_laps], axis=0)
        my = np.mean([l.gy for l in geo_laps], axis=0)
        tgx, tgy = np.gradient(mx), np.gradient(my)
        tn = np.hypot(tgx, tgy)
        tn = np.where(tn < 1e-9, 1e-9, tn)
        nx_, ny_ = -tgy / tn, tgx / tn
    path_x: list[float] = []
    path_y: list[float] = []
    for item, lap in zip(lap_json, lap_objs):
        if lap.gx is None or not geo_laps:
            item["path"], item["dev"] = [], []
            continue
        item["path"] = [[round(float(lap.gx[i]), 1), round(float(lap.gy[i]), 1)]
                        for i in gi]
        off = (lap.gx - mx) * nx_ + (lap.gy - my) * ny_
        item["dev"] = [round(float(off[i]), 2) for i in gi]
        if lap.valid:
            path_x.extend(item["path"][j][0] for j in range(len(gi)))
            path_y.extend(item["path"][j][1] for j in range(len(gi)))

    lat, lon = ana.gg_points(ls)
    n_gg = 4000
    if lat.size > n_gg:
        gidx = np.linspace(0, lat.size - 1, n_gg).astype(int)
        lat, lon = lat[gidx], lon[gidx]
    ang, rad = ana.friction_envelope(*ana.gg_points(ls), bins=48)
    env_lat = (rad * np.cos(ang)).tolist() if ang.size else []
    env_lon = (rad * np.sin(ang)).tolist() if ang.size else []
    gg_max = float(max(np.max(np.abs(lat)) if lat.size else 1.0,
                       np.max(np.abs(lon)) if lon.size else 1.0))

    # 赛道俯视图：用所有有效圈的平均走线，单圈 GPS 轨迹毛刺太多
    valid_laps = [l for l in laps_used if l.valid and l.gx is not None]
    if valid_laps:
        cgx = np.mean([l.gx for l in valid_laps], axis=0)
        cgy = np.mean([l.gy for l in valid_laps], axis=0)
        csp = np.mean([l.speed for l in valid_laps], axis=0)
        tx, ty, tspd = _subsample(cgx, 700), _subsample(cgy, 700), _subsample(csp * 3.6, 700)
    else:
        tx = ty = tspd = np.zeros(0)

    g = ls.gate
    nvec = np.array([-g.direction[1], g.direction[0]])
    gate_line = [[g.x - nvec[0] * 14, g.y - nvec[1] * 14],
                 [g.x + nvec[0] * 14, g.y + nvec[1] * 14]]

    corner_json = []
    for c in sa.corners:
        corner_json.append({
            "index": c.index,
            "direction": "L" if c.direction > 0 else "R",
            "d_start_m": c.d_start,
            "d_apex_m": c.d_apex,
            "d_end_m": c.d_end,
            "radius_m": None if not np.isfinite(c.radius) else round(c.radius, 1),
            "apex_avg_kmh": round(float(np.mean(c.valid_apex_speed())) * 3.6, 1),
            "apex_speed_kmh": [round(float(v) * 3.6, 1) for v in c.apex_speed],
            "best_lap_idx": c.best_lap_position(),
            "spread_kmh": round(c.apex_spread() * 3.6, 2) if np.isfinite(c.apex_spread()) else None,
            "brake_spread_m": None if not np.isfinite(c.brake_spread()) else round(c.brake_spread(), 1),
        })

    data = {
        "grid": [round(float(v), 1) for v in grid],
        "laps": lap_json,
        "sector_labels": [f"分段{i + 1}" for i in range(ls.sector_count)],
        "corner_bands": [[round(c.d_start, 1), round(c.d_end, 1)] for c in sa.corners],
        # 每圈走线图的等比例坐标窗口：x、y 取同一个半径，容器再做正方形，
        # 两者同时成立 1 米横向和 1 米纵向才占同样多的像素（Chart.js 没有内置等比例轴）
        "line_extent": ({
            "cx": float((min(path_x) + max(path_x)) / 2),
            "cy": float((min(path_y) + max(path_y)) / 2),
            "half": float(max(max(path_x) - min(path_x),
                              max(path_y) - min(path_y)) / 2 * 1.06),
        } if path_x else None),
        "mean_lap": round(ls.mean_lap, 3) if np.isfinite(ls.mean_lap) else None,
        "mean_lap_text": laps.format_lap_time(ls.mean_lap),
        "theoretical_best": round(ls.theoretical_best, 3) if np.isfinite(ls.theoretical_best) else None,
        "theoretical_best_text": laps.format_lap_time(ls.theoretical_best),
        "rolling_best": round(sa.rolling_best, 3) if np.isfinite(sa.rolling_best) else None,
        "rolling_best_text": laps.format_lap_time(sa.rolling_best),
        "corners": corner_json,
        "gg": {"lat": [round(float(v), 3) for v in lat],
               "lon": [round(float(v), 3) for v in lon],
               "env_lat": [round(float(v), 3) for v in env_lat],
               "env_lon": [round(float(v), 3) for v in env_lon],
               "max": gg_max},
        "track_map": {
            "x": [round(float(v), 1) for v in tx],
            "y": [round(float(v), 1) for v in ty],
            "speed_kmh": [round(float(v), 1) for v in tspd],
            "vmin": float(np.min(tspd)) if tspd.size else 0.0,
            "vmax": float(np.max(tspd)) if tspd.size else 1.0,
            "gate": [[round(v, 1) for v in p] for p in gate_line],
        },
    }

    t = ls.telemetry
    html = (
        _TEMPLATE
        .replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
        .replace("__CARDS__", _cards(sa))
        .replace("__TITLE__", f"卡丁车遥测分析 — {t.source if t else ''}")
        .replace("__SOURCE__", t.source if t else "")
        .replace("__BESTLAP__", laps.format_lap_time(ls.best_lap.duration) if ls.best_lap else "—")
        .replace("__NLAPS__", str(len(ls.laps)))
        .replace("__DIST__", f"{float(t.dist[-1]):.0f}" if t else "—")
        .replace("__DATE__", __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M"))
        .replace("__FILES__", _file_index(path.parent))
    )
    path.write_text(html, encoding="utf-8")

    # 安全网：真的把内嵌脚本解析一遍。语法错的话整个看板的图表都会白屏，
    # 而从 Python 这边看文件是正常生成的，根本看不出来。
    err = _check_inline_js(html)
    if err:
        print(f"  ⚠ 看板内嵌脚本有语法错误，图表将无法显示：{err}")
        print(f"    （文件已写在 {path}，但用浏览器打开会看到空白图表）")

    return path


__all__ = ["build"]
