"""Build a self-contained dashboard from results.json (no hand-typed numbers)."""
import json, html

R = json.load(open("results.json"))
META, CELLS, CONTRASTS = R["meta"], R["cells"], R["contrasts"]
FMTS, TEMPS = META["formats"], META["temps"]
PARSERS = META["parsers"]
SUITES = sorted({c["suite"] for c in CELLS}, key=lambda s: (s != "gsm8k", s))
PLABEL = {"parse_strict": "strict", "parse_flexible": "flexible",
          "parse_last_number": "permissive"}
FLABEL = {"bare": "bare", "cot_zero_shot": "CoT, no contract",
          "cot_tagged": "CoT + '#### x'", "fewshot_tagged": "3-shot + '#### x'"}

def cell(s, f, t, p):
    for c in CELLS:
        if c["suite"] == s and c["fmt"] == f and c["temp"] == t and c["parser"] == p:
            return c
    return None

# ---- derived: temperature-effect decomposition, per suite/format ----------
decomp = {}
for s in SUITES:
    rows = []
    for f in FMTS:
        a0, a1 = cell(s, f, TEMPS[0], "parse_strict"), cell(s, f, TEMPS[-1], "parse_strict")
        b0, b1 = cell(s, f, TEMPS[0], "parse_flexible"), cell(s, f, TEMPS[-1], "parse_flexible")
        if not all([a0, a1, b0, b1]):
            continue
        reported = a0["acc"] - a1["acc"]
        reasoning = b0["acc"] - b1["acc"]
        rows.append({"fmt": f, "reported": reported, "reasoning": reasoning,
                     "drift": reported - reasoning,
                     "u0": a0["unparsed_rate"], "u1": a1["unparsed_rate"]})
    decomp[s] = rows

spread = {}
for s in SUITES:
    g = [c for c in CELLS if c["suite"] == s]
    lo, hi = min(g, key=lambda c: c["acc"]), max(g, key=lambda c: c["acc"])
    spread[s] = {"lo": lo, "hi": hi, "range": hi["acc"] - lo["acc"]}

PAYLOAD = json.dumps({"meta": META, "cells": CELLS, "contrasts": CONTRASTS,
                      "decomp": decomp, "spread": spread,
                      "plabel": PLABEL, "flabel": FLABEL, "suites": SUITES})

CSS = """
:root{
  --paper:#EDF0F3; --ink:#12181F; --ink-2:#4A5561; --rule:#C6CDD6; --card:#FFFFFF;
  --p1:#0B3D5C; --p2:#3C7EA6; --p3:#9FC8DF;   /* parser permissiveness ramp */
  --warn:#C1741A;                              /* reserved: data loss only   */
  --flag:#7A2E2E; --flag-bg:#F6E9E7;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--paper);color:var(--ink);
  font-family:"IBM Plex Sans",ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
  font-size:16px;line-height:1.5}
.mono{font-family:"IBM Plex Mono",ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  font-variant-numeric:tabular-nums}
.wrap{max-width:1080px;margin:0 auto;padding:0 24px}

.flag{background:var(--flag-bg);border-bottom:2px solid var(--flag);
  color:var(--flag);padding:14px 0}
.flag .wrap{display:flex;gap:14px;align-items:flex-start}
.flag b{font-weight:700;letter-spacing:.04em;text-transform:uppercase;font-size:12px;
  border:1.5px solid var(--flag);padding:3px 7px;white-space:nowrap;margin-top:1px}
.flag p{margin:0;font-size:14px;line-height:1.45;max-width:75ch}

header{padding:56px 0 8px}
.eyebrow{font-size:12px;letter-spacing:.16em;text-transform:uppercase;
  color:var(--ink-2);margin:0 0 18px}
h1{font-family:"Archivo",ui-sans-serif,system-ui,sans-serif;font-weight:700;
  font-size:clamp(34px,6vw,60px);line-height:1.02;letter-spacing:-.025em;margin:0;
  max-width:16ch}
h1 em{font-style:normal;color:var(--p2)}
.stand{margin:20px 0 0;font-size:19px;color:var(--ink-2);max-width:62ch}

section{padding:52px 0}
h2{font-family:"Archivo",ui-sans-serif,system-ui,sans-serif;font-size:13px;font-weight:700;
  letter-spacing:.15em;text-transform:uppercase;margin:0 0 6px;
  padding-bottom:8px;border-bottom:2px solid var(--ink)}
.note{color:var(--ink-2);font-size:15px;margin:14px 0 26px;max-width:68ch}

/* ---- signature: the dial ---- */
.dial{background:var(--card);border:1px solid var(--rule);padding:30px 28px 22px}
.readout{display:flex;align-items:baseline;gap:16px;flex-wrap:wrap;
  min-height:76px;margin-bottom:6px}
.readout .big{font-size:clamp(46px,9vw,76px);font-weight:600;line-height:.95;
  letter-spacing:-.03em;color:var(--p1)}
.readout .desc{font-size:14px;color:var(--ink-2);line-height:1.45}
.readout .desc b{color:var(--ink);font-weight:600}
.axis{width:100%;height:112px;display:block;touch-action:none}
.axis .tick{cursor:pointer}
.legend{display:flex;gap:20px;flex-wrap:wrap;margin-top:14px;
  font-size:12px;color:var(--ink-2)}
.legend i{display:inline-block;width:11px;height:11px;margin-right:6px;
  vertical-align:-1px;border-radius:50%}
.tabs{display:flex;gap:0;margin-bottom:18px;border:1px solid var(--rule);width:max-content}
.tabs button{font:inherit;font-size:13px;padding:8px 18px;background:var(--card);
  border:0;border-right:1px solid var(--rule);cursor:pointer;color:var(--ink-2)}
.tabs button:last-child{border-right:0}
.tabs button[aria-selected=true]{background:var(--ink);color:#fff}
.tabs button:focus-visible{outline:2px solid var(--p2);outline-offset:-2px}

/* ---- grid ---- */
table{width:100%;border-collapse:collapse;background:var(--card);
  border:1px solid var(--rule)}
th,td{padding:9px 12px;text-align:right;font-size:14px;border-bottom:1px solid #E4E8ED}
th{font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--ink-2);
  font-weight:600;border-bottom:1.5px solid var(--rule)}
th:first-child,td:first-child{text-align:left}
tbody tr:last-child td{border-bottom:0}
tr.grp td{background:#F2F5F8;font-weight:600;font-size:12px;letter-spacing:.07em;
  text-transform:uppercase;color:var(--ink-2)}
.ci{color:var(--ink-2);font-size:12px;margin-left:5px}
.u-hi{color:var(--warn);font-weight:600}

.cols{display:grid;grid-template-columns:1fr 1fr;gap:32px}
@media(max-width:860px){.cols{grid-template-columns:1fr}}
figure{margin:0}
figcaption{font-size:13px;color:var(--ink-2);margin-top:12px;line-height:1.5}

pre{background:var(--ink);color:#DCE3EA;padding:16px 18px;overflow-x:auto;
  font-size:13px;line-height:1.65;border-radius:2px}
pre b{color:#8FC4DE;font-weight:600}
footer{border-top:1px solid var(--rule);padding:30px 0 60px;font-size:13px;
  color:var(--ink-2)}
footer p{max-width:72ch}
a{color:var(--p1)}
@media(prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
.tick,.readout .big{transition:opacity .12s ease}
"""

JS = r"""
const D = __PAYLOAD__;
const PC = {parse_strict:'var(--p1)', parse_flexible:'var(--p2)', parse_last_number:'var(--p3)'};
let suite = D.suites[0];

const pct = v => (v*100).toFixed(1)+'%';

/* ---------- signature dial ---------- */
function drawDial(){
  const W=1000, H=112, L=34, R=34, y=64;
  const cells = D.cells.filter(c=>c.suite===suite);
  const x = a => L + a*(W-L-R);
  let s = `<svg class="axis" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none"
            role="img" aria-label="All reportable scores for one model">`;
  s += `<line x1="${x(0)}" y1="${y}" x2="${x(1)}" y2="${y}" stroke="var(--rule)" stroke-width="1.5"/>`;
  for(let i=0;i<=10;i+=2){
    s+=`<line x1="${x(i/10)}" y1="${y}" x2="${x(i/10)}" y2="${y+8}" stroke="var(--rule)"/>`;
    s+=`<text x="${x(i/10)}" y="${y+26}" font-size="11" fill="#4A5561"
         text-anchor="middle" font-family="IBM Plex Mono,monospace">${i*10}%</text>`;
  }
  cells.forEach((c,i)=>{
    s+=`<line class="tick" data-i="${i}" x1="${x(c.acc)}" y1="${y-34}" x2="${x(c.acc)}" y2="${y}"
        stroke="${PC[c.parser]}" stroke-width="3" opacity=".78" tabindex="0"/>`;
  });
  s+='</svg>';
  document.getElementById('dial').innerHTML = s;

  const svg = document.querySelector('.axis');
  const show = i => {
    const c = cells[i];
    document.getElementById('readout').innerHTML =
      `<div class="big mono">${pct(c.acc)}</div>
       <div class="desc">prompt <b>${D.flabel[c.fmt]}</b> &nbsp;·&nbsp; temperature <b class="mono">${c.temp}</b>
        &nbsp;·&nbsp; <b>${D.plabel[c.parser]}</b> answer extraction<br>
        95% CI <span class="mono">[${pct(c.ci_lo)}, ${pct(c.ci_hi)}]</span>
        &nbsp;·&nbsp; unreadable completions <span class="mono">${pct(c.unparsed_rate)}</span></div>`;
    svg.querySelectorAll('.tick').forEach(t=>t.setAttribute('opacity', t.dataset.i==i?'1':'.28'));
  };
  svg.querySelectorAll('.tick').forEach(t=>{
    t.addEventListener('mouseenter',()=>show(t.dataset.i));
    t.addEventListener('focus',()=>show(t.dataset.i));
  });
  const sp = D.spread[suite];
  const hi = cells.findIndex(c=>c.acc===sp.hi.acc);
  show(hi<0?0:hi);
  document.getElementById('claim').textContent =
    `${pct(sp.lo.acc)} to ${pct(sp.hi.acc)}`;
  document.getElementById('claimpp').textContent =
    `${(sp.range*100).toFixed(1)} points`;
}

/* ---------- condition table ---------- */
function drawTable(){
  const rows = D.meta.formats.map(f=>{
    let out = `<tr class="grp"><td colspan="5">${D.flabel[f]}</td></tr>`;
    D.meta.temps.forEach(t=>{
      const g = p => D.cells.find(c=>c.suite===suite&&c.fmt===f&&c.temp===t&&c.parser===p);
      const st=g('parse_strict'), fl=g('parse_flexible'), ln=g('parse_last_number');
      if(!st) return;
      const u = st.unparsed_rate;
      out += `<tr><td class="mono">T = ${t.toFixed(1)}</td>
        <td class="mono">${pct(st.acc)}<span class="ci">±${((st.ci_hi-st.ci_lo)/2*100).toFixed(1)}</span></td>
        <td class="mono">${pct(fl.acc)}<span class="ci">±${((fl.ci_hi-fl.ci_lo)/2*100).toFixed(1)}</span></td>
        <td class="mono">${pct(ln.acc)}<span class="ci">±${((ln.ci_hi-ln.ci_lo)/2*100).toFixed(1)}</span></td>
        <td class="mono ${u>=0.3?'u-hi':''}">${pct(u)}</td></tr>`;
    });
    return out;
  }).join('');
  document.getElementById('tbody').innerHTML = rows;
}

/* ---------- decomposition ---------- */
function drawDecomp(){
  const rows = D.decomp[suite].filter(r=>r.reported>0.001);
  const W=520, rowH=54, H=rows.length*rowH+34, L=140;
  const max = Math.max(...rows.map(r=>Math.max(r.reported,0.01)));
  const x = v => L + (v/max)*(W-L-24);
  let s=`<svg viewBox="0 0 ${W} ${H}" width="100%" role="img"
    aria-label="Temperature effect split into reasoning loss and format drift">`;
  rows.forEach((r,i)=>{
    const y=i*rowH+12, h=17;
    const wR=x(Math.max(r.reasoning,0))-L, wD=x(r.reported)-x(Math.max(r.reasoning,0));
    s+=`<text x="0" y="${y+13}" font-size="12" fill="#12181F">${D.flabel[r.fmt]}</text>`;
    s+=`<rect x="${L}" y="${y}" width="${Math.max(wR,0)}" height="${h}" fill="var(--p2)"/>`;
    s+=`<rect x="${x(Math.max(r.reasoning,0))}" y="${y}" width="${Math.max(wD,0)}" height="${h}" fill="var(--warn)"/>`;
    s+=`<text x="${x(r.reported)+7}" y="${y+13}" font-size="11.5" fill="#4A5561"
        font-family="IBM Plex Mono,monospace">${(r.reported*100).toFixed(1)} pp</text>`;
    s+=`<text x="${L}" y="${y+34}" font-size="11" fill="#4A5561">`+
       `unreadable ${(r.u0*100).toFixed(0)}% → ${(r.u1*100).toFixed(0)}%</text>`;
  });
  s+='</svg>';
  document.getElementById('decomp').innerHTML=s;
}

function render(){ drawDial(); drawTable(); drawDecomp(); }
document.querySelectorAll('.tabs button').forEach(b=>{
  b.addEventListener('click',()=>{
    suite=b.dataset.suite;
    document.querySelectorAll('.tabs button').forEach(o=>
      o.setAttribute('aria-selected', String(o===b)));
    render();
  });
});
render();
"""

def suite_label(s):
    return "GSM8K" if s == "gsm8k" else "BBH · date understanding"

tabs = "".join(
    f'<button data-suite="{s}" aria-selected="{str(i==0).lower()}">{suite_label(s)}</button>'
    for i, s in enumerate(SUITES))

DOC = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Eval fragility · one model, many scores</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700&family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body>

<div class="flag"><div class="wrap">
  <b>Synthetic</b>
  <p>No real model was evaluated. This sandbox blocks <span class="mono">huggingface.co</span>
  and has no inference API key, so completions come from a stipulated offline stub. The benchmark
  items, the extraction rules, the statistics and this page are real; <b>the accuracy levels and the
  size of the temperature effect are not measurements of anything.</b> Point the harness at a live
  model to replace them.</p>
</div></div>

<div class="wrap">
<header>
  <p class="eyebrow">Inspect AI · GSM8K + BIG-Bench Hard · n = {META['n_per_condition']} per condition</p>
  <h1>One model. <em id="claim">36 defensible scores.</em></h1>
  <p class="stand">Fix the model and the questions. Vary only the prompt format, the sampling
  temperature, and the regex that reads the answer out. The reported score moves by
  <span id="claimpp" class="mono"></span> — before anyone has changed a single weight.</p>
</header>

<section>
  <h2>Every number you could have published</h2>
  <p class="note">Each tick is one complete evaluation run. Hover or tab through them to see
  which methodology produces which headline.</p>
  <div class="tabs" role="tablist">{tabs}</div>
  <div class="dial">
    <div class="readout mono" id="readout"></div>
    <div id="dial"></div>
    <div class="legend">
      <span><i style="background:var(--p1)"></i>strict — requires the <span class="mono">#### x</span> contract</span>
      <span><i style="background:var(--p2)"></i>flexible — also accepts "the answer is x"</span>
      <span><i style="background:var(--p3)"></i>permissive — last number in the completion</span>
    </div>
  </div>
</section>

<section>
  <h2>The condition grid</h2>
  <p class="note">The three accuracy columns score the <em>same</em> completions. Any gap between
  them is produced entirely by the answer-extraction rule. The final column is the share of
  completions no strict parser could read — the quiet failure that gets recorded as
  a wrong answer.</p>
  <table><thead><tr><th>condition</th><th>strict</th><th>flexible</th>
    <th>permissive</th><th>unreadable</th></tr></thead>
    <tbody id="tbody"></tbody></table>
</section>

<section class="cols">
  <figure>
    <h2>What the temperature effect really is</h2>
    <p class="note">Raising temperature makes a model reason worse — and also makes it stop
    following the output format. A strict parser charges both to the same account.</p>
    <div id="decomp"></div>
    <figcaption><span style="color:var(--p2)">■</span> genuine reasoning loss (visible to a
    flexible parser) &nbsp; <span style="color:var(--warn)">■</span> format drift misread as
    wrong answers. Bars show the drop from T=0.0 to T=1.0 under the strict parser.</figcaption>
  </figure>
  <figure>
    <h2>Reproducing this</h2>
    <p class="note">The offline stub is the only synthetic component, and it is one flag.
    Everything else — items, prompts, parsers, bootstrap CIs, McNemar tests — runs unchanged
    against a real endpoint.</p>
<pre># stub, no network needed
python3 run_grid.py -n 200

# a real open model, served locally
python3 run_grid.py <b>--live</b> \\
  --model vllm/Qwen/Qwen2.5-7B-Instruct -n 200

python3 build_dashboard.py</pre>
    <figcaption>Paired McNemar tests are valid here because both parsers see identical
    completions, so the pairing is exact and the model contributes zero variance to the
    contrast.</figcaption>
  </figure>
</section>

<footer>
  <p><b>Method.</b> GSM8K test split (openai/grade-school-math) and BIG-Bench Hard
  (suzgunmirac/BIG-Bench-Hard), items sampled with a fixed seed and held identical across all
  conditions. Confidence intervals are percentile bootstrap over items, 5,000 resamples;
  at n={META['n_per_condition']} a single condition is only precise to roughly ±7 points, so
  the wide contrasts here matter and the narrow ones do not. Harness: Inspect AI
  {META.get('harness_version','0.3.251')}.</p>
</footer>
</div>
<script>{JS.replace('__PAYLOAD__', PAYLOAD)}</script>
</body></html>"""

open("dashboard.html", "w").write(DOC)
print(f"dashboard.html written ({len(DOC)/1024:.1f} KB)")
print(f"  suites={SUITES}  cells={len(CELLS)}  contrasts={len(CONTRASTS)}")
for s in SUITES:
    sp = spread[s]
    print(f"  {s}: {sp['lo']['acc']:.1%} -> {sp['hi']['acc']:.1%} "
          f"({sp['range']*100:.1f} pp spread)")
