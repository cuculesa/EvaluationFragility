# ruff: noqa: E501
from __future__ import annotations

import html
import json
from pathlib import Path

from .schema import validate_results
from .util import atomic_write_text, read_json_object

CSS = r"""
:root{--bg:#eef1f4;--card:#fff;--ink:#111820;--muted:#52606d;--rule:#c9d1da;
--navy:#123f5b;--blue:#4d89ad;--pale:#acd1e4;--amber:#a65f0a;--red:#842d2d;
--redbg:#f7e9e7;--green:#175c45;--greenbg:#e7f3ee}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}.mono{font-family:ui-monospace,
SFMono-Regular,Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}.wrap{max-width:1120px;
margin:auto;padding:0 24px}.banner{border-bottom:2px solid;padding:13px 0}.banner.live{background:var(--greenbg);
color:var(--green);border-color:var(--green)}.banner.synthetic{background:var(--redbg);color:var(--red);
border-color:var(--red)}.banner b{letter-spacing:.12em;text-transform:uppercase;font-size:12px}
header{padding:50px 0 12px}.eyebrow{letter-spacing:.14em;text-transform:uppercase;color:var(--muted);
font-size:12px;margin:0 0 14px}h1{font-size:clamp(36px,6vw,64px);line-height:1;letter-spacing:-.035em;
margin:0;max-width:16ch}.lede{font-size:18px;color:var(--muted);max-width:70ch;margin:18px 0 0}
.meta{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:28px 0 0}.meta div{background:var(--card);
border:1px solid var(--rule);padding:14px}.meta span{display:block;color:var(--muted);font-size:11px;
letter-spacing:.1em;text-transform:uppercase}.meta b{display:block;margin-top:4px;overflow-wrap:anywhere}
section{padding:42px 0}h2{font-size:12px;letter-spacing:.15em;text-transform:uppercase;border-bottom:2px solid;
padding-bottom:8px;margin:0}.note{color:var(--muted);max-width:75ch;margin:12px 0 22px}.tabs{display:flex;
flex-wrap:wrap;margin:18px 0}.tabs button{border:1px solid var(--rule);border-right:0;background:var(--card);
padding:9px 16px;cursor:pointer}.tabs button:last-child{border-right:1px solid var(--rule)}.tabs button.active{
background:var(--ink);color:#fff}.range{background:var(--card);border:1px solid var(--rule);padding:26px}
.range-value{font-size:clamp(42px,7vw,76px);font-weight:700;letter-spacing:-.04em;color:var(--navy)}
.range-desc{color:var(--muted);margin-top:4px}.axis{height:85px;width:100%;display:block;margin-top:10px}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--rule)}th,td{
padding:9px 11px;border-bottom:1px solid #e3e7eb;text-align:right}th{font-size:11px;text-transform:uppercase;
letter-spacing:.08em;color:var(--muted)}th:first-child,td:first-child{text-align:left}.group td{background:#f4f6f8;
font-weight:700;color:var(--muted);text-transform:uppercase;font-size:11px;letter-spacing:.08em}.warn{color:var(--amber);
font-weight:700}.grid{display:grid;grid-template-columns:1.15fr .85fr;gap:28px}.panel{background:var(--card);
border:1px solid var(--rule);padding:20px}.bars{display:grid;gap:14px}.barrow{display:grid;grid-template-columns:150px 1fr 88px;
gap:10px;align-items:center}.track{height:18px;background:#e7ebef;position:relative}.bar{height:100%;position:absolute;left:50%}
.zero{position:absolute;left:50%;top:-4px;bottom:-4px;border-left:1px solid #87929d}.ci-line{position:absolute;top:7px;height:4px;
border-top:2px solid var(--ink)}.warnings{display:grid;gap:9px}.warning{background:#fff8e9;border-left:4px solid var(--amber);
padding:10px 12px}.provenance{display:grid;grid-template-columns:160px 1fr;gap:7px 12px;font-size:13px}.provenance dt{
color:var(--muted)}.provenance dd{margin:0;overflow-wrap:anywhere}footer{border-top:1px solid var(--rule);padding:28px 0 55px;
color:var(--muted);font-size:13px}@media(max-width:850px){.meta{grid-template-columns:1fr 1fr}.grid{grid-template-columns:1fr}}
@media(max-width:560px){.meta{grid-template-columns:1fr}.barrow{grid-template-columns:1fr}.wrap{padding:0 16px}}
@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important}}
"""

JS = r"""
const D=__DATA__;const F={bare:'Bare',cot_zero_shot:'CoT, no output contract',cot_tagged:'CoT + output contract',fewshot_tagged:'Few-shot + output contract'};
const P={parse_strict:'Strict',parse_flexible:'Flexible',parse_last_number:'Permissive'};
let suite=D.meta.suites[0];const pooled=()=>D.cells.filter(c=>c.seed===null&&c.suite===suite);
const pct=x=>(x*100).toFixed(1)+'%';const pp=x=>(x*100).toFixed(1)+' pp';
function el(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n}
function renderRange(){const cells=pooled();const lo=cells.reduce((a,b)=>a.acc<b.acc?a:b),hi=cells.reduce((a,b)=>a.acc>b.acc?a:b);
 document.getElementById('rangeValue').textContent=pct(lo.acc)+' → '+pct(hi.acc);document.getElementById('rangeDesc').textContent=
 `${pp(hi.acc-lo.acc)} spread across configured prompt, temperature, and extraction choices.`;
 const svg=document.getElementById('axis');svg.replaceChildren();svg.setAttribute('viewBox','0 0 1000 85');
 const ns='http://www.w3.org/2000/svg',x=v=>35+v*930;let line=document.createElementNS(ns,'line');line.setAttribute('x1',x(0));line.setAttribute('x2',x(1));line.setAttribute('y1',38);line.setAttribute('y2',38);line.setAttribute('stroke','#c9d1da');svg.append(line);
 for(let i=0;i<=10;i+=2){let t=document.createElementNS(ns,'text');t.setAttribute('x',x(i/10));t.setAttribute('y',68);t.setAttribute('text-anchor','middle');t.setAttribute('font-size','11');t.textContent=i*10+'%';svg.append(t)}
 const colors={parse_strict:'#123f5b',parse_flexible:'#4d89ad',parse_last_number:'#acd1e4'};cells.forEach(c=>{let tick=document.createElementNS(ns,'line');tick.setAttribute('x1',x(c.acc));tick.setAttribute('x2',x(c.acc));tick.setAttribute('y1',10);tick.setAttribute('y2',38);tick.setAttribute('stroke',colors[c.parser]);tick.setAttribute('stroke-width','3');let title=document.createElementNS(ns,'title');title.textContent=`${F[c.fmt]} · T=${c.temp} · ${P[c.parser]}: ${pct(c.acc)}`;tick.append(title);svg.append(tick)})}
function renderTable(){const body=document.getElementById('body');body.replaceChildren();for(const fmt of D.meta.prompt_formats){let g=el('tr');g.className='group';let d=el('td',F[fmt]);d.colSpan=5;g.append(d);body.append(g);for(const t of D.meta.temperatures){const find=p=>D.cells.find(c=>c.seed===null&&c.suite===suite&&c.fmt===fmt&&c.temp===t&&c.parser===p);const a=find('parse_strict'),b=find('parse_flexible'),c=find('parse_last_number');if(!a)continue;let tr=el('tr');tr.append(el('td','T = '+Number(t).toFixed(1),'mono'));for(const x of [a,b,c]){let td=el('td',pct(x.acc)+' ['+pct(x.ci_lo)+', '+pct(x.ci_hi)+']','mono');tr.append(td)}let u=el('td',pct(a.unparsed_rate),'mono '+(a.unparsed_rate>=D.dashboard.high_unparsed_threshold?'warn':''));tr.append(u);body.append(tr)}}}
function renderBars(){const box=document.getElementById('bars');box.replaceChildren();const parser='parse_flexible';const effects=D.condition_contrasts.filter(c=>c.suite===suite&&c.parser===parser&&c.fmt_a===c.fmt_b&&c.temp_a!==c.temp_b);const max=Math.max(.01,...effects.map(e=>Math.max(Math.abs(e.ci_lo),Math.abs(e.ci_hi))));for(const e of effects){let row=el('div',undefined,'barrow');row.append(el('div',F[e.fmt_a]));let track=el('div',undefined,'track');track.append(el('div',undefined,'zero'));let bar=el('div',undefined,'bar');const v=e.delta;bar.style.left=v>=0?'50%':(50-50*Math.abs(v)/max)+'%';bar.style.width=(50*Math.abs(v)/max)+'%';bar.style.background=v>=0?'#175c45':'#a65f0a';track.append(bar);let ci=el('div',undefined,'ci-line');ci.style.left=(50+50*e.ci_lo/max)+'%';ci.style.width=(50*(e.ci_hi-e.ci_lo)/max)+'%';track.append(ci);row.append(track);row.append(el('div',pp(v),'mono'));box.append(row)}if(!effects.length)box.append(el('p','Sample-level records are required for paired temperature intervals.','note'))}
function renderParser(){const body=document.getElementById('parserBody');body.replaceChildren();const rows=D.parser_contrasts.filter(c=>c.suite===suite);rows.sort((a,b)=>Math.abs(b.delta)-Math.abs(a.delta));for(const c of rows.slice(0,10)){let tr=el('tr');tr.append(el('td',F[c.fmt]+' · T='+c.temp+(c.seed===null?' · legacy pooled':' · seed '+c.seed)));tr.append(el('td',P[c.parser_a]+' → '+P[c.parser_b]));tr.append(el('td',pp(c.delta),'mono'));tr.append(el('td',c.q_value===null?'—':c.q_value.toExponential(2),'mono'));body.append(tr)}}
function render(){renderRange();renderTable();renderBars();renderParser();document.querySelectorAll('.tabs button').forEach(b=>b.classList.toggle('active',b.dataset.suite===suite))}
document.querySelectorAll('.tabs button').forEach(b=>b.addEventListener('click',()=>{suite=b.dataset.suite;render()}));render();
"""


def build_dashboard(
    *,
    results_path: Path,
    output_path: Path,
    title: str = "Evaluation methodology sensitivity",
    high_unparsed_threshold: float = 0.10,
) -> None:
    payload = read_json_object(results_path)
    results = validate_results(payload)
    data = results.model_dump(mode="json")
    data["dashboard"] = {"high_unparsed_threshold": high_unparsed_threshold}
    safe_json = (
        json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    meta = data["meta"]
    mode = "synthetic" if meta.get("synthetic") else "live"
    suites = meta.get("suites") or sorted({cell["suite"] for cell in data["cells"]})
    tabs = "".join(
        f'<button type="button" data-suite="{html.escape(s)}">{html.escape(s.replace("_", " ").upper())}</button>'
        for s in suites
    )
    warnings = "".join(
        f'<div class="warning">{html.escape(warning)}</div>' for warning in data["warnings"]
    )
    banner_text = (
        "Scores are generated by an explicitly synthetic backend and are not model measurements."
        if mode == "synthetic"
        else "Scores were produced by the configured model endpoint; inspect raw logs before publication."
    )
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'">
<title>{html.escape(title)}</title>
<style>{CSS}</style></head><body><div class="banner {mode}"><div class="wrap"><b>{mode} evaluation</b> · {html.escape(banner_text)}</div></div><div class="wrap"><header><p class="eyebrow">Inspect AI evaluation methodology audit</p>
<h1>{html.escape(title)}</h1><p class="lede">The model and benchmark items are held fixed while prompt format,
sampling temperature, and answer extraction vary. Parser failures are reported separately from wrong answers.</p>
<div class="meta"><div><span>Model</span><b class="mono">{html.escape(str(meta.get('model','unknown')))}</b></div>
<div><span>Items / condition</span><b class="mono">{html.escape(str(meta.get('n_per_condition','—')))}</b></div>
<div><span>Generation seeds</span><b class="mono">{html.escape(', '.join(map(str,meta.get('seeds',[]))) or '—')}</b></div>
<div><span>Run ID</span><b class="mono">{html.escape(str(meta.get('run_id','legacy')))}</b></div></div></header>
<section><h2>Methodological score range</h2><p class="note">The range below is descriptive, not a model comparison. It shows how much the published number can move under the configured evaluation choices.</p>
<div class="tabs">{tabs}</div><div class="range"><div id="rangeValue" class="range-value mono"></div><div id="rangeDesc" class="range-desc"></div><svg id="axis" class="axis" role="img" aria-label="Score distribution"></svg></div></section>
<section><h2>Condition grid</h2><p class="note">Pooled scores average across configured generation seeds. Intervals resample benchmark items; the unreadable column uses the strict output contract.</p>
<table><thead><tr><th>Condition</th><th>Strict</th><th>Flexible</th><th>Permissive</th><th>Unreadable</th></tr></thead><tbody id="body"></tbody></table></section>
<section class="grid"><div class="panel"><h2>Temperature change</h2><p class="note">Change from the lowest to highest configured temperature under the flexible parser. Error bars are paired item-bootstrap intervals. This is not labeled “reasoning loss”: extraction remains an imperfect proxy.</p><div id="bars" class="bars"></div></div>
<div class="panel"><h2>Largest parser effects</h2><p class="note">The same completions are rescored within each generation seed. q-values use Benjamini–Hochberg correction across all seed-level parser contrasts.</p><table><thead><tr><th>Condition</th><th>Parsers</th><th>Delta</th><th>q</th></tr></thead><tbody id="parserBody"></tbody></table></div></section>
<section class="grid"><div><h2>Warnings and scope</h2><div class="warnings">{warnings}</div></div><div><h2>Provenance</h2><dl class="provenance">
<dt>Created</dt><dd class="mono">{html.escape(str(meta.get('created_at','—')))}</dd><dt>Config SHA-256</dt><dd class="mono">{html.escape(str(meta.get('config_sha256','—')))}</dd>
<dt>Dataset manifest</dt><dd class="mono">{html.escape(str(meta.get('dataset_manifest_sha256','—')))}</dd><dt>Inspect AI</dt><dd class="mono">{html.escape(str(meta.get('inspect_ai_version','—')))}</dd>
<dt>Model revision</dt><dd class="mono">{html.escape(str(meta.get('model_revision') or '—'))}</dd><dt>Serving engine</dt><dd class="mono">{html.escape(' '.join(filter(None,[str(meta.get('serving_engine') or ''),str(meta.get('serving_engine_version') or '')])) or '—')}</dd>
<dt>Task version</dt><dd class="mono">{html.escape(str(meta.get('task_version') or '—'))}</dd><dt>Completion text stored</dt><dd>{html.escape(str(meta.get('completion_text_stored','unknown')))}</dd></dl></div></section>
<footer>Generated only from validated <span class="mono">results.json</span>. Review the Inspect logs, sample errors, dataset manifest, and resolved configuration before making external claims.</footer></div>
<script>const __DATA__={safe_json};</script><script>{JS.replace('__DATA__','__DATA__')}</script></body></html>'''
    atomic_write_text(output_path, document)
