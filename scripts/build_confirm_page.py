"""Write data/savic-review/confirm.html: a local page for Andrew to confirm or
correct claim decisions and Claude's scoring-question answers.

Answers autosave in the browser and are exported with the "Export answers"
button as confirmations.json; put that file in data/savic-review/ and rerun
build_savic_review.py to fold the answers into the training set.

With --jev-unsure N it writes jev_review.html instead: only the N claims Jev was
least sure about (most answers in build_savic_review.JEV_UNSURE, then closest to
0.5), skipping claims Andrew already answered. Jev's numbers are not shown, so
the answers stay blind. It shares confirm.html's saved answers and export.

Usage: python -I scripts/build_confirm_page.py <path to savic-prospect-intelligence> [--jev-unsure N]
"""
import json
import os
import sys

_ARGS = sys.argv[1:]
JEV_N = int(_ARGS[_ARGS.index("--jev-unsure") + 1]) if "--jev-unsure" in _ARGS else 0
_POS = [a for i, a in enumerate(_ARGS) if not a.startswith("--") and (i == 0 or _ARGS[i - 1] != "--jev-unsure")]
APP = os.path.abspath(_POS[0] if _POS else r"C:\Users\andre\dev\savic-prospect-intelligence")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "savic-review")
PAGE_NAME = "jev_review.html" if JEV_N else "confirm.html"

with open(os.path.join(OUT, "all_cases.json"), encoding="utf-8") as f:
    claims = json.load(f)
questions = []
if JEV_N:
    lo, hi = 0.3, 0.7  # keep in step with build_savic_review.JEV_UNSURE
    with open(os.path.join(OUT, "teacher.jev.jsonl"), encoding="utf-8") as f:
        jev = {r["id"]: r["p_true"] for r in map(json.loads, filter(str.strip, f))}
    answered = set()
    conf_path = os.path.join(OUT, "confirmations.json")
    if os.path.exists(conf_path):
        with open(conf_path, encoding="utf-8") as f:
            answered = {i for i, a in json.load(f).get("claims", {}).items()
                        if a.get("decision") and not str(a.get("note", "")).startswith("[claude-draft]")}

    def unsure(c):
        ps = list(jev.get(c["id"], {}).values())
        return (sum(lo < p < hi for p in ps), -min((abs(p - 0.5) for p in ps), default=1))

    claims = sorted((c for c in claims if c["id"] in jev and c["id"] not in answered and unsure(c)[0]),
                    key=unsure, reverse=True)[:JEV_N]
else:
    with open(os.path.join(APP, "Claude outputs", "scoring-questions", "questions.jsonl"), encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

data = json.dumps({"claims": claims, "questions": questions}, ensure_ascii=False).replace("</", "<\\/")

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Claim Confirmations</title>
<style>
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e2e1dc;--accent:#2f5bd3;--ok:#1f7a4d;--bad:#b3261e;--warn:#9a6700;--chip:#efeee9}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#20201e;--ink:#ecebe6;--muted:#a3a29b;--line:#34332f;--accent:#7b9cff;--ok:#5cc28f;--bad:#ff8a80;--warn:#e0b252;--chip:#2b2a27}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,Segoe UI,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 16px;z-index:2}
.wrap{max-width:860px;margin:0 auto}
h1{font-size:18px;margin:0 0 8px}.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:8px;padding:6px 12px;cursor:pointer}
button.on{background:var(--accent);border-color:var(--accent);color:#fff}
button.ok.on{background:var(--ok);border-color:var(--ok)}button.bad.on{background:var(--bad);border-color:var(--bad)}
select{font:inherit;padding:5px 8px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--ink)}
.bar{height:6px;background:var(--chip);border-radius:3px;flex:1;min-width:120px;overflow:hidden}.bar i{display:block;height:100%;background:var(--ok)}
main{padding:16px}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:14px}
.card.done{border-color:var(--ok)}.meta{color:var(--muted);font-size:13px}
.chip{display:inline-block;background:var(--chip);border-radius:999px;padding:1px 9px;font-size:12px;margin-right:4px}
.stmt{font-weight:600;margin:8px 0}.ex{background:var(--chip);border-radius:8px;padding:10px;font-size:14px;white-space:pre-wrap;max-height:220px;overflow:auto}
.q{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:6px 12px;align-items:center;margin-top:10px}
.q span{font-size:14px}textarea{width:100%;font:inherit;font-size:14px;border:1px solid var(--line);border-radius:8px;padding:6px;background:var(--card);color:var(--ink);margin-top:8px}
.warn{color:var(--warn)}.hidden{display:none}
@media (max-width:560px){.q{grid-template-columns:1fr}}
</style></head><body>
<header><div class="wrap">
<h1>Confirm claim decisions</h1>
<div class="row">
<button id="tabC" class="on">Claims</button><button id="tabQ">Scoring questions</button>
<select id="filter"><option value="todo">Not yet answered</option><option value="all">All</option><option value="done">Answered</option></select>
<select id="who"><option value="">Any reviewer</option></select>
<div class="bar"><i id="prog"></i></div><span id="count" class="meta"></span>
<button id="export">Export answers</button>
</div></div></header>
<main class="wrap" id="list"></main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
const KEY='savic-confirm-v1';let S={claims:{},questions:{}};
try{S=Object.assign(S,JSON.parse(localStorage.getItem(KEY)||'{}'))}catch(e){}
function save(){try{localStorage.setItem(KEY,JSON.stringify(S))}catch(e){}render(false)}
let tab='claims';
const SPLIT=[["supported","Excerpt supports the wording?",["yes","partly","no"]],
 ["about_prospect","About this prospect (not another company)?",["yes","no"]],
 ["relevant","Relevant to an ERP / operations pitch?",["yes","no"]],
 ["current","Current enough to use?",["yes","no"]]];
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const whoSel=document.getElementById('who');
[...new Set(D.claims.map(c=>c.reviewer.replace(/\s*\(.*$/,'').trim()))].forEach(r=>whoSel.add(new Option(r,r)));
function btns(group,field,opts,cur){return opts.map(o=>`<button data-g="${group}" data-f="${field}" data-v="${o}" class="${o=='approve'||o=='yes'?'ok':o=='reject'||o=='no'?'bad':''} ${cur==o?'on':''}">${o}</button>`).join(' ')}
function claimCard(c){const a=S.claims[c.id]||{};const done=!!a.decision;
 return `<div class="card ${done?'done':''}" id="c-${esc(c.id)}"><div class="meta"><span class="chip">${esc(c.id)}</span>
 recorded: <b>${esc(c.decision)}</b> by ${esc(c.reviewer)}${c.rationale?` — “${esc(c.rationale)}”`:''}</div>
 <div class="stmt">${esc(c.statement)}</div>
 ${c.excerpt?`<div class="ex">${esc(c.excerpt)}</div>${c.excerpt_source=='matched'?'<div class="meta warn">Excerpt matched by word overlap — may not be the exact passage you reviewed.</div>':''}`:'<div class="meta warn">No excerpt on file.</div>'}
 <div class="q">${SPLIT.map(([f,l,o])=>`<span>${l}</span><div>${btns('c:'+c.id,f,o,a[f])}</div>`).join('')}
 <span><b>Your decision</b></span><div>${btns('c:'+c.id,'decision',['approve','reject'],a.decision)} <button data-agree="${esc(c.id)}">Agree with recorded</button></div></div>
 <textarea data-g="c:${esc(c.id)}" data-f="note" rows="1" placeholder="Note (optional)">${esc(a.note||'')}</textarea></div>`}
function qCard(q){const a=S.questions[q.id]||{};const done=!!a.answer;const cx=q.context||{};
 return `<div class="card ${done?'done':''}"><div class="meta"><span class="chip">${esc(q.id)}</span><span class="chip">${esc(q.type)}</span>${esc(q.prospect)}</div>
 <div class="stmt">${esc(q.question)}</div>
 ${cx.statement?`<div><b>Claim:</b> ${esc(cx.statement)}</div>`:''}
 ${cx.quote?`<div class="ex">${esc(cx.quote)}</div>`:''}
 <div class="meta">${esc(cx.source||'')} ${cx.source_date?'· '+esc(cx.source_date):''} ${cx.source_url?`· <a href="${esc(cx.source_url)}" target="_blank" rel="noopener">source</a>`:''}</div>
 ${Object.keys(cx).filter(k=>!['statement','quote','source','source_url','source_date','claim_id'].includes(k)).map(k=>`<div class="meta"><b>${esc(k)}:</b> ${esc(typeof cx[k]=='object'?JSON.stringify(cx[k]):cx[k])}</div>`).join('')}
 <div class="meta" style="margin-top:8px">Claude answered <b>${esc(q.answer)}</b> (${esc(q.confidence)}): ${esc(q.rationale)}</div>
 <div class="q"><span><b>Your answer</b></span><div>${btns('q:'+q.id,'answer',q.options||['yes','no'],a.answer)} <button data-qagree="${esc(q.id)}">Agree with Claude</button></div></div>
 <textarea data-g="q:${esc(q.id)}" data-f="note" rows="1" placeholder="Note (optional)">${esc(a.note||'')}</textarea></div>`}
function render(full=true){
 const f=document.getElementById('filter').value,who=whoSel.value;
 const items=tab=='claims'?D.claims:D.questions,store=tab=='claims'?S.claims:S.questions,k=tab=='claims'?'decision':'answer';
 const n=items.filter(i=>store[i.id]&&store[i.id][k]).length;
 document.getElementById('count').textContent=`${n} / ${items.length} answered`;
 document.getElementById('prog').style.width=(100*n/items.length)+'%';
 if(!full)return;
 const shown=items.filter(i=>{const d=store[i.id]&&store[i.id][k];if(f=='todo'&&d)return false;if(f=='done'&&!d)return false;
  if(tab=='claims'&&who&&!i.reviewer.startsWith(who))return false;return true}).slice(0,40);
 document.getElementById('list').innerHTML=shown.map(tab=='claims'?claimCard:qCard).join('')||'<p class="meta">Nothing left in this view.</p>';
}
document.addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;
 if(b.dataset.g){const[t,id]=b.dataset.g.split(/:(.*)/s);const st=t=='c'?S.claims:S.questions;(st[id]=st[id]||{})[b.dataset.f]=b.dataset.v;
  b.parentNode.querySelectorAll('button[data-f]').forEach(x=>x.classList.toggle('on',x==b));
  if(b.dataset.f=='decision'||b.dataset.f=='answer')b.closest('.card').classList.add('done');save()}
 if(b.dataset.agree){const c=D.claims.find(x=>x.id==b.dataset.agree);(S.claims[c.id]=S.claims[c.id]||{}).decision=c.decision=='approve'?'approve':'reject';save();render()}
 if(b.dataset.qagree){const q=D.questions.find(x=>x.id==b.dataset.qagree);(S.questions[q.id]=S.questions[q.id]||{}).answer=q.answer;save();render()}});
document.addEventListener('change',e=>{const t=e.target;if(t.tagName=='TEXTAREA'){const[g,id]=t.dataset.g.split(/:(.*)/s);const st=g=='c'?S.claims:S.questions;(st[id]=st[id]||{}).note=t.value;save()}});
document.getElementById('filter').onchange=whoSel.onchange=()=>render();
document.getElementById('tabC').onclick=()=>{tab='claims';tabC.classList.add('on');tabQ.classList.remove('on');whoSel.classList.remove('hidden');render()};
document.getElementById('tabQ').onclick=()=>{tab='questions';tabQ.classList.add('on');tabC.classList.remove('on');whoSel.classList.add('hidden');render()};
document.getElementById('export').onclick=()=>{const out=Object.assign({exported_at:new Date().toISOString()},S);
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{type:'application/json'}));a.download='confirmations.json';a.click()};
render();
</script></body></html>"""

if JEV_N:
    PAGE = PAGE.replace("<title>Claim Confirmations</title>", "<title>Jev Review List</title>")
with open(os.path.join(OUT, PAGE_NAME), "w", encoding="utf-8") as f:
    f.write(PAGE.replace("__DATA__", data))
print("wrote", os.path.join(OUT, PAGE_NAME), len(claims), "claims,", len(questions), "questions")
