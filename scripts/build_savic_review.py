"""Build a Laya dataset from SAVIC Prospect Intelligence claim review decisions.

Reads the app's SQLite databases read-only and writes data/savic-review/:
  questions.json   one choice question: approve or reject the claim
  train.jsonl      cases decided by anyone (Andrew, Claude on delegation, rules)
  test.jsonl       held-out cases decided by Andrew himself (gold only)
  report.txt       counts and coverage

With --split it writes data/savic-review-v2/ instead and leaves the files above
untouched (runs/savic-review-v1 still points at them):
  questions.json   the decision question plus 4 noul split questions
                   (supported, about_prospect, relevant, current)
  train.jsonl      decision gold as above, except rule approvals ("Rule: verified
                   source"), which answer no decision and count only as one weak
                   "yes" vote for `supported`. Split questions: teacher soft p_true
                   (teacher.jsonl), overridden by hard labels read off the rationale.
  test.jsonl       the same 43 held-out cases; decision plus rationale-derived hard
                   split labels only (no teacher, no rule labels)
  report.txt       counts and coverage
Teacher labels are model-made: a question's teacher labels are used only if
teacher.jsonl, as it is now, has a passing audit for that question
(scripts/label_audit.py). --allow-unaudited uses them anyway, with a warning.

Usage: python -I scripts/build_savic_review.py <path to savic-prospect-intelligence> [--split [--allow-unaudited]]
"""
import glob
import hashlib
import json
import os
import sqlite3
import sys

_ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
SPLIT = "--split" in sys.argv[1:]
ALLOW_UNAUDITED = "--allow-unaudited" in sys.argv[1:]
APP = os.path.abspath(_ARGS[0] if _ARGS else r"C:\Users\andre\dev\savic-prospect-intelligence")
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUT = os.path.join(DATA, "savic-review")
OUT_SPLIT = os.path.join(DATA, "savic-review-v2")
WINDOW = 700  # characters of source text kept around the quote
SPLIT_KEYS = ("supported", "about_prospect", "relevant", "current")

QUESTIONS = {
    "decision": {
        "type": "choice",
        "instructions": (
            "A reviewer is vetting a claim about a prospect before it can be used in sales material "
            "for SAVIC, an ERP and technology-modernization firm. Approve only if the excerpt supports "
            "the statement exactly as worded and the claim is relevant to the pursuit."
        ),
        "criteria": {
            "approve": "the excerpt states or directly implies every part of the claim, and it is relevant",
            "reject": "unsupported, overstated, contested, off-topic or website housekeeping",
        },
    }
}

# Wording follows scripts/teacher_label.py, so the soft labels answer the same question.
SPLIT_QUESTIONS = {
    "supported": {
        "type": "noul",
        "instructions": "Does the excerpt support the statement exactly as worded?",
        "criteria": {
            "true": "the excerpt states or directly implies every part of the statement as worded",
            "false": "some part of the statement is missing from, overstated or contradicted by the excerpt",
        },
    },
    "about_prospect": {
        "type": "noul",
        "instructions": "Is the statement about the prospect company itself?",
        "criteria": {
            "true": "it is about the prospect company itself",
            "false": "it is about another company, vendor, customer or the market in general",
        },
    },
    "relevant": {
        "type": "noul",
        "instructions": "Is the statement relevant for an ERP or IT-services pitch to the prospect?",
        "criteria": {
            "true": "it says something about the prospect's operations, systems, finances, growth, risks or change",
            "false": "product marketing, legal boilerplate, careers or biography",
        },
    },
    "current": {
        "type": "noul",
        "instructions": "Is the statement current?",
        "criteria": {
            "true": "nothing in the text suggests it is stale",
            "false": "it comes from an old filing or describes a situation many years past",
        },
    },
}


def rationale_labels(text, decision):
    """Hard split labels where the recorded rationale names the reason outright.

    Only reasons that map to one split question unambiguously; anything else
    (duplicates, aggregators, provenance, "vague") gives no label.
    """
    t = " ".join(str(text or "").lower().split())
    out = {}
    if decision == "reject":
        if "stale" in t or "not current" in t:
            out["current"] = False
        if "about other manufacturers" in t or "wrong company" in t:
            out["about_prospect"] = False
        if ("product marketing" in t or "boilerplate" in t or "careers" in t
                or "staffing or biography" in t or "not relevant to a systems or operations pitch" in t):
            out["relevant"] = False
        if "inferred from a web page, not stated" in t:
            out["supported"] = False
    elif decision == "approve":
        if "(kv4 pre-sort): current:" in t or "(kv4 pre-sort): recent," in t:
            out["current"] = True
    return out


def ro(path):
    return sqlite3.connect("file:{}?mode=ro".format(path.replace("\\", "/")), uri=True)


STOP = set("the a an and or of to in on for with by at from as is are was were be been it its this that "
           "their they has have had will can our we his her which who".split())


def words(text):
    return {w for w in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()
            if len(w) > 2 and w not in STOP}


def window(text, quote):
    text = " ".join(str(text or "").split())
    if not text:
        return ""
    q = " ".join(str(quote or "").split())
    i = text.find(q[:60]) if q else -1
    if i < 0:
        return text[:WINDOW]
    start = max(0, i - (WINDOW - len(q)) // 2)
    return text[start:start + max(WINDOW, len(q) + 200)]


def main():
    main_db = ro(os.path.join(APP, "data", "savic.sqlite3"))
    kv4_db = ro(os.path.join(APP, "output", "kv4", "kv4_repository.sqlite3"))

    statements = {cid: s for cid, s in main_db.execute("select claim_id, statement from claim_namespace")}
    evidence_main = {eid: ex for eid, ex in main_db.execute("select evidence_id, excerpt from evidence")}
    evidence_kv4 = {eid: t for eid, t in kv4_db.execute("select evidence_id, text from evidence")}

    # KV4 claims surface in review as KV4-<12 hex>; their store id is claim_<12 hex>.
    kv4 = {}
    for cid, stmt, quote, ev in kv4_db.execute("select claim_id, statement, quote, evidence_ids from claims"):
        kv4["KV4-" + cid.split("_", 1)[-1]] = (stmt, quote, json.loads(ev or "[]"))

    # Pipeline reruns clear governed_claims, but the dated backups keep each
    # run's claims with their evidence. Claim ids get reused across reruns, so
    # a backup only counts when its statement matches the reviewed one.
    governed = {}
    for f in sorted(glob.glob(os.path.join(APP, "data", "savic*.sqlite3"))):
        db = ro(f)
        try:
            ev = {eid: ex for eid, ex in db.execute("select evidence_id, excerpt from evidence")}
            for cid, stmt, ev_ids in db.execute("select claim_id, statement, evidence_ids_json from governed_claims"):
                excerpts = [ev[e] for e in json.loads(ev_ids or "[]") if ev.get(e)]
                if excerpts:
                    governed.setdefault(cid, {})[" ".join(stmt.split())] = " ... ".join(excerpts)[:2000]
        except sqlite3.Error:
            pass
        finally:
            db.close()

    # Claims registered by the main pipeline (BAU-001...) keep no evidence link.
    # Fall back to the prospect's evidence passage sharing the most content
    # words with the statement: approximate, so these cases are flagged.
    ns_prospect = dict(main_db.execute("select claim_id, prospect_id from claim_namespace"))
    pool = {}
    for f in sorted(glob.glob(os.path.join(APP, "data", "savic*.sqlite3"))):
        db = ro(f)
        try:
            for pid, ex in db.execute("select s.prospect_id, e.excerpt from evidence e join sources s using(source_id)"):
                if ex and len(ex) > 40:
                    pool.setdefault(pid, set()).add(" ".join(ex.split()))
        except sqlite3.Error:
            pass
        finally:
            db.close()
    pool = {pid: [(p, words(p)) for p in ps] for pid, ps in pool.items()}

    def best_passage(cid, stmt):
        want = words(stmt)
        if not want:
            return ""
        scored = max(((len(want & w) / len(want), p) for p, w in pool.get(ns_prospect.get(cid), [])),
                     default=(0, ""))
        return window(scored[1], "") if scored[0] >= 0.5 else ""

    prospectus = {}
    for f in glob.glob(os.path.join(APP, "output", "*.prospectus.json")):
        with open(f, encoding="utf-8") as fh:
            for c in json.load(fh).get("claims", []):
                prospectus[c["claim_id"]] = c

    # Latest decision per claim wins (reviewers reverse themselves).
    latest, rationale = {}, {}
    for rid, oid, decision, reviewer, why in main_db.execute(
        "select review_id, object_id, decision, reviewer, rationale from operational_review_decisions "
        "where object_type='claim' order by review_id"
    ):
        latest[oid] = (decision, reviewer)
        rationale[oid] = why or ""

    # Answers Andrew gave on confirm.html override the recorded decision and
    # make the case human gold, whoever decided it first.
    confirmed = {}
    conf_path = os.path.join(OUT, "confirmations.json")
    if os.path.exists(conf_path):
        with open(conf_path, encoding="utf-8") as f:
            confirmed = {k: v for k, v in json.load(f).get("claims", {}).items() if v.get("decision")}

    train, test, skipped = [], [], {"no_statement": 0, "no_excerpt": 0, "request_changes": 0}
    stats = {"linked": 0, "matched": 0}
    everything = []
    for oid, (decision, reviewer) in sorted(latest.items()):
        recorded = decision
        if oid in confirmed:
            decision, reviewer = confirmed[oid]["decision"], "Andrew Freeman (confirmed on confirm.html)"
        stmt, excerpt = None, ""
        if oid in kv4:
            stmt, quote, ev = kv4[oid]
            excerpt = window(evidence_kv4.get(ev[0], "") if ev else "", quote) or quote
        if oid in prospectus:
            c = prospectus[oid]
            stmt = stmt or c.get("statement")
            ev = c.get("evidence_ids") or []
            excerpt = excerpt or window(evidence_main.get(ev[0], "") if ev else "", c.get("supporting_quote")) \
                or c.get("supporting_quote", "")
        stmt = stmt or statements.get(oid)
        if stmt and not excerpt:
            excerpt = governed.get(oid, {}).get(" ".join(stmt.split()), "")
        matched = False
        if stmt and not excerpt:
            excerpt = best_passage(oid, stmt)
            matched = bool(excerpt)
        everything.append({
            "id": oid, "statement": stmt or "", "excerpt": excerpt,
            "excerpt_source": "matched" if matched else ("linked" if excerpt else "none"),
            "decision": decision, "reviewer": reviewer, "rationale": rationale.get(oid, ""),
            "confirmed": confirmed.get(oid),
        })
        if decision not in ("approve", "reject"):
            skipped["request_changes"] += 1
            continue
        if not stmt:
            skipped["no_statement"] += 1
            continue
        if not excerpt:
            skipped["no_excerpt"] += 1
            continue
        human = reviewer.startswith("Andrew Freeman") and "Claude" not in reviewer
        stats["matched" if matched else "linked"] += 1
        case = {"id": oid, "state": {"statement": stmt, "excerpt": excerpt},
                "gold": {"decision": decision}, "_reviewer": reviewer, "_human": human,
                "_rationale": rationale.get(oid, "") if decision == recorded else ""}
        # Hold out roughly a third of Andrew's own decisions as the test set.
        if human and int(hashlib.sha1(oid.encode()).hexdigest(), 16) % 3 == 0:
            test.append(case)
        else:
            train.append(case)

    if SPLIT:
        write_split(train, test)
        return

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "questions.json"), "w", encoding="utf-8") as f:
        json.dump(QUESTIONS, f, indent=2)
    with open(os.path.join(OUT, "all_cases.json"), "w", encoding="utf-8") as f:
        json.dump(everything, f, ensure_ascii=False, indent=1)
    for name, rows in (("train", train), ("test", test)):
        with open(os.path.join(OUT, name + ".jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}, ensure_ascii=False) + "\n")

    def counts(rows):
        out = {}
        for r in rows:
            out[r["gold"]["decision"]] = out.get(r["gold"]["decision"], 0) + 1
        return out

    by_reviewer = {}
    for r in train + test:
        by_reviewer[r["_reviewer"]] = by_reviewer.get(r["_reviewer"], 0) + 1
    report = [
        "decided claims: {}".format(len(latest)),
        "train: {} {}".format(len(train), counts(train)),
        "test (Andrew only): {} {}".format(len(test), counts(test)),
        "skipped: {}".format(skipped),
        "excerpt source: {} linked to the claim's evidence, {} matched by word overlap".format(
            stats["linked"], stats["matched"]),
        "by reviewer: {}".format(json.dumps(by_reviewer, indent=1)),
    ]
    with open(os.path.join(OUT, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print("\n".join(report))


def load_teacher():
    path = os.path.join(OUT, "teacher.jsonl")
    out = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    out[row["id"]] = row
    return out


def write_split(train, test):
    """data/savic-review-v2/: decision + 4 split questions (see the module docstring)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from label_audit import require_audit
    teacher_path = os.path.join(OUT, "teacher.jsonl")
    audited = {k: os.path.exists(teacher_path) and require_audit("teacher", teacher_path, k, ALLOW_UNAUDITED)
               for k in SPLIT_KEYS}
    teacher = load_teacher()
    rows = {"train": [], "test": []}
    tally = {"train": {}, "test": {}}
    notes = {"rule_rows": 0, "no_teacher": 0, "rationale_overrides": 0}

    def bump(split, key, value):
        d = tally[split].setdefault(key, {})
        d[value] = d.get(value, 0) + 1

    for split, cases in (("train", train), ("test", test)):
        for c in cases:
            rule = c["_reviewer"].startswith("Rule:")
            hard = rationale_labels(c["_rationale"], c["gold"]["decision"])
            gold = {"decision": None if rule else c["gold"]["decision"]}
            if split == "train":
                t = teacher.get(c["id"])
                if t is None:
                    notes["no_teacher"] += 1
                p_true, n = (t or {}).get("p_true", {}), (t or {}).get("samples", 0)
                for k in SPLIT_KEYS:
                    p = p_true.get(k) if audited[k] else None
                    if k == "supported" and rule:
                        # A verified-source rule approval: one weak "yes" vote beside the teacher's.
                        p = (n * p + 1.0) / (n + 1) if p is not None else 0.75
                    gold[k] = None if p is None else round(float(p), 3)
                if rule:
                    notes["rule_rows"] += 1
            else:
                for k in SPLIT_KEYS:
                    gold[k] = None
            for k, v in hard.items():
                if split == "train" and gold.get(k) is not None:
                    notes["rationale_overrides"] += 1
                gold[k] = v
            for k, v in gold.items():
                if v is None:
                    continue
                label = v if isinstance(v, str) else ("hard " + str(v).lower() if isinstance(v, bool)
                                                       else "soft p>=0.5" if v >= 0.5 else "soft p<0.5")
                bump(split, k, label)
            rows[split].append({"id": c["id"], "workflow": "claim-review", "state": c["state"], "gold": gold})

    os.makedirs(OUT_SPLIT, exist_ok=True)
    questions = dict(QUESTIONS)
    questions.update(SPLIT_QUESTIONS)
    with open(os.path.join(OUT_SPLIT, "questions.json"), "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2)
    for name, rs in rows.items():
        with open(os.path.join(OUT_SPLIT, name + ".jsonl"), "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = [
        "split-question dataset (build_savic_review.py --split)",
        "train cases: {}  test cases: {}".format(len(rows["train"]), len(rows["test"])),
        "teacher labels used (passing audit{}): {}".format(
            " or --allow-unaudited" if ALLOW_UNAUDITED else "", json.dumps(audited, sort_keys=True)),
        "rule approvals in train (no decision label, weak 'supported' vote): {}".format(notes["rule_rows"]),
        "train cases without teacher labels: {}".format(notes["no_teacher"]),
        "teacher labels replaced by a rationale-derived hard label: {}".format(notes["rationale_overrides"]),
        "train labels per question: {}".format(json.dumps(tally["train"], indent=1, sort_keys=True)),
        "test labels per question (human decisions and rationales only): {}".format(
            json.dumps(tally["test"], indent=1, sort_keys=True)),
    ]
    with open(os.path.join(OUT_SPLIT, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
