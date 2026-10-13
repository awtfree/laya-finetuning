"""Audit model-made labels before they are trained on.

A local model (Ollama) labels data cheaply; this script checks it. Claude, as the
judge, labels a random stratified sample without seeing the model's answer.
The script scores agreement per question and records the result against the
label file's sha256. Build scripts call `require_audit` and leave out any
model-made labels whose file has no passing audit. Any change to a label file
(new prospects, a relabel) changes its hash, so the next build forces a new audit.

    python -I scripts/label_audit.py sample --labels data/savic-review/teacher.jsonl --source teacher --n 30 --seed 1
    # judge data/audits/<source>-<sha8>-blind.jsonl, write <source>-<sha8>-answers.jsonl
    python -I scripts/label_audit.py score --source teacher

Files in data/audits/:
  <source>-<sha8>-sample.jsonl   the sample, with the model label and recorded decision (the key)
  <source>-<sha8>-blind.jsonl    the same rows without them: what the judge reads
  <source>-<sha8>-answers.jsonl  the judge's labels: {"key", "label": true|false, "reason": "..."}
  registry.json                  one entry per (source, file hash, question) audit

Label files it reads:
  * teacher.jsonl rows {"id", "p_true": {question: p}}; statement and excerpt come
    from all_cases.json beside it, and so does the recorded decision.
  * flat rows {"id", "question", "p_true" | "label", "statement"?, "excerpt"?,
    "decision"?}, as JSON Lines or a JSON list (e.g. data/kv4-screens/labels.jsonl).
"""
import argparse
import glob
import hashlib
import json
import math
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDITS = os.path.join(ROOT, "data", "audits")
REGISTRY = os.path.join(AUDITS, "registry.json")
MIN_AGREEMENT = 0.85
MIN_WILSON_LB = 0.70

# What each question asks, for the judge. Teacher wording: scripts/teacher_label.py.
QUESTION_TEXT = {
    "supported": "The excerpt states or directly implies every part of the statement as worded.",
    "about_prospect": "The statement is about the prospect company itself, not another company, vendor or customer.",
    "relevant": "It says something about the prospect's operations, systems, finances, growth, risks or change "
                "that an ERP or modernization pitch could use (not product marketing, legal boilerplate, "
                "careers or biography).",
    "current": "True if still true today, or a dated milestone in the company's ownership, acquisition, divestiture or legal-entity history (these shape the ERP landscape at any age). False if a past or superseded one-off: a launch, campaign, sponsorship, job posting, plan, forecast, expected synergy, deal rumour, or risk boilerplate from an old filing. Age alone never makes it false.",
    "decision": "Approve the claim: the excerpt supports the statement exactly as worded and it is relevant to "
                "an ERP or modernization pitch (true = approve, false = reject).",
    "worth_reading":"The passage could say something about the company's operations, systems, costs, delays, "
                     "risks, growth, customers, suppliers, leadership or plans (false only for navigation, "
                     "cookie or legal boilerplate, contract text, price lists, store locators or link lists).",
}


# --------------------------------------------------------------------------- helpers
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wilson_lb(k, n, z=1.96):
    if n == 0:
        return 0.0
    p = k / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / (1 + z * z / n)


def as_bool(v):
    """Model label -> bool. Soft labels are thresholded at 0.5."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v) >= 0.5
    if isinstance(v, str) and v.lower() in ("true", "yes", "keep", "1", "approve"):
        return True
    if isinstance(v, str) and v.lower() in ("false", "no", "drop", "0", "reject"):
        return False
    return None


def read_rows(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_registry():
    if not os.path.exists(REGISTRY):
        return []
    with open(REGISTRY, encoding="utf-8") as f:
        return json.load(f)


def explode(path):
    """A label file -> one row per (case, question) with the context a judge needs."""
    rows = read_rows(path)
    out = []
    context = {}
    if rows and isinstance(rows[0].get("p_true"), dict):
        cases_path = os.path.join(os.path.dirname(os.path.abspath(path)), "all_cases.json")
        if os.path.exists(cases_path):
            with open(cases_path, encoding="utf-8") as f:
                context = {c["id"]: c for c in json.load(f)}
        for r in rows:
            c = context.get(r["id"], {})
            for q, p in r["p_true"].items():
                out.append({"id": r["id"], "question": q, "model_p_true": p,
                            "statement": c.get("statement", ""), "excerpt": c.get("excerpt", ""),
                            "recorded_decision": c.get("decision"), "recorded_reviewer": c.get("reviewer")})
        return out
    for r in rows:
        p = r.get("p_true", r.get("label", r.get("answer")))
        if r.get("id") is None or r.get("question") is None or p is None:
            raise SystemExit("%s: rows need id, question and p_true/label (got keys %s)" % (path, sorted(r)))
        out.append({"id": r["id"], "question": r["question"], "model_p_true": p,
                    "statement": r.get("statement", ""), "excerpt": r.get("excerpt", r.get("text", "")),
                    "recorded_decision": r.get("decision"), "recorded_reviewer": r.get("reviewer")})
    return out


def agrees_with_decision(row):
    """Does the model's yes/no line up with the recorded approve/reject? None if no decision."""
    d = row.get("recorded_decision")
    m = as_bool(row["model_p_true"])
    if d not in ("approve", "reject") or m is None:
        return None
    return m == (d == "approve")


# --------------------------------------------------------------------------- sample
def cmd_sample(a):
    path = os.path.abspath(a.labels)
    digest = sha256(path)
    rows = explode(path)
    if a.question:
        rows = [r for r in rows if r["question"] in a.question]
    rng = random.Random(a.seed)
    picked = []
    for q in sorted({r["question"] for r in rows}):
        qrows = [r for r in rows if r["question"] == q and as_bool(r["model_p_true"]) is not None]
        # Strata: model label x agreement with the recorded decision, so the sample holds
        # agreements as well as disagreements, and both label values.
        strata = {}
        for r in qrows:
            strata.setdefault((as_bool(r["model_p_true"]), agrees_with_decision(r)), []).append(r)
        for s in strata.values():
            rng.shuffle(s)
        keys = sorted(strata, key=str)
        take = {k: 0 for k in keys}
        left = min(a.n, len(qrows))
        while left:  # round-robin: as even as the strata allow
            moved = False
            for k in keys:
                if left and take[k] < len(strata[k]):
                    take[k] += 1
                    left -= 1
                    moved = True
            if not moved:
                break
        for k in keys:
            picked += strata[k][:take[k]]
    rng.shuffle(picked)
    os.makedirs(AUDITS, exist_ok=True)
    stem = os.path.join(AUDITS, "%s-%s" % (a.source, digest[:8]))
    with open(stem + "-sample.jsonl", "w", encoding="utf-8") as fs, \
            open(stem + "-blind.jsonl", "w", encoding="utf-8") as fb:
        for i, r in enumerate(picked, 1):
            key = "%s#%03d" % (a.source, i)
            full = {"key": key, "id": r["id"], "question": r["question"],
                    "question_text": QUESTION_TEXT.get(r["question"], r["question"]),
                    "statement": r["statement"], "excerpt": r["excerpt"],
                    "model_p_true": r["model_p_true"], "model_label": as_bool(r["model_p_true"]),
                    "recorded_decision": r["recorded_decision"], "recorded_reviewer": r["recorded_reviewer"],
                    "agrees_with_decision": agrees_with_decision(r),
                    "labels_file": path, "labels_sha256": digest, "seed": a.seed}
            fs.write(json.dumps(full, ensure_ascii=False) + "\n")
            fb.write(json.dumps({k: full[k] for k in ("key", "question", "question_text", "statement", "excerpt")},
                                ensure_ascii=False) + "\n")
    counts = {}
    for r in picked:
        counts[r["question"]] = counts.get(r["question"], 0) + 1
    print("sampled %d rows %s from %s (sha256 %s)" % (len(picked), counts, path, digest[:8]))
    print("judge: %s-blind.jsonl  ->  write %s-answers.jsonl" % (stem, stem))
    return 0


# --------------------------------------------------------------------------- score
def cmd_score(a):
    if a.sample:
        sample_path = a.sample
    else:
        found = sorted(glob.glob(os.path.join(AUDITS, "%s-*-sample.jsonl" % a.source)), key=os.path.getmtime)
        if not found:
            raise SystemExit("no sample for source %r in %s" % (a.source, AUDITS))
        sample_path = found[-1]
    answers_path = sample_path.replace("-sample.jsonl", "-answers.jsonl")
    sample = {r["key"]: r for r in read_rows(sample_path)}
    answers = {r["key"]: r for r in read_rows(answers_path)}
    missing = sorted(set(sample) - set(answers))
    if missing:
        raise SystemExit("%d sample rows have no answer, e.g. %s" % (len(missing), missing[:5]))
    labels_file = next(iter(sample.values()))["labels_file"]
    digest = next(iter(sample.values()))["labels_sha256"]
    if os.path.exists(labels_file) and sha256(labels_file) != digest:
        print("WARN %s changed since it was sampled; this audit covers the sampled version only" % labels_file,
              file=sys.stderr)
    per_q = {}
    for key, s in sample.items():
        judge = as_bool(answers[key]["label"])
        if judge is None:
            raise SystemExit("answer %s: label must be true/false" % key)
        ok = judge == s["model_label"]
        per_q.setdefault(s["question"], []).append((ok, s, answers[key]))
    registry = load_registry()
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    total_ok = total_n = 0
    print("source %s, labels %s (sha256 %s)" % (a.source, labels_file, digest[:8]))
    for q in sorted(per_q):
        res = per_q[q]
        k, n = sum(ok for ok, _, _ in res), len(res)
        total_ok, total_n = total_ok + k, total_n + n
        agreement, lb = k / n, wilson_lb(k, n)
        passed = agreement >= a.min_agreement and lb >= a.min_wilson_lb
        registry.append({"source": a.source, "labels_file": labels_file, "labels_sha256": digest, "question": q,
                         "n": n, "agreement": round(agreement, 4), "wilson_lb": round(lb, 4), "pass": passed,
                         "thresholds": {"agreement": a.min_agreement, "wilson_lb": a.min_wilson_lb},
                         "sample": os.path.relpath(sample_path, ROOT), "auditor": "claude", "audited_at": now})
        print("  %-16s n=%3d  agreement %.3f  wilson95 lb %.3f  %s"
              % (q, n, agreement, lb, "PASS" if passed else "FAIL"))
        for ok, s, ans in res:
            if not ok and a.show_errors:
                print("      x %s %s model=%s judge=%s: %s"
                      % (s["key"], s["id"], s["model_p_true"], ans["label"], ans.get("reason", "")))
    print("  overall          n=%3d  agreement %.3f  wilson95 lb %.3f"
          % (total_n, total_ok / total_n, wilson_lb(total_ok, total_n)))
    with open(REGISTRY, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=1)
    print("appended %d entries to %s" % (len(per_q), REGISTRY))
    return 0


# --------------------------------------------------------------------------- gate
def require_audit(source, path, question, allow_unaudited=False):
    """True if `path` (as it is now) has a passing audit for `question` under `source`.

    The latest audit of this exact file content decides. Otherwise warn on stderr and
    return False, so the caller leaves those labels out - unless allow_unaudited.
    """
    digest = sha256(path)
    hits = [e for e in load_registry()
            if e["source"] == source and e["labels_sha256"] == digest and e["question"] == question]
    if hits and hits[-1]["pass"]:
        return True
    why = ("its last audit failed (agreement %.3f, wilson lb %.3f, n=%d)"
           % (hits[-1]["agreement"], hits[-1]["wilson_lb"], hits[-1]["n"])) if hits \
        else "no audit of this file version (sha256 %s)" % digest[:8]
    if allow_unaudited:
        print("WARN using unaudited %s labels for %r from %s: %s (--allow-unaudited)"
              % (source, question, path, why), file=sys.stderr)
        return True
    print("WARN leaving out %s labels for %r from %s: %s. Run scripts/label_audit.py sample/score, "
          "or pass --allow-unaudited." % (source, question, path, why), file=sys.stderr)
    return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="draw a random stratified sample for the judge")
    s.add_argument("--labels", required=True, help="model-made label file")
    s.add_argument("--source", required=True, help="name for this label source, e.g. teacher, kv4-screens")
    s.add_argument("--n", type=int, default=30, help="rows per question")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--question", nargs="*", help="only these questions")
    c = sub.add_parser("score", help="score the judge's answers and record the audit")
    c.add_argument("--source", required=True)
    c.add_argument("--sample", help="sample file (default: the newest for this source)")
    c.add_argument("--min-agreement", type=float, default=MIN_AGREEMENT)
    c.add_argument("--min-wilson-lb", type=float, default=MIN_WILSON_LB)
    c.add_argument("--show-errors", action="store_true", help="list every disagreement")
    a = ap.parse_args(argv)
    return cmd_sample(a) if a.cmd == "sample" else cmd_score(a)


if __name__ == "__main__":
    sys.exit(main())
