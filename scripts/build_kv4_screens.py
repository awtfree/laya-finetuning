"""Build a Laya training set from the KV4 screen decisions in SAVIC Prospect Intelligence.

The KV4 engine screens every evidence block with a local LLM ("could this say
something about the company's operations, systems, ...?") and stores keep/drop
in output/kv4/kv4_repository.sqlite3, table `screens`. Those labels are
model-made, so this set is for TRAINING ONLY: it writes no test split.

Reads the database read-only and writes data/kv4-screens/:
  questions.json   one noul question: is the passage worth reading for an ERP pitch?
  train.jsonl      one case per screened evidence block; state = the block's text,
                   cut to fit the 1,024-token sequence. A block screened by more than
                   one screen version gets the share of "keep" votes as a soft label.
  report.txt       counts, screener agreement, and blocks left out

  labels.jsonl     every screen label (all blocks), the file label_audit.py audits

Blocks whose text contains an excerpt from the held-out claim-review test set
(data/savic-review-v2/test.jsonl, if present) are left out, so the test
excerpts are never seen in training. Without a passing audit of labels.jsonl
(scripts/label_audit.py, source "kv4-screens") no screen cases are written,
unless --allow-unaudited.

Usage: python -I scripts/build_kv4_screens.py <path to savic-prospect-intelligence> [--allow-unaudited]
"""
import json
import os
import sqlite3
import sys

_ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
ALLOW_UNAUDITED = "--allow-unaudited" in sys.argv[1:]
APP = os.path.abspath(_ARGS[0] if _ARGS else r"C:\Users\andre\dev\savic-prospect-intelligence")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_audit import require_audit  # noqa: E402
OUT = os.path.join(ROOT, "data", "kv4-screens")
TEST = os.path.join(ROOT, "data", "savic-review-v2", "test.jsonl")
TOKENIZER = os.path.join(ROOT, "models", "convaiinnovations__laya", "tokenizer", "tokenizer.json")
# train.py: max_len 1024 per sequence, of which head_max_len 256 is reserved for the
# instructions and options. Leave a margin for the state's JSON wrapping.
STATE_TOKENS = 1024 - 256 - 48

QUESTIONS = {
    "worth_reading": {
        "type": "noul",
        "instructions": "Is this passage worth reading for an ERP pitch?",
        "criteria": {
            "true": "it could say something about the company's operations, systems, costs, delays, risks, "
                    "growth, customers, suppliers, leadership or plans",
            "false": "navigation, cookie or legal boilerplate, contract text, price lists, store locators "
                     "or lists of links",
        },
    }
}


def ro(path):
    return sqlite3.connect("file:{}?mode=ro".format(path.replace("\\", "/")), uri=True)


def norm(text):
    return " ".join(str(text or "").split())


def main():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOKENIZER)

    def fit(text):
        enc = tok.encode(text, add_special_tokens=False)
        if len(enc.ids) <= STATE_TOKENS:
            return text, False
        return text[:enc.offsets[STATE_TOKENS - 1][1]], True

    test_excerpts = []
    if os.path.exists(TEST):
        with open(TEST, encoding="utf-8") as f:
            test_excerpts = [norm(json.loads(line)["state"]["excerpt"]) for line in f if line.strip()]
    probes = [e[:100] for e in test_excerpts if len(e) >= 60]

    db = ro(os.path.join(APP, "output", "kv4", "kv4_repository.sqlite3"))
    votes, texts = {}, {}
    rows = db.execute(
        "select s.prospect_id, s.evidence_id, s.keep, s.produced_by, e.text "
        "from screens s join evidence e on e.prospect_id = s.prospect_id and e.evidence_id = s.evidence_id "
        "order by s.prospect_id, s.evidence_id, s.produced_by")
    n_rows = 0
    for pid, eid, keep, producer, text in rows:
        n_rows += 1
        votes.setdefault((pid, eid), {})[producer] = int(keep)
        texts[(pid, eid)] = text
    db.close()

    # The model-made labels as one file, so they can be audited and the audit tied to its hash.
    os.makedirs(OUT, exist_ok=True)
    labels_path = os.path.join(OUT, "labels.jsonl")
    with open(labels_path, "w", encoding="utf-8") as f:
        for key in sorted(votes):
            v = votes[key]
            f.write(json.dumps({"id": "kv4screen-" + key[1].split("_", 1)[-1], "question": "worth_reading",
                                "p_true": round(sum(v.values()) / len(v), 3), "votes": v,
                                "excerpt": fit(norm(texts[key]))[0]}, ensure_ascii=False) + "\n")
    audited = require_audit("kv4-screens", labels_path, "worth_reading", ALLOW_UNAUDITED)

    cases, left_out, truncated = [], {"test_excerpt": 0, "empty": 0, "no_passing_audit": 0}, 0
    agree = disagree = 0
    for key in sorted(votes):
        text = norm(texts[key])
        if not audited:
            left_out["no_passing_audit"] += 1
            continue
        if not text:
            left_out["empty"] += 1
            continue
        if any(p in text or text[:100] in e for p, e in zip(probes, test_excerpts)):
            left_out["test_excerpt"] += 1
            continue
        v = votes[key]
        if len(v) > 1:
            agree += len(set(v.values())) == 1
            disagree += len(set(v.values())) > 1
        p = sum(v.values()) / len(v)
        text, cut = fit(text)
        truncated += cut
        cases.append({"id": "kv4screen-" + key[1].split("_", 1)[-1], "workflow": "kv4-screen",
                      "state": {"excerpt": text}, "gold": {"worth_reading": round(p, 3)}})

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "questions.json"), "w", encoding="utf-8") as f:
        json.dump(QUESTIONS, f, indent=2)
    with open(os.path.join(OUT, "train.jsonl"), "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    dist = {}
    for c in cases:
        p = c["gold"]["worth_reading"]
        k = "keep" if p == 1 else "drop" if p == 0 else "split vote"
        dist[k] = dist.get(k, 0) + 1
    report = [
        "KV4 screens (model-made labels, training only)",
        "screen rows joined to evidence: {}  distinct blocks: {}".format(n_rows, len(votes)),
        "blocks screened by two versions: {} agree, {} disagree (soft label 0.5)".format(agree, disagree),
        "left out: {}".format(left_out),
        "cases written: {} {}".format(len(cases), dist),
        "texts cut to {} tokens: {}".format(STATE_TOKENS, truncated),
    ]
    with open(os.path.join(OUT, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print("\n".join(report))
    merge(cases)


def merge(cases):
    """data/savic-review-v2b/: the split-question train set plus these screens, one schema.
    Test with data/savic-review-v2/test.jsonl (no screens in it)."""
    v2 = os.path.dirname(TEST)
    if not os.path.exists(os.path.join(v2, "train.jsonl")):
        print("no %s/train.jsonl: run build_savic_review.py --split first to get the merged set" % v2)
        return
    out = os.path.join(ROOT, "data", "savic-review-v2b")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(v2, "questions.json"), encoding="utf-8") as f:
        questions = json.load(f)
    questions.update(QUESTIONS)
    with open(os.path.join(out, "questions.json"), "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2)
    with open(os.path.join(v2, "train.jsonl"), encoding="utf-8") as f:
        review = [line.rstrip("\n") for line in f if line.strip()]
    with open(os.path.join(out, "train.jsonl"), "w", encoding="utf-8") as f:
        for line in review:
            f.write(line + "\n")
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print("merged: %s/train.jsonl = %d claim-review + %d kv4-screen cases" % (out, len(review), len(cases)))


if __name__ == "__main__":
    main()
