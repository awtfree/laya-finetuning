#!/usr/bin/env python3
"""Tests for the [claude-draft] handling of confirmations.json in build_savic_review.py.

    python -I scripts/test_claude_drafts.py

Plain Python, no container. Everything is written to a temporary directory: the
build's output folders and the audit registry are redirected there. The second
part runs the real build against the app databases (read-only) when they exist.
"""
import hashlib
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.argv = [sys.argv[0]]  # the build reads its flags at import
import build_savic_review as b  # noqa: E402
import label_audit as la  # noqa: E402

FAILS = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        FAILS.append(what)


def redirect(tmp):
    b.OUT = os.path.join(tmp, "savic-review")
    b.OUT_SPLIT = os.path.join(tmp, "savic-review-v2")
    la.AUDITS = os.path.join(tmp, "audits")
    la.REGISTRY = os.path.join(la.AUDITS, "registry.json")
    os.makedirs(b.OUT, exist_ok=True)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r for r in (json.loads(line) for line in f if line.strip())}


def case(cid, decision, reviewer, draft=None, rationale=""):
    return {"id": cid, "state": {"statement": "s " + cid, "excerpt": "e " + cid},
            "gold": {"decision": decision}, "_reviewer": reviewer, "_human": reviewer == "Andrew Freeman",
            "_rationale": rationale, "_draft": draft}


def draft(**kw):
    d = {"note": "[claude-draft] reason"}
    d.update(kw)
    return d


def unit_tests():
    human, drafts = b.partition_confirmations({
        "H1": {"decision": "reject", "note": "Andrew's own"},
        "H2": {"decision": "approve", "note": ""},
        "H0": {"supported": "yes", "note": "no decision"},          # dropped, as before
        "D1": {"decision": "approve", "note": "[claude-draft] x"},
        "D2": {"supported": "no", "note": "  [claude-draft] split answers only"},
        "X1": {"decision": "reject", "note": "see [claude-draft] later"},  # prefix only counts at the start
    })
    check(sorted(human) == ["H1", "H2", "X1"], "partition: human gold = entries with a decision, no draft prefix")
    check(sorted(drafts) == ["D1", "D2"], "partition: drafts = notes starting with [claude-draft]")
    check(b.draft_value("supported", "partly") == 0.5 and b.draft_value("current", "No") is False
          and b.draft_value("decision", "approve") == "approve" and b.draft_value("relevant", "") is None,
          "draft_value maps yes/no/partly and decisions")

    with tempfile.TemporaryDirectory() as tmp:
        redirect(tmp)
        train = [
            # A recorded decision (Andrew's) with a contrary draft: the decision must stand.
            case("T-andrew", "approve", "Andrew Freeman", draft(decision="reject", supported="no", current="yes")),
            # A rule approval: no decision label of its own, so an audited draft decision may fill it.
            case("T-rule", "approve", "Rule: verified source", draft(decision="reject", supported="partly")),
            # A rationale-derived hard label wins over the draft.
            case("T-stale", "reject", "Claude (delegated)", draft(current="yes", relevant="no"), rationale="stale"),
            case("T-plain", "reject", "Claude (delegated)"),
        ]
        test = [case("X-andrew", "reject", "Andrew Freeman", draft(decision="approve", supported="yes",
                                                                      about_prospect="yes", relevant="yes",
                                                                      current="yes"))]

        b.write_split(train, test)  # no audit yet: drafts must be left out entirely
        rows = read_jsonl(os.path.join(b.OUT_SPLIT, "train.jsonl"))
        tst = read_jsonl(os.path.join(b.OUT_SPLIT, "test.jsonl"))
        check(rows["T-andrew"]["gold"]["decision"] == "approve", "unaudited: recorded decision kept")
        check(rows["T-andrew"]["gold"]["supported"] is None, "unaudited: draft split label left out")
        check(rows["T-rule"]["gold"]["decision"] is None, "unaudited: no draft decision on the rule row")
        check(rows["T-rule"]["gold"]["supported"] == 0.75, "unaudited: rule row keeps only its weak vote")
        check(tst["X-andrew"]["gold"] == {"decision": "reject", "supported": None, "about_prospect": None,
                                          "relevant": None, "current": None}, "test case ignores its draft")
        extract = os.path.join(b.OUT_SPLIT, "claude_draft_labels.jsonl")
        with open(extract, encoding="utf-8") as f:
            ex = [json.loads(line) for line in f]
        check(all(r["id"].startswith("T-") for r in ex), "audit extract holds train cases only")
        check(not any(r["question"] == "decision" and r["id"] != "T-rule" for r in ex),
              "audit extract offers a draft decision only where no decision is recorded")

        # Record a passing audit of the extract for every question, then rebuild.
        digest = la.sha256(extract)
        os.makedirs(la.AUDITS, exist_ok=True)
        with open(la.REGISTRY, "w", encoding="utf-8") as f:
            json.dump([{"source": "claude-drafts", "labels_sha256": digest, "question": q, "n": 30,
                        "agreement": 1.0, "wilson_lb": 0.89, "pass": True}
                       for q in ("decision",) + b.SPLIT_KEYS], f)
        b.write_split(train, test)
        rows = read_jsonl(os.path.join(b.OUT_SPLIT, "train.jsonl"))
        tst = read_jsonl(os.path.join(b.OUT_SPLIT, "test.jsonl"))
        check(rows["T-andrew"]["gold"]["decision"] == "approve", "audited: a draft never overrides a recorded decision")
        check(rows["T-andrew"]["gold"]["supported"] is False and rows["T-andrew"]["gold"]["current"] is True,
              "audited: draft split labels used on train")
        check(rows["T-rule"]["gold"]["decision"] == "reject", "audited: draft decision fills the rule row")
        check(rows["T-rule"]["gold"]["supported"] == 0.5, "audited: 'partly' becomes 0.5")
        check(rows["T-stale"]["gold"]["current"] is False and rows["T-stale"]["gold"]["relevant"] is False,
              "rationale hard label beats the draft; agreeing draft kept")
        check(tst["X-andrew"]["gold"]["decision"] == "reject" and tst["X-andrew"]["gold"]["supported"] is None,
              "audited: test case still ignores its draft")


def integration_test():
    """The real build, app databases read-only: a draft must not make a case Andrew's or move it to test."""
    app_db = os.path.join(b.APP, "data", "savic.sqlite3")
    if not os.path.exists(app_db):
        print("skip integration test: no %s" % app_db)
        return
    with tempfile.TemporaryDirectory() as tmp:
        redirect(tmp)
        b.SPLIT = False
        b.main()  # baseline, no confirmations.json
        with open(os.path.join(b.OUT, "all_cases.json"), encoding="utf-8") as f:
            cases = json.load(f)
        base_train = read_jsonl(os.path.join(b.OUT, "train.jsonl"))
        base_test = read_jsonl(os.path.join(b.OUT, "test.jsonl"))

        def held_out(cid):
            return int(hashlib.sha1(cid.encode()).hexdigest(), 16) % 3 == 0

        # A non-Andrew train case that would be held out if it became human gold.
        movable = next(c["id"] for c in cases if c["id"] in base_train and held_out(c["id"])
                       and not c["reviewer"].startswith("Andrew"))
        tested = next(iter(base_test))
        flip = {"approve": "reject", "reject": "approve"}
        conf = {"claims": {
            movable: {"decision": flip[base_train[movable]["gold"]["decision"]], "note": "[claude-draft] x"},
            tested: {"decision": flip[base_test[tested]["gold"]["decision"]], "note": "[claude-draft] y"},
        }}
        with open(os.path.join(b.OUT, "confirmations.json"), "w", encoding="utf-8") as f:
            json.dump(conf, f)
        b.main()
        train = read_jsonl(os.path.join(b.OUT, "train.jsonl"))
        test = read_jsonl(os.path.join(b.OUT, "test.jsonl"))
        check(set(train) == set(base_train) and set(test) == set(base_test), "drafts move no case between splits")
        check(train[movable]["gold"] == base_train[movable]["gold"], "draft decision does not override train gold")
        check(test[tested]["gold"] == base_test[tested]["gold"], "draft decision does not override test gold")

        # The same entries without the prefix are human gold: the train case moves to test.
        for v in conf["claims"].values():
            v["note"] = "Andrew"
        with open(os.path.join(b.OUT, "confirmations.json"), "w", encoding="utf-8") as f:
            json.dump(conf, f)
        b.main()
        test = read_jsonl(os.path.join(b.OUT, "test.jsonl"))
        check(movable in test and test[tested]["gold"]["decision"] != base_test[tested]["gold"]["decision"],
              "control: unprefixed entries stay human gold (override, may enter test)")


if __name__ == "__main__":
    unit_tests()
    integration_test()
    print("%d failure(s)" % len(FAILS))
    sys.exit(1 if FAILS else 0)
