"""Label the split review questions with TypeSafe's Jev as a teacher.

Same cases and questions as scripts/teacher_label.py (Ollama), but Jev answers
each as a noul, so one call per case gives p_true directly (no sampling).
Writes data/savic-review/teacher.jev.jsonl in the teacher.jsonl row layout and
never touches teacher.jsonl. Like any model-made labels it is used only after
a passing audit (scripts/label_audit.py --labels <this file> --source teacher-jev).
Resumable: cases already in the output are skipped.

The key comes from JEV_API_KEY, else from --env-file (a KEY=value file such as
frzrbrn-prospect-intelligence/.env.jev). It is never printed.

Usage: python -I scripts/jev_teacher_label.py [--limit N] [--env-file PATH] [--model jev-latest]
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.request

URL = "https://api.typesafe.ai/v1/systemone"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "savic-review")

CONTEXT = (
    "A reviewer is vetting a claim about a prospect company for SAVIC, an ERP and "
    "technology-modernization consultancy. Judge only from `statement` and `excerpt`."
)
# Wording follows scripts/teacher_label.py, so both teachers answer the same questions.
QUESTIONS = {
    "supported": "Does `excerpt` state or directly imply every part of `statement` as worded?",
    "about_prospect": "Is `statement` about the prospect company itself, not another company, vendor or customer?",
    "relevant": (
        "Does `statement` say something about the prospect's operations, systems, finances, growth, "
        "risks or change that an ERP or modernization pitch could use (not product marketing, legal "
        "boilerplate, careers or biography)?"
    ),
    # Tried 2026-10-14: asking this as two questions (still true / ownership milestone) and
    # recombining scored worse on the 30 checked `current` answers (0.53-0.67 vs 0.67).
    "current": (
        "Is `statement` still true today, or a dated milestone in the company's ownership, acquisition, "
        "divestiture or legal-entity history (these shape the ERP landscape at any age)? Answer no for a "
        "past or superseded one-off: a launch, campaign, sponsorship, job posting, plan, forecast, "
        "expected synergy, deal rumour, or risk boilerplate from an old filing. Age alone never makes it no."
    ),
}


def load_key(env_file):
    key = os.environ.get("JEV_API_KEY", "")
    if not key and env_file:
        with open(env_file, encoding="utf-8") as f:
            for line in f:
                if line.startswith("JEV_API_KEY="):
                    key = line.split("=", 1)[1].strip()
    if not key:
        raise SystemExit("No JEV_API_KEY (set it or pass --env-file).")
    return key


def ask(case, key, model):
    body = json.dumps({
        "model": model,
        "state": {"context": CONTEXT, "statement": case["statement"], "excerpt": case["excerpt"]},
        "questions": {k: {"type": "noul", "instructions": q} for k, q in QUESTIONS.items()},
    }).encode()
    req = urllib.request.Request(URL, body, {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                reply = json.loads(r.read())
            return reply["model"], {k: round(a["noul"], 3) for k, a in reply["answers"].items()}
        except urllib.error.HTTPError as e:
            if e.code in (429, 529) and attempt < 4:
                time.sleep(2 ** attempt)
                continue
            raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0, help="label at most N new cases (0 = all)")
    p.add_argument("--env-file")
    p.add_argument("--model", default="jev-latest")
    a = p.parse_args()
    key = load_key(a.env_file)
    with open(os.path.join(OUT, "all_cases.json"), encoding="utf-8") as f:
        cases = [c for c in json.load(f) if c["excerpt"] and c["statement"]]
    path = os.path.join(OUT, "teacher.jev.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    todo = [c for c in cases if c["id"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    with open(path, "a", encoding="utf-8") as out:
        for i, c in enumerate(todo):
            try:
                served, soft = ask(c, key, a.model)
            except Exception as e:  # one bad case should not stop the run
                print("case failed", c["id"], type(e).__name__, getattr(e, "code", ""), flush=True)
                continue
            out.write(json.dumps({"id": c["id"], "teacher": served, "samples": 1, "p_true": soft}) + "\n")
            out.flush()
            if i % 20 == 0:
                print("{}/{} {}".format(i + 1, len(todo), c["id"]), flush=True)
    print("done", path, flush=True)


if __name__ == "__main__":
    main()
