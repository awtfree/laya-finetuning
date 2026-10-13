"""Label the split review questions with the local Ollama model as a teacher.

For every case in data/savic-review/all_cases.json that has an excerpt, asks the
local model the four split questions SAMPLES times at temperature 0.7 and
records the share of "yes" answers as a soft label. Stays on this machine
(Ollama at 127.0.0.1). Resumable: cases already in teacher.jsonl are skipped.

Usage: python -I scripts/teacher_label.py [model] [samples]
"""
import json
import os
import sys
import urllib.request

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:14b"
SAMPLES = int(sys.argv[2]) if len(sys.argv) > 2 else 3
OLLAMA = "http://127.0.0.1:11434/api/chat"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "savic-review")

SYSTEM = (
    "You vet claims about a prospect company for SAVIC, an ERP and technology-modernization "
    "consultancy. Judge only from the statement and excerpt given. Answer with JSON only."
)
ASK = """Statement: {statement}

Excerpt: {excerpt}

Answer each with true or false:
- supported: the excerpt states or directly implies every part of the statement as worded
- about_prospect: the statement is about the prospect company itself, not another company, vendor or customer
- relevant: it says something about the prospect's operations, systems, finances, growth, risks or change that an ERP or modernization pitch could use (not product marketing, legal boilerplate, careers or biography)
- current: nothing in the text suggests it is stale (for example from a filing many years old)

Return {{"supported": bool, "about_prospect": bool, "relevant": bool, "current": bool}}"""
KEYS = ("supported", "about_prospect", "relevant", "current")


def ask(case):
    body = json.dumps({
        "model": MODEL, "stream": False, "format": "json", "keep_alive": "30m",
        "options": {"temperature": 0.7, "num_ctx": 4096},
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": ASK.format(**case)}],
    }).encode()
    req = urllib.request.Request(OLLAMA, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(json.loads(r.read())["message"]["content"])


def main():
    with open(os.path.join(OUT, "all_cases.json"), encoding="utf-8") as f:
        cases = [c for c in json.load(f) if c["excerpt"] and c["statement"]]
    path = os.path.join(OUT, "teacher.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    with open(path, "a", encoding="utf-8") as out:
        for i, c in enumerate(cases):
            if c["id"] in done:
                continue
            votes = {k: [] for k in KEYS}
            for _ in range(SAMPLES):
                try:
                    a = ask(c)
                except Exception as e:  # one bad sample should not stop the run
                    print("sample failed", c["id"], e, flush=True)
                    continue
                for k in KEYS:
                    if isinstance(a.get(k), bool):
                        votes[k].append(a[k])
            soft = {k: round(sum(v) / len(v), 3) for k, v in votes.items() if v}
            out.write(json.dumps({"id": c["id"], "teacher": MODEL, "samples": SAMPLES, "p_true": soft}) + "\n")
            out.flush()
            if i % 20 == 0:
                print("{}/{} {}".format(i + 1, len(cases), c["id"]), flush=True)
    print("done", path, flush=True)


if __name__ == "__main__":
    main()
