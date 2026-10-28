"""LLM-as-judge: grade each answer against the expected answer from the eval set."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

IN_PATH = Path("results/answers_v2.jsonl")
OUT_PATH = Path("results/judged_v2.jsonl")
VERDICTS = ("correct", "partial", "incorrect")

JUDGE_PROMPT = """You are grading answers about UK visas against a reference answer written from official gov.uk guidance.

Grade the CANDIDATE answer:
- "correct": it contains the key facts of the reference and states nothing that contradicts it.
- "partial": it gets the main point right but misses an important condition or number from the reference.
- "incorrect": it contradicts the reference, gives a wrong key fact, or answers a different question.

Extra details not in the reference are fine unless they contradict it. Ignore citation markers like [1].

Reply with JSON only: {"verdict": "correct" | "partial" | "incorrect", "reason": "<one sentence>"}"""

def judge(gen, question: str, expected: str, answer: str) -> dict:
    text = gen._call([
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": f"QUESTION: {question}\n\nREFERENCE: {expected}\n\nCANDIDATE: {answer}"},
    ])
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        result = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        result = {}
    if result.get("verdict") not in VERDICTS:
        return {"verdict": "unparsed", "reason": text[:200]}
    return result

def run() -> list[dict]:
    from src.generate import AnswerGenerator

    gen = AnswerGenerator()
    rows = [json.loads(line) for line in IN_PATH.open()]
    judged = []
    with OUT_PATH.open("w") as f:
        for r in rows:
            if r["category"] == "out_of_scope":
                continue
            for mode in ("rag", "closed_book"):
                a = r[mode]
                if a["not_found"]:
                    v = {"verdict": "refused", "reason": "system refused to answer"}
                else:
                    v = judge(gen, r["question"], r["expected"], a["answer"])

                out = {
                    "id": r["id"],
                    "category": r["category"], 
                    "mode": mode,
                    "question": r["question"], 
                    "expected": r["expected"],
                    "answer": a["answer"], 
                    **v
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                f.flush()
                judged.append(out)
                print(f"{r['id']} {mode:<11} {v['verdict']}")
    return judged

def report(rows: list[dict]) -> None:
    for mode in ("rag", "closed_book"):
        sub = [r for r in rows if r["mode"] == mode]
        counts = Counter(r["verdict"] for r in sub)
        print(f"\n=== {mode}: {len(sub)} in-scope questions ===")
        for v in (*VERDICTS, "refused", "unparsed"):
            if counts[v]:
                print(f"  {v:<10} {counts[v]:>2}  ({counts[v] / len(sub):.0%})")
    print("\nIncorrect or partial:")
    for r in rows:
        if r["verdict"] in ("incorrect", "partial"):
            print(f"  {r['id']} {r['mode']:<11} {r['verdict']:<9} {r['reason']}")

if __name__ == "__main__":
    if "--report" in sys.argv:
        report([json.loads(line) for line in OUT_PATH.open()])
    else:
        report(run())

                