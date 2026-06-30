# Human-run held-out evaluation. Not part of the agent's loop: the agent never
# sees these tasks or their scores. Run it on the saved MODEL_DIR to check
# generalization, and to audit whether training data leaked into the held-out set.
import json
import time

from prepare import MODEL_DIR, RUNS_DIR, evaluate, _eval_texts

HELDOUT_TASKS = ["FiQA2018", "TRECCOVID", "CQADupstackEnglishRetrieval"]


def main():
    score, per_task = evaluate(tasks=HELDOUT_TASKS, tag="heldout")

    contam = None
    sample = RUNS_DIR / "_train_texts.json"
    if sample.exists():
        train = set(json.loads(sample.read_text()))
        evalset = _eval_texts(HELDOUT_TASKS)
        hits = sum(1 for t in train if t in evalset)
        contam = {"train_texts": len(train), "eval_texts": len(evalset),
                  "hits": hits, "frac": round(hits / max(len(train), 1), 6)}

    line = f"- {time.strftime('%Y%m%d-%H%M%S')}  heldout={score:.4f}  {per_task}  contam={contam}"
    with open(RUNS_DIR / "heldout.md", "a") as f:
        f.write(line + "\n")
    print(line)


if __name__ == "__main__":
    main()
