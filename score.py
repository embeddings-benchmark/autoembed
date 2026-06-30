# Harness scoring of the agent's final_model, run OUTSIDE the agent's workdir so
# the held-out tasks are never exposed. Reports dev (what the agent optimized),
# hidden held-out (generalization), and a training-data contamination audit.
import json
import sys
from pathlib import Path

from task import evaluate, _eval_texts, MODEL_DIR

# Held-out retrieval tasks: disjoint from the dev proxy and from the training data.
HELDOUT_TASKS = ["FiQA2018", "TRECCOVID", "CQADupstackEnglishRetrieval"]

model = sys.argv[1] if len(sys.argv) > 1 else str(MODEL_DIR)


def main():
    dev, dev_per = evaluate(model_path=model, tag="dev")
    ho, ho_per = evaluate(model_path=model, tasks=HELDOUT_TASKS, tag="heldout")

    contam = None
    sample = Path(model).resolve().parent / "runs" / "_train_texts.json"
    if sample.exists():
        train = set(json.loads(sample.read_text()))
        evalset = _eval_texts(HELDOUT_TASKS)
        hits = sum(1 for t in train if t in evalset)
        contam = {"hits": hits, "frac": round(hits / max(len(train), 1), 6)}

    print(f"DEV={dev:.4f}  HELDOUT={ho:.4f}  contam={contam}")
    print(f"dev_per_task={dev_per}")
    print(f"heldout_per_task={ho_per}")


if __name__ == "__main__":
    main()
