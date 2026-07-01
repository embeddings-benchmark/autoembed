# Harness scoring of the agent's final_model, run OUTSIDE the agent's workdir so
# the held-out tasks are never exposed. Reports dev, hidden held-out, and a
# training-data contamination audit; writes scores.json if an output path is given.
import json
import sys
from pathlib import Path

import mteb

from task import evaluate, _eval_texts, MODEL_DIR

# Held-out: MTEB(eng, v2) minus the dev datasets (MindSmall + StackExchange-P2P dropped).
_RESERVED = {"ArguAna", "SCIDOCS", "STS12", "STSBenchmark", "Banking77Classification",
             "StackExchangeClustering.v2", "StackExchangeClusteringP2P.v2",
             "AskUbuntuDupQuestions", "SprintDuplicateQuestions", "MindSmallReranking"}
HELDOUT_TASKS = [t for t in mteb.get_benchmark("MTEB(eng, v2)").tasks
                 if t.metadata.name not in _RESERVED]

model = sys.argv[1] if len(sys.argv) > 1 else str(MODEL_DIR)
out = sys.argv[2] if len(sys.argv) > 2 else None


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

    result = {"dev": dev, "heldout": ho, "dev_per_task": dev_per,
              "heldout_per_task": ho_per, "contamination": contam}
    print(f"DEV={dev:.4f}  HELDOUT={ho:.4f}  contam={contam}")
    if out:
        Path(out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
