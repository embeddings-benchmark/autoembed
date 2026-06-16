# Frozen. The agent must not edit this file.
import json
import time
from pathlib import Path

ROOT = Path(__file__).parent
RUNS_DIR = ROOT / "runs"
MODEL_DIR = ROOT / "out" / "model"
SEED = 0
TRAIN_BUDGET_SECONDS = 5 * 60

# Dev proxy, disjoint from train. Verify it tracks full MTEB before trusting it.
DEV_TASKS = ["SciFact", "NFCorpus", "STSBenchmark", "Banking77Classification"]
HELDOUT_TASKS = ["FiQA2018", "TwentyNewsgroupsClustering", "SprintDuplicateQuestions"]
TASK_LANGS = ["eng"]


def load_train_pool():
    # NLI + MS MARCO; disjoint from the BEIR eval tasks.
    from datasets import load_dataset, concatenate_datasets

    def triplet(name):
        ds = load_dataset(name, "triplet", split="train")
        cols = ds.column_names[:3]
        return ds.select_columns(cols).rename_columns(
            dict(zip(cols, ["anchor", "positive", "negative"])))

    return concatenate_datasets([triplet("sentence-transformers/all-nli"),
                                 triplet("sentence-transformers/msmarco")]).shuffle(seed=SEED)


def evaluate(model_path=MODEL_DIR, tasks=DEV_TASKS, tag="dev"):
    import mteb
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(model_path))
    suite = mteb.MTEB(tasks=mteb.get_tasks(tasks=tasks, languages=TASK_LANGS))
    results = suite.run(model, output_folder=str(RUNS_DIR / "mteb" / tag),
                        verbosity=0, overwrite_results=True)
    per_task = {(getattr(r, "task_name", None) or r.task.metadata.name): float(r.get_score())
                for r in results}
    return sum(per_task.values()) / len(per_task), per_task


def record_run(score, per_task, notes=""):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    (RUNS_DIR / f"{run_id}.json").write_text(json.dumps(
        {"id": run_id, "score": score, "per_task": per_task, "notes": notes,
         "train_py": (ROOT / "train.py").read_text()}, indent=2))
    with open(RUNS_DIR / "leaderboard.md", "a") as f:
        f.write(f"- {run_id}  score={score:.4f}  {notes}\n")
    return run_id
