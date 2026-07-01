# The task: base model, dev eval, and contamination check.
import json
import os
from pathlib import Path

ROOT = Path(__file__).parent
RUNS_DIR = ROOT / "runs"
MODEL_DIR = ROOT / "final_model"  # the agent's submitted model
BASE_MODEL = os.environ.get("AUTOEMBED_BASE_MODEL", "microsoft/mpnet-base")

# Dev proxy, disjoint from the held-out.
DEV_TASKS = ["NanoArguAnaRetrieval", "NanoSCIDOCSRetrieval",   # retrieval
             "STS12", "STSBenchmark",                          # STS
             "Banking77Classification",                        # classification
             "StackExchangeClustering.v2",                     # clustering
             "AskUbuntuDupQuestions",                          # reranking
             "SprintDuplicateQuestions"]                       # pair classification
TASK_LANGS = ["eng"]


def _resolve(tasks):   # accept task names or task objects
    import mteb
    if tasks and isinstance(tasks[0], str):
        return mteb.get_tasks(tasks=tasks, languages=TASK_LANGS)
    return list(tasks)


def evaluate(model_path=MODEL_DIR, tasks=DEV_TASKS, tag="dev"):
    import mteb
    from sentence_transformers import SentenceTransformer
    task_objs = _resolve(tasks)
    type_of = {t.metadata.name: t.metadata.type for t in task_objs}
    model = SentenceTransformer(str(model_path))
    results = mteb.MTEB(tasks=task_objs).run(
        model, output_folder=str(RUNS_DIR / "mteb" / tag),
        verbosity=0, overwrite_results=True)
    per_task = {(getattr(r, "task_name", None) or r.task.metadata.name): float(r.get_score())
                for r in results}
    per_type = {}
    for name, sc in per_task.items():
        per_type.setdefault(type_of.get(name, name), []).append(sc)
    type_means = [sum(v) / len(v) for v in per_type.values()]
    return sum(type_means) / len(type_means), per_task


def _norm(s):
    return " ".join(s.lower().split())


def _collect(obj, out, cap):
    if len(out) >= cap:
        return
    if isinstance(obj, str):
        if obj.strip():
            out.add(_norm(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect(v, out, cap)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect(v, out, cap)
    elif hasattr(obj, "column_names"):  # datasets.Dataset
        for col in obj.column_names:
            _collect(obj[col], out, cap)


def _eval_texts(tasks, cap=200_000):
    out = set()
    for t in _resolve(tasks):
        t.load_data()
        for attr in ("corpus", "queries", "dataset"):
            _collect(getattr(t, attr, None), out, cap)
    return out


def check_contamination(train_dataset, tasks=None, sample=100_000):
    # Exact-match overlap of training text with the eval; writes a training-text sample.
    tasks = tasks or DEV_TASKS
    evalset = _eval_texts(tasks)
    n = min(sample, len(train_dataset))
    ds = train_dataset.select(range(n))
    cols = [c for c in ("anchor", "positive", "negative") if c in ds.column_names]
    train_texts, hits, examples = set(), 0, []
    for col in cols:
        for s in ds[col]:
            if not s:
                continue
            t = _norm(s)
            train_texts.add(t)
            if t in evalset:
                hits += 1
                if len(examples) < 5:
                    examples.append(s[:80])
    total = max(n * len(cols), 1)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / "_train_texts.json").write_text(json.dumps(sorted(train_texts)))
    return {"checked": total, "eval_texts": len(evalset), "hits": hits,
            "frac": round(hits / total, 6), "examples": examples}
