# Frozen. The agent must not edit this file.
import json
import time
from pathlib import Path

ROOT = Path(__file__).parent
RUNS_DIR = ROOT / "runs"
MODEL_DIR = ROOT / "out" / "model"
SEED = 0
TRAIN_BUDGET_SECONDS = 30 * 60
N_MSMARCO = 1_000_000
BASE_MODEL = "microsoft/mpnet-base"  # fixed: study isolates method/data, not base choice

# Fast dev proxy: NanoBEIR retrieval, minus MSMARCO (in train) and FiQA (in heldout).
# The held-out set lives in eval_heldout.py and is never exposed to the agent.
DEV_TASKS = ["NanoArguAnaRetrieval", "NanoClimateFeverRetrieval", "NanoDBPediaRetrieval",
             "NanoFEVERRetrieval", "NanoHotpotQARetrieval", "NanoNFCorpusRetrieval",
             "NanoNQRetrieval", "NanoQuoraRetrieval", "NanoSCIDOCSRetrieval",
             "NanoSciFactRetrieval", "NanoTouche2020Retrieval"]
TASK_LANGS = ["eng"]


def load_train_pool():
    # NLI + MS MARCO, disjoint from the eval tasks. MS MARCO ships as ID triplets;
    # we join them to the real query/passage text, not the bare ID strings.
    from datasets import load_dataset, concatenate_datasets

    nli = load_dataset("sentence-transformers/all-nli", "triplet", split="train")
    nli = nli.select_columns(["anchor", "positive", "negative"])

    q = load_dataset("sentence-transformers/msmarco", "queries", split="train")
    qmap = dict(zip(q["query_id"], q["query"]))
    c = load_dataset("sentence-transformers/msmarco", "corpus", split="train")
    cmap = dict(zip(c["passage_id"], c["passage"]))

    trip = load_dataset("sentence-transformers/msmarco", "triplets", split="train")
    step = max(1, len(trip) // N_MSMARCO)  # strided pick for diversity, no full shuffle
    trip = trip.select(range(0, step * N_MSMARCO, step))

    def to_text(b):
        return {"anchor": [qmap.get(i, "") for i in b["query_id"]],
                "positive": [cmap.get(i, "") for i in b["positive_id"]],
                "negative": [cmap.get(i, "") for i in b["negative_id"]]}

    ms = trip.map(to_text, batched=True, remove_columns=trip.column_names)
    ms = ms.filter(lambda b: [bool(a and p and n) for a, p, n in
                              zip(b["anchor"], b["positive"], b["negative"])],
                   batched=True)
    return concatenate_datasets([nli, ms]).shuffle(seed=SEED)


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
    import mteb
    out = set()
    for t in mteb.get_tasks(tasks=tasks, languages=TASK_LANGS):
        t.load_data()
        for attr in ("corpus", "queries", "dataset"):
            _collect(getattr(t, attr, None), out, cap)
    return out


def check_contamination(train_dataset, tasks=None, sample=100_000):
    # Exact-match overlap between training and eval text; pass the result to
    # record_run. Persists the sampled training text for the held-out audit.
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


def record_run(score, per_task, notes="", contamination=None, examples=None):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    payload = {"id": run_id, "score": score, "per_task": per_task, "notes": notes,
               "train_py": (ROOT / "train.py").read_text()}
    if contamination is not None:
        payload["contamination"] = contamination
    if examples is not None:
        payload["examples_seen"] = examples
    (RUNS_DIR / f"{run_id}.json").write_text(json.dumps(payload, indent=2))
    ex = f" ex={examples}" if examples is not None else ""
    with open(RUNS_DIR / "leaderboard.md", "a") as f:
        f.write(f"- {run_id}  score={score:.4f}{ex}  {notes}\n")
    return run_id
