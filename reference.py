# Reference anchors: score off-the-shelf models on the DEV proxy for comparison.
# Writes the results to runs/reference.md.
from task import evaluate, DEV_TASKS, RUNS_DIR, BASE_MODEL

MODELS = [BASE_MODEL,                                  # raw base = the floor to beat
          "sentence-transformers/all-mpnet-base-v2"]   # same-base anchor (fully trained)


def main():
    lines = []
    for m in MODELS:
        try:
            s, _ = evaluate(model_path=m, tasks=DEV_TASKS, tag="ref")
            line = f"- {s:.4f}  {m}"
        except Exception as e:
            line = f"- FAIL  {m}: {repr(e)[:120]}"
        print(line)
        lines.append(line)
    (RUNS_DIR / "reference.md").write_text(
        "# Reference anchors (DEV proxy)\n\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
