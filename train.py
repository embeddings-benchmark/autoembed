# The agent edits this file. Everything is fair game except prepare.py.
import time

from prepare import (TRAIN_BUDGET_SECONDS, MODEL_DIR, SEED, BASE_MODEL,
                     load_train_pool, evaluate, record_run, check_contamination)
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from transformers import TrainerCallback

TRAIN_EXAMPLES = 200_000
BATCH_SIZE = 128
LR = 2e-5
MAX_STEPS = 1000


class WallClock(TrainerCallback):
    def on_train_begin(self, args, state, control, **kw):
        self.deadline = time.time() + TRAIN_BUDGET_SECONDS

    def on_step_end(self, args, state, control, **kw):
        if time.time() > self.deadline:
            control.should_training_stop = True
        return control


def main():
    t0 = time.time()
    model = SentenceTransformer(BASE_MODEL)
    pool = load_train_pool()
    train_ds = pool.select(range(min(TRAIN_EXAMPLES, len(pool))))
    contam = check_contamination(train_ds)
    print(f"CONTAMINATION {contam}")

    loss = MultipleNegativesRankingLoss(model)
    args = SentenceTransformerTrainingArguments(
        output_dir="out/ckpt", per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LR, max_steps=MAX_STEPS, seed=SEED, bf16=True,
        report_to=[], logging_steps=50, save_strategy="no")
    trainer = SentenceTransformerTrainer(model=model, args=args,
                                         train_dataset=train_ds, loss=loss,
                                         callbacks=[WallClock()])
    trainer.train()
    model.save_pretrained(str(MODEL_DIR))

    steps = trainer.state.global_step
    examples = steps * BATCH_SIZE  # effective batch on a single device
    score, per_task = evaluate()
    print(f"DEV={score:.4f} steps={steps} examples={examples} time={time.time()-t0:.0f}s")
    record_run(score, per_task,
               notes=f"{BASE_MODEL} bs={BATCH_SIZE} lr={LR} steps={steps}",
               contamination=contam, examples=examples)


if __name__ == "__main__":
    main()
