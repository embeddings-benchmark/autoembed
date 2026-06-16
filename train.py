# The agent edits this file. Everything is fair game except prepare.py.
import time

from prepare import (TRAIN_BUDGET_SECONDS, MODEL_DIR, SEED,
                     load_train_pool, evaluate, record_run)
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from transformers import TrainerCallback

BASE_MODEL = "nreimers/MiniLM-L6-H384-uncased"
TRAIN_EXAMPLES = 50_000
BATCH_SIZE = 64
LR = 2e-5
MAX_STEPS = 2000


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
    score, per_task = evaluate()
    print(f"DEV score={score:.4f} per_task={per_task} time={time.time()-t0:.0f}s")
    record_run(score, per_task, notes=f"{BASE_MODEL} bs={BATCH_SIZE} lr={LR}")


if __name__ == "__main__":
    main()
