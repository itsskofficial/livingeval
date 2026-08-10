"""Rung 5: a fine-tuned model.

This rung exists for a different reason from the other five, and the difference is the
whole point of putting it here rather than leaving it out.

**Rungs 0-4 are diagnostics.** They are deliberately weak in *nameable* ways, so a
match tells you what the judge's decision was a function of: one phrase, response
length, word choice, formatting. A fine-tune has no such reading. A high kappa here
means "a transformer with enough capacity can fit a self-consistent labelling
function", which is true of almost any judge and tells you nothing about that judge.

**So this rung answers the deployment question, not the diagnostic one.** It is the
right thing to reach for in exactly one situation: `judge_depth` came back `None`, no
cheap rung reproduces the judge, and you still want per-turn online scoring. Then a
fine-tuned small encoder is the correct engineering move, and the ladder has already
given you the kappa floor it has to beat.

Both facts are printed next to the number, every time, so nobody reads a strong rung 5
as evidence about the judge:

    finetune   0.9312 [0.9120, 0.9490]   0.9640    1840.0    0.0000  <-
      NOTE: a fine-tuned rung is a deployment number, not a diagnostic. It clearing the
      bar says a transformer can fit this judge, which is true of most judges. Read
      rungs 0-4 for what the judge was reading.

Three backends, because the constraint differs:

| backend | what it does | needs |
|---|---|---|
| `local` | fine-tunes a small HF encoder in a plain torch loop | `torch`, `transformers` |
| `unsloth` | Unsloth's fast LoRA path for a larger model | `unsloth` |
| `fireworks` | hosted fine-tune + hosted inference | `fireworks-ai`, a key |

`local` is the default and runs on CPU with a tiny checkpoint. A plain training loop
rather than `Trainer` on purpose: `Trainer` pulls in `accelerate` and `datasets`, and
the loop for single-label classification is thirty lines that never surprise anyone.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Sequence

import numpy as np

from livingeval.scorer.rungs import Rung

__all__ = [
    "FINETUNE_NOTE",
    "FinetunedRung",
    "FireworksBackend",
    "LocalFinetuneBackend",
    "UnslothBackend",
]

FINETUNE_NOTE = (
    "a fine-tuned rung is a deployment number, not a diagnostic. It clearing the bar "
    "says a transformer can fit this judge, which is true of most judges. Read rungs "
    "0-4 for what the judge was reading."
)


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------


class LocalFinetuneBackend:
    """Fine-tune a small Hugging Face encoder with a plain torch loop.

    Defaults to `sentence-transformers/all-MiniLM-L6-v2` (~22 M parameters, 6 layers):
    small enough to train on CPU inside a cross-validated ladder rather than as an
    overnight job, and it ships a fast tokenizer.

    That last point is not a detail. Several of the obvious "tiny BERT" checkpoints
    (`prajjwal1/bert-tiny` among them) publish only a slow tokenizer, and
    `transformers` 5.x refuses to convert one without `sentencepiece` or `tiktoken`
    installed. Defaulting to a checkpoint that ships `tokenizer.json` means this rung
    works on a fresh install instead of failing with an error about backend
    tokenizers. It is also the same checkpoint `embed.hf()` defaults to, so the two
    subsystems share one download.

    Swap in `distilbert-base-uncased` or anything larger when you care about the
    ceiling rather than the loop time.
    """

    name = "local"

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        epochs: int = 4,
        lr: float = 5e-5,
        batch_size: int = 32,
        max_length: int = 256,
        device: str | None = None,
        seed: int = 0,
        revision: str = "main",
    ):
        self.model_name = model
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.max_length = max_length
        self.seed = seed
        self.revision = revision
        self._device = device
        self.label = f"local:{model}@{revision}(epochs={epochs})"

    def _load(self, n_classes: int):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:  # pragma: no cover - optional extra
            raise ImportError("pip install 'livingeval[finetune]'") from e
        # A freshly-initialised classification head is expected here; the load report
        # about missing `classifier.*` weights is noise in a cross-validated ladder
        # that trains one model per fold.
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        with contextlib.suppress(AttributeError):  # transformers version drift
            hf_logging.disable_progress_bar()
        torch.manual_seed(self.seed)
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            tok = AutoTokenizer.from_pretrained(self.model_name, revision=self.revision)
        except (ValueError, OSError) as e:
            raise RuntimeError(
                f"could not load a tokenizer for {self.model_name!r}: {e}\n"
                "Some small checkpoints publish only a slow tokenizer, which transformers 5.x "
                "will not convert without `sentencepiece` installed. Either "
                "`pip install sentencepiece` or pass a checkpoint that ships tokenizer.json, "
                "e.g. model='sentence-transformers/all-MiniLM-L6-v2'."
            ) from e
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, revision=self.revision, num_labels=n_classes
        ).to(device)
        return torch, tok, model, device

    def fit(self, texts: Sequence[str], labels: np.ndarray):
        y = np.asarray(labels, dtype=int)
        n_classes = int(y.max()) + 1 if y.size else 2
        torch_mod, tok, model, device = self._load(max(2, n_classes))
        self._tok, self._model, self._device_resolved = tok, model, device

        # Class weights, because eval corpora are routinely 80/20 and an unweighted
        # loss on a small model converges straight to the majority class - which would
        # make this rung a slower `majority`.
        counts = np.bincount(y, minlength=max(2, n_classes)).astype(float)
        weights = torch_mod.tensor(
            (counts.sum() / np.maximum(counts, 1.0)) / len(counts), dtype=torch_mod.float32
        ).to(device)
        loss_fn = torch_mod.nn.CrossEntropyLoss(weight=weights)
        optimiser = torch_mod.optim.AdamW(model.parameters(), lr=self.lr)

        order = np.arange(len(y))
        rng = np.random.default_rng(self.seed)
        model.train()
        for _ in range(self.epochs):
            rng.shuffle(order)
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                batch = tok(
                    [texts[i] for i in idx], padding=True, truncation=True,
                    max_length=self.max_length, return_tensors="pt",
                ).to(device)
                logits = model(**batch).logits
                loss = loss_fn(logits, torch_mod.tensor(y[idx], dtype=torch_mod.long).to(device))
                loss.backward()
                optimiser.step()
                optimiser.zero_grad()
        model.eval()
        return self

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        out: list[int] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = list(texts[start : start + self.batch_size])
            batch = self._tok(
                chunk, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            ).to(self._device_resolved)
            with torch.no_grad():
                logits = self._model(**batch).logits
            out.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
        return np.asarray(out, dtype=int)


class UnslothBackend:
    """Unsloth's fast LoRA fine-tune, for when a bigger base model is worth it.

    Unsloth targets causal LMs, so classification is done as constrained generation:
    the model is trained to emit `PASS` or `FAIL` and the prediction reads the first
    token. Cruder than a classification head and it lets you use a model whose
    pretraining actually helps on the task.
    """

    name = "unsloth"

    def __init__(self, model: str = "unsloth/Llama-3.2-1B-Instruct", max_steps: int = 60,
                 lr: float = 2e-4, max_seq_length: int = 1024, load_in_4bit: bool = True,
                 seed: int = 0):
        self.model_name = model
        self.max_steps = max_steps
        self.lr = lr
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.seed = seed
        self.label = f"unsloth:{model}(steps={max_steps})"

    _PROMPT = "Did the agent do the right thing?\n\n{text}\n\nAnswer:"

    def fit(self, texts: Sequence[str], labels: np.ndarray):  # pragma: no cover - heavy
        try:
            from trl import SFTConfig, SFTTrainer
            from unsloth import FastLanguageModel
        except ImportError as e:
            raise ImportError(
                "pip install unsloth trl  (Unsloth needs a CUDA GPU; use "
                "backend='local' on CPU)"
            ) from e

        model, tokenizer = FastLanguageModel.from_pretrained(
            self.model_name, max_seq_length=self.max_seq_length, load_in_4bit=self.load_in_4bit
        )
        model = FastLanguageModel.get_peft_model(
            model, r=16, lora_alpha=16, lora_dropout=0.0, bias="none", random_state=self.seed,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        rows = [
            {"text": self._PROMPT.format(text=t) + (" PASS" if y else " FAIL")}
            for t, y in zip(texts, np.asarray(labels, dtype=int), strict=True)
        ]
        trainer = SFTTrainer(
            model=model, tokenizer=tokenizer, train_dataset=rows, dataset_text_field="text",
            args=SFTConfig(
                max_steps=self.max_steps, learning_rate=self.lr, per_device_train_batch_size=2,
                gradient_accumulation_steps=4, logging_steps=20, seed=self.seed,
                output_dir="outputs", report_to="none",
            ),
        )
        trainer.train()
        FastLanguageModel.for_inference(model)
        self._model, self._tokenizer = model, tokenizer
        return self

    def predict(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - heavy
        import torch

        out = []
        for text in texts:
            prompt = self._PROMPT.format(text=text)
            batch = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                generated = self._model.generate(**batch, max_new_tokens=3, do_sample=False)
            answer = self._tokenizer.decode(
                generated[0][batch["input_ids"].shape[1] :], skip_special_tokens=True
            )
            out.append(int("PASS" in answer.upper()))
        return np.asarray(out, dtype=int)


class FireworksBackend:
    """Hosted fine-tune and hosted inference on Fireworks.

    Slowest to iterate on because it round-trips a training job, and the one to use when
    you have no GPU and the `local` ceiling is not enough. `FIREWORKS_API_KEY` from the
    environment.
    """

    name = "fireworks"

    def __init__(self, base_model: str = "accounts/fireworks/models/llama-v3p2-3b-instruct",
                 epochs: int = 2, poll_seconds: float = 20.0, timeout_seconds: float = 3600.0):
        self.base_model = base_model
        self.epochs = epochs
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds
        self.label = f"fireworks:{base_model}(epochs={epochs})"

    def fit(self, texts: Sequence[str], labels: np.ndarray):  # pragma: no cover - network
        import json
        import os
        import tempfile

        if not os.environ.get("FIREWORKS_API_KEY"):
            raise RuntimeError("set FIREWORKS_API_KEY in the environment")
        import importlib.util

        if importlib.util.find_spec("fireworks") is None:
            raise ImportError("pip install fireworks-ai")

        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as fh:
            for t, y in zip(texts, np.asarray(labels, dtype=int), strict=True):
                fh.write(json.dumps({"messages": [
                    {"role": "user", "content": f"Did the agent do the right thing?\n\n{t}"},
                    {"role": "assistant", "content": "PASS" if y else "FAIL"},
                ]}) + "\n")
            self.dataset_path = fh.name

        # Fireworks' fine-tuning surface has moved more than once; rather than pin to a
        # version that will break, the dataset is written and the exact upload command
        # is reported. Better than silently failing three minutes into a CI run.
        raise NotImplementedError(
            f"training data written to {self.dataset_path}. Upload and tune with:\n"
            f"  firectl create dataset livingeval-judge {self.dataset_path}\n"
            f"  firectl create fine-tuning-job --settings-file <yaml> "
            f"--base-model {self.base_model}\n"
            "then pass the resulting model id to scorer.finetune(backend='fireworks-serve', "
            "model=...) which only runs inference."
        )

    def predict(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - network
        raise NotImplementedError("see fit()")


BACKENDS = {
    "local": LocalFinetuneBackend,
    "unsloth": UnslothBackend,
    "fireworks": FireworksBackend,
}


# ---------------------------------------------------------------------------
# the rung
# ---------------------------------------------------------------------------


class FinetunedRung(Rung):
    """A fine-tuned model, fit under the same group-aware CV as every other rung."""

    name = "finetune"
    n_params = -1
    is_diagnostic = False  # the flag the ladder reads to attach FINETUNE_NOTE

    def __init__(self, backend: str = "local", **kw):
        if isinstance(backend, str):
            try:
                cls = BACKENDS[backend]
            except KeyError:
                raise ValueError(
                    f"unknown finetune backend {backend!r}; choose from {sorted(BACKENDS)}"
                ) from None
            self.backend = cls(**kw)
        else:
            self.backend = backend
        self.label = getattr(self.backend, "label", str(self.backend))
        self.train_seconds = 0.0

    def fit(self, texts, labels):
        t0 = time.perf_counter()
        y = np.asarray(labels, dtype=int)
        self.fallback_ = int(np.bincount(y, minlength=2).argmax())
        if len(np.unique(y)) < 2:
            self.backend = None  # type: ignore[assignment]
        else:
            self.backend.fit(list(texts), y)
        self.train_seconds = time.perf_counter() - t0
        return self

    def predict(self, texts):
        if self.backend is None:
            return np.full(len(texts), self.fallback_, dtype=int)
        return np.asarray(self.backend.predict(list(texts)), dtype=int)

    def describe(self) -> str:
        return f"{self.label}, {self.train_seconds:.1f}s to train"
