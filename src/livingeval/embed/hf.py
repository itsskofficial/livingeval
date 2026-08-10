"""Local neural embeddings: Hugging Face encoders and sentence-transformers.

Two backends because they answer different constraints.

`HFEmbedder` uses `transformers` directly and mean-pools the last hidden state with the
attention mask applied. It works with any encoder checkpoint and needs only
`transformers` + `torch` — no `sentence-transformers`, no `accelerate`, no `datasets`.
That matters because those three drag in a lot and are often the reason an install
fails on someone's machine.

`SentenceTransformerEmbedder` uses `sentence-transformers` when it is installed, which
is preferable when available: it applies the model's *own* pooling and normalisation
configuration rather than assuming mean-pooling. For models trained with CLS pooling,
mean-pooling is simply the wrong read, and the difference is not subtle.

Both are wrapped by `CachedEmbedder`, so a corpus is encoded once.

**Masked mean-pooling, not `[:, 0]` and not an unmasked mean.** Taking the first token
gives you `[CLS]`, which is only meaningful for models trained to use it. Averaging
without the mask averages in the padding, so the vector you get depends on the longest
text in the batch — a bug that produces plausible numbers and moves them when you
change `batch_size`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["HFEmbedder", "SentenceTransformerEmbedder"]


class HFEmbedder:
    """Mean-pooled last hidden state from any Hugging Face encoder."""

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        max_length: int = 512,
        revision: str = "main",
        normalize: bool = True,
    ):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:  # pragma: no cover - optional extra
            raise ImportError("pip install 'livingeval[embed]'") from e

        self._torch = torch
        self.model_name = model
        self.revision = revision
        self.max_length = max_length
        self.normalize = normalize
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.model = AutoModel.from_pretrained(model, revision=revision).to(self.device).eval()
        self.dim = int(self.model.config.hidden_size)
        # The revision is in the name because Hugging Face repositories are mutable,
        # and a coverage number keyed on "main" is not reproducible.
        self.name = f"hf:{model}@{revision}(dim={self.dim},pool=mean)"

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        batch = self.tokenizer(
            list(texts), padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            hidden = self.model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts
        if self.normalize:
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy().astype(np.float32)


class SentenceTransformerEmbedder:
    """`sentence-transformers`, which applies the model's own pooling config."""

    def __init__(self, model: str = "all-MiniLM-L6-v2", device: str | None = None,
                 normalize: bool = True):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - optional extra
            raise ImportError(
                "pip install sentence-transformers  (or use hf:<model>, which needs "
                "only transformers + torch)"
            ) from e
        self.model = SentenceTransformer(model, device=device)
        self.normalize = normalize
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.name = f"st:{model}(dim={self.dim})"

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                list(texts), normalize_embeddings=self.normalize,
                show_progress_bar=False, convert_to_numpy=True,
            ),
            dtype=np.float32,
        )
