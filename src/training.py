"""Training utilities for the released self-only single-modality predictors."""

import copy
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class AdditiveAttentionPooling(nn.Module):
    """Additive attention over an utterance sequence: [B, T, d] -> [B, d]."""

    def __init__(self, d: int):
        super().__init__()
        self.w = nn.Linear(d, 1, bias=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.w(x).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        alpha = F.softmax(scores, dim=-1)
        return (alpha.unsqueeze(-1) * x).sum(dim=1)


class PredictionHead(nn.Module):
    """Linear projection, ReLU, dropout, and scalar regression output."""

    def __init__(
        self,
        d: int,
        dropout: float = 0.2,
        hidden: int | None = None,
    ):
        super().__init__()
        hidden = hidden if hidden is not None else d // 5
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SingleModalModel(nn.Module):
    """Pool only the rating-side participant's utterances, then regress."""

    def __init__(self, d: int, dropout: float = 0.2):
        super().__init__()
        self.attn = AdditiveAttentionPooling(d)
        self.head = PredictionHead(d, dropout, hidden=max(d // 5, 16))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.attn(x, mask))


def _ccc_tensor(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """CCC for one-dimensional tensors containing at least two elements."""
    mu_a = y_true.mean()
    mu_p = y_pred.mean()
    var_a = ((y_true - mu_a) ** 2).mean()
    var_p = ((y_pred - mu_p) ** 2).mean()
    cov = ((y_true - mu_a) * (y_pred - mu_p)).mean()
    denom = var_a + var_p + (mu_a - mu_p) ** 2
    return 2.0 * cov / (denom + 1e-8)


def _ccc_loss_from_scores(
    scores: torch.Tensor,
    targets: torch.Tensor,
    participant_ids: list,
) -> torch.Tensor:
    unique_ids = list(dict.fromkeys(participant_ids))
    cccs = []
    for pid in unique_ids:
        idx = [i for i, value in enumerate(participant_ids) if value == pid]
        if len(idx) < 2:
            continue
        cccs.append(_ccc_tensor(targets[idx], scores[idx]))
    if not cccs:
        raise ValueError(
            "CCC loss requires at least one participant with two or more "
            "samples in every training batch"
        )
    return 1.0 - torch.stack(cccs).mean()


def _participant_id(sample: dict) -> str:
    if sample["speaker"] == "female":
        return sample["female_id"]
    return sample["male_id"]


def make_participant_batches(
    samples: list,
    batch_size: int = 6,
    seed: int | None = None,
):
    grouped: dict[str, list] = defaultdict(list)
    for sample in samples:
        grouped[_participant_id(sample)].append(sample)
    participants = list(grouped)
    random.Random(seed).shuffle(participants)
    for start in range(0, len(participants), batch_size):
        batch = []
        for pid in participants[start:start + batch_size]:
            batch.extend(grouped[pid])
        yield batch


def _collate_single_batch(
    batch: list,
    device: torch.device,
    embed_key: str = "embeddings",
):
    seqs = [sample[embed_key] for sample in batch]
    padded = pad_sequence(seqs, batch_first=True).to(device).float()
    lengths = torch.tensor([tensor.shape[0] for tensor in seqs], device=device)
    max_length = padded.shape[1]
    mask = (
        torch.arange(max_length, device=device).unsqueeze(0)
        < lengths.unsqueeze(1)
    )
    targets = torch.tensor(
        [sample["_target_scaled"] for sample in batch],
        dtype=torch.float32,
        device=device,
    )
    participant_ids = [_participant_id(sample) for sample in batch]
    return padded, mask, targets, participant_ids


def train_one_epoch_single(
    model,
    train_samples: list,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: str,
    batch_size: int = 6,
    embed_key: str = "embeddings",
) -> float:
    """Run one training epoch for a self-only single-modality model."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    epoch_seed = random.randint(0, 2**31)

    if loss_fn != "ccc":
        raise ValueError(
            f"Unsupported loss_fn={loss_fn!r}; the released paper path uses "
            "participant-macro CCC only"
        )
    batches = make_participant_batches(
        train_samples, batch_size, seed=epoch_seed,
    )

    for batch in batches:
        x, mask, targets, participant_ids = _collate_single_batch(
            batch, device, embed_key,
        )
        optimizer.zero_grad()
        scores = model(x, mask)
        loss = _ccc_loss_from_scores(
            scores, targets, participant_ids,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


def evaluate_val_single(
    model,
    val_samples: list,
    device: torch.device,
    embed_key: str = "embeddings",
) -> float:
    """Compute participant-macro validation CCC."""
    model.eval()
    participant_ids = []
    y_true = []
    y_pred = []

    with torch.no_grad():
        for sample in val_samples:
            x = sample[embed_key].unsqueeze(0).to(device).float()
            mask = torch.ones(
                1, x.shape[1], dtype=torch.bool, device=device,
            )
            score = model(x, mask)
            y_pred.append(score.item())
            y_true.append(sample["_target_scaled"])
            participant_ids.append(_participant_id(sample))

    grouped: dict[str, tuple] = defaultdict(lambda: ([], []))
    for pid, truth, prediction in zip(participant_ids, y_true, y_pred):
        grouped[pid][0].append(truth)
        grouped[pid][1].append(prediction)

    from evaluation_metrics import ccc as eval_ccc

    cccs = []
    for truths, predictions in grouped.values():
        if len(truths) < 2:
            continue
        cccs.append(eval_ccc(np.array(truths), np.array(predictions)))
    return float(np.mean(cccs)) if cccs else 0.0


def fit_single(
    model,
    train_samples: list,
    val_samples: list,
    device: torch.device,
    loss_fn: str,
    lr: float = 0.001,
    patience: int = 20,
    batch_size: int = 6,
    max_epochs: int = 500,
    min_epochs: int = 20,
    verbose: bool = True,
    embed_key: str = "embeddings",
    save_path: str | None = None,
):
    """Train with early stopping based on participant-macro validation CCC."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_ccc = -float("inf")
    best_state = None
    patience_count = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch_single(
            model,
            train_samples,
            optimizer,
            device,
            loss_fn,
            batch_size,
            embed_key,
        )
        val_ccc = evaluate_val_single(
            model, val_samples, device, embed_key,
        )

        if epoch >= min_epochs:
            if val_ccc > best_val_ccc:
                best_val_ccc = val_ccc
                best_state = copy.deepcopy(model.state_dict())
                patience_count = 0
            else:
                patience_count += 1

        if verbose:
            flag = " *" if patience_count == 0 else ""
            print(
                f"  epoch {epoch:4d}  train_loss={train_loss:.4f}  "
                f"val_ccc={val_ccc:.4f}{flag}"
            )

        if epoch >= min_epochs and patience_count >= patience:
            if verbose:
                print(
                    f"  Early stopping at epoch {epoch}  "
                    f"(best val CCC={best_val_ccc:.4f})"
                )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        if save_path is not None:
            torch.save(best_state, save_path)
            if verbose:
                print(f"  Model saved to: {save_path}")
    return model
