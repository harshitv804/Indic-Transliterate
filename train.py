from __future__ import annotations

import random
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

import torch
from datasets import load_from_disk
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR

from components import Config, EncoderDecoderTransformer, TrainingConfig
from components.liger_ops import LigerCrossEntropyLoss, use_liger
from csv_logger import CSVLogger, perplexity
from dataloader import IGNORE_INDEX, create_dataloader, load_tokenizer


# --------------------------------------------------------------------------- #
# LR schedule / helpers
# --------------------------------------------------------------------------- #
def Scheduler_WarmUp_Constant(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int | None = None,
    last_epoch: int = -1,
) -> LambdaLR:
    """Linear warmup followed by a constant learning rate."""

    if num_warmup_steps < 0:
        raise ValueError("num_warmup_steps must be >= 0")

    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)

        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def build_optimizer(
    model: nn.Module, tcfg: TrainingConfig, device: torch.device
) -> AdamW:
    """Single AdamW. No weight decay on embeddings, norms, or biases."""
    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "tok_emb" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    print(
        f"[optim] AdamW decay   : {len(decay):>4d} tensors, "
        f"{sum(p.numel() for p in decay) / 1e6:8.1f}M params\n"
        f"[optim] AdamW no-decay: {len(no_decay):>4d} tensors, "
        f"{sum(p.numel() for p in no_decay) / 1e6:8.1f}M params"
    )

    kwargs = {"fused": True} if device.type == "cuda" else {}

    return AdamW(
        [
            {"params": decay, "weight_decay": tcfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=tcfg.lr_adamw,
        betas=tcfg.betas,
        **kwargs,
    )


def autocast_context(device: torch.device, use_bf16: bool):
    if device.type == "cuda" and use_bf16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def build_loss(device: torch.device) -> Callable:
    """Liger fused CE when possible, torch CE otherwise (same call signature)."""
    if use_liger(device):
        print("[loss] LigerCrossEntropyLoss (fused)")
        return LigerCrossEntropyLoss(ignore_index=IGNORE_INDEX)

    reason = "CUDA unavailable" if device.type != "cuda" else "liger_kernel missing"
    print(f"[loss] torch CrossEntropyLoss ({reason})")
    return nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)


# --------------------------------------------------------------------------- #
# Train / eval steps
# --------------------------------------------------------------------------- #
def train_step(
    model: EncoderDecoderTransformer,
    optimizer: AdamW,
    scheduler: LambdaLR | None,
    batch: dict[str, torch.Tensor],
    tcfg: TrainingConfig,
    ce: Callable,
    device: torch.device,
) -> dict[str, float] | None:
    src_ids = batch["src_ids"].to(device, non_blocking=True)
    tgt_ids = batch["tgt_ids"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    src_mask = batch["src_mask"].to(device, non_blocking=True)
    tgt_mask = batch["tgt_mask"].to(device, non_blocking=True)

    if not labels.ne(IGNORE_INDEX).any():
        print("[WARNING] Skipping batch: no valid target tokens.")
        return None

    with autocast_context(device, tcfg.use_bf16):
        logits = model(
            src_ids=src_ids,
            tgt_ids=tgt_ids,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
        )

        # labels are ALREADY shifted by the data pipeline: no logits[:, :-1]
        loss = ce(logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1))

    if not torch.isfinite(loss):
        print(f"[WARNING] Skipping batch: non-finite loss {loss.item()}")
        optimizer.zero_grad(set_to_none=True)
        return None

    lr = optimizer.param_groups[0]["lr"]

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.max_grad_norm)
    optimizer.step()

    if scheduler is not None:
        scheduler.step()

    optimizer.zero_grad(set_to_none=True)

    # Source tokens + supervised target tokens in this batch.
    n_tokens = int(src_mask.sum() + labels.ne(IGNORE_INDEX).sum())

    return {
        "lm_loss": loss.detach().item(),
        "lr": lr,
        "tokens": n_tokens,
    }


def evaluate(
    model: EncoderDecoderTransformer,
    val_loader,
    tcfg: TrainingConfig,
    ce: Callable,
    device: torch.device,
) -> float | None:
    """Token-weighted mean CE over supervised validation tokens."""
    total_loss = 0.0
    total_tokens = 0

    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            for batch in val_loader:
                src_ids = batch["src_ids"].to(device, non_blocking=True)
                tgt_ids = batch["tgt_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                src_mask = batch["src_mask"].to(device, non_blocking=True)
                tgt_mask = batch["tgt_mask"].to(device, non_blocking=True)

                n_tokens = int(labels.ne(IGNORE_INDEX).sum())
                if n_tokens == 0:
                    continue

                with autocast_context(device, tcfg.use_bf16):
                    logits = model(
                        src_ids=src_ids,
                        tgt_ids=tgt_ids,
                        src_mask=src_mask,
                        tgt_mask=tgt_mask,
                    )
                    loss = ce(
                        logits.float().reshape(-1, logits.size(-1)),
                        labels.reshape(-1),
                    )

                total_loss += loss.item() * n_tokens
                total_tokens += n_tokens

    finally:
        if was_training:
            model.train()

    if total_tokens == 0:
        return None

    return total_loss / total_tokens


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
def save_checkpoint(
    path: Path,
    model: EncoderDecoderTransformer,
    optimizer: AdamW,
    step: int,
    epoch: int,
    tokens_seen: int = 0,
) -> None:
    """Atomically write model + optimizer state (tmp file, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(
        {
            # ``step`` batches have completed and ``tokens_seen`` is the
            # cumulative token count across all of them. ``epoch`` is the
            # 1-based epoch of the next batch.
            "step": step,
            "epoch": epoch,
            "epoch_1_based": True,
            "tokens_seen": tokens_seen,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        tmp,
    )
    tmp.replace(path)


def load_checkpoint(
    path: str | Path,
    model: EncoderDecoderTransformer,
    optimizer: AdamW,
    device: torch.device,
    load_optimizer: bool = True,
) -> tuple[int, int, int]:
    """Restore state; return ``(step, epoch, tokens_seen)`` (epoch is 1-based)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    if load_optimizer:
        if "optimizer" not in ckpt:
            raise KeyError(
                "checkpoint has no 'optimizer' key; resume with load_optimizer=False"
            )
        optimizer.load_state_dict(ckpt["optimizer"])
    epoch = int(ckpt["epoch"])
    if not ckpt.get("epoch_1_based", False):
        epoch += 1
    return (
        int(ckpt["step"]),
        epoch,
        int(ckpt.get("tokens_seen", 0)),
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(
    mcfg: Config | None = None,
    tcfg: TrainingConfig | None = None,
) -> None:
    mcfg = mcfg or Config()
    tcfg = tcfg or TrainingConfig()

    print(f"[config] model: {mcfg}")
    print(f"[config] train: {tcfg}")

    torch.set_default_dtype(torch.float32)
    random.seed(tcfg.seed)
    torch.manual_seed(tcfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tcfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    ce = build_loss(device)

    tokenizer = load_tokenizer()

    model = EncoderDecoderTransformer(mcfg)
    if tcfg.resume_from is None:
        model.init_weights()
        print("[init] initialized model weights for a fresh run")
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {n_params / 1e6:.2f}M trainable params")

    optimizer = build_optimizer(model, tcfg, device)

    # ---- resume ---- #
    resume_step, resume_epoch, tokens_seen = 0, 1, 0
    if tcfg.resume_from is not None:
        resume_step, resume_epoch, tokens_seen = load_checkpoint(
            tcfg.resume_from,
            model,
            optimizer,
            device,
            load_optimizer=tcfg.load_optimizer,
        )

    raw = load_from_disk(tcfg.data_dir)
    train_rows = len(raw["train"])
    print(f"[data] train={train_rows:,}  valid={len(raw['validation']):,}")
    if train_rows == 0:
        raise ValueError("training dataset is empty")

    steps_per_epoch = (train_rows + tcfg.batch_size - 1) // tcfg.batch_size

    if tcfg.resume_from is not None:
        print(
            f"[ckpt] resumed step={resume_step} epoch={resume_epoch} "
            f"tokens={tokens_seen:,} from {tcfg.resume_from}"
        )

    if resume_epoch > tcfg.num_epochs:
        print(
            f"[ckpt] checkpoint is already past the last epoch "
            f"({resume_epoch} > {tcfg.num_epochs}); nothing to train"
        )
        return

    total_steps = tcfg.num_training_steps or steps_per_epoch * tcfg.num_epochs
    print(
        f"[loop] {steps_per_epoch:,} steps/epoch, {total_steps:,} total optimizer steps"
    )

    if resume_step >= total_steps:
        print(
            f"[ckpt] checkpoint is already at total_steps "
            f"({resume_step} >= {total_steps}); nothing to train"
        )
        return

    # ---- LR schedule ---- #
    if tcfg.use_lr_scheduler:
        num_warmup_steps = min(tcfg.num_warmup_steps, total_steps)
        scheduler = Scheduler_WarmUp_Constant(optimizer, num_warmup_steps, total_steps)
        scheduler.last_epoch = resume_step
        for group, base_lr, lr_lambda in zip(
            optimizer.param_groups, scheduler.base_lrs, scheduler.lr_lambdas
        ):
            group["lr"] = base_lr * lr_lambda(scheduler.last_epoch)
        print(
            f"[sched] warmup + constant: {num_warmup_steps} warmup / "
            f"{total_steps} total; peak lr={tcfg.lr_adamw:.3e}"
        )
    else:
        scheduler = None
        for group in optimizer.param_groups:
            group["lr"] = tcfg.lr_adamw
        print(f"[sched] disabled; fixed lr={tcfg.lr_adamw:.3e}")

    # ---- data loaders ---- #
    train_loader = create_dataloader(
        raw["train"],
        tokenizer,
        batch_size=tcfg.batch_size,
        shuffle=tcfg.shuffle,
        num_workers=tcfg.num_workers,
        pin_memory=device.type == "cuda",
        max_src_len=tcfg.max_src_len,
        max_tgt_len=tcfg.max_tgt_len,
    )
    val_loader = create_dataloader(
        raw["validation"],
        tokenizer,
        batch_size=tcfg.eval_batch_size,
        shuffle=False,
        num_workers=tcfg.num_workers,
        pin_memory=device.type == "cuda",
        max_src_len=tcfg.max_src_len,
        max_tgt_len=tcfg.max_tgt_len,
    )

    csv_logger = CSVLogger(tcfg.log_dir, tcfg.run_id)
    ckpt_dir = Path(tcfg.checkpoint_dir)

    model.train()

    step = resume_step
    try:
        for epoch in range(resume_epoch, tcfg.num_epochs + 1):
            for batch in train_loader:
                logs = train_step(model, optimizer, scheduler, batch, tcfg, ce, device)
                if logs is None:
                    continue

                # ``step`` is the 1-based number of completed optimizer updates.
                step += 1
                tokens_seen += logs["tokens"]

                if step % tcfg.log_every == 0:
                    print(
                        f"[train] epoch: {epoch} | step: {step} | "
                        f"tokens: {tokens_seen:,} | "
                        f"lm_loss: {logs['lm_loss']:.4f} | "
                        f"ppl: {perplexity(logs['lm_loss']):.2f} | "
                        f"lr: {logs['lr']:.3e}"
                    )

                # Epoch for the next batch after this update (1-based). At an
                # epoch boundary this is already the next epoch.
                next_epoch = step // steps_per_epoch + 1

                if step % tcfg.checkpoint_every == 0:
                    val_loss = evaluate(model, val_loader, tcfg, ce, device)
                    if val_loss is not None:
                        print(
                            f"[val] step: {step} | "
                            f"val_lm_loss: {val_loss:.4f} | "
                            f"val_ppl: {perplexity(val_loss):.2f}"
                        )

                    csv_logger.log(epoch, step, logs, val_loss)

                    ckpt_path = ckpt_dir / f"step_{step}.pt"
                    save_checkpoint(
                        ckpt_path,
                        model,
                        optimizer,
                        step,
                        next_epoch,
                        tokens_seen,
                    )
                    print(f"[ckpt] saved step={step} -> {ckpt_path}")

                if step >= total_steps:
                    print(f"reached total_steps={total_steps}, stopping")
                    final_path = ckpt_dir / f"step_{step}_epoch{epoch}_final.pt"
                    save_checkpoint(
                        final_path,
                        model,
                        optimizer,
                        step,
                        next_epoch,
                        tokens_seen,
                    )
                    print(f"[ckpt] saved final step={step} -> {final_path}")
                    return

            # final checkpoint at the end of training
            final_path = ckpt_dir / f"step_{step}_epoch{epoch}_final.pt"
            save_checkpoint(
                final_path,
                model,
                optimizer,
                step,
                step // steps_per_epoch + 1,
                tokens_seen,
            )
            print(f"[ckpt] saved final step={step} -> {final_path}")

    finally:
        csv_logger.close()


if __name__ == "__main__":
    main()
