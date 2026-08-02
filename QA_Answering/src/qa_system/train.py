"""Fine-tuning loop in native PyTorch (Requirement 4.3).

A plain loop is used instead of `Trainer` on purpose: the argument names of
`TrainingArguments` have churned across transformers releases, and an explicit
loop makes the optimisation choices (warm-up, clipping, AMP, accumulation)
visible and reviewable.
"""

from __future__ import annotations

import math
import os
import random
import time
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import default_data_collator, get_linear_schedule_with_warmup

from .config import QAConfig
from .evaluation import evaluate_model
from .modeling import get_device, save_artifacts


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model, cfg: QAConfig):
    """AdamW with no weight decay on biases and LayerNorm."""
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return torch.optim.AdamW(groups, lr=cfg.learning_rate)


def train_model(
    model,
    tokenizer,
    train_features,
    cfg: QAConfig,
    eval_examples=None,
    eval_features=None,
    device: Optional[torch.device] = None,
) -> Dict:
    """Fine-tune and return a training history dict."""
    set_seed(cfg.seed)
    device = device or get_device()
    model.to(device)

    keep = ("input_ids", "attention_mask", "token_type_ids",
            "start_positions", "end_positions")
    ds = train_features.remove_columns(
        [c for c in train_features.column_names if c not in keep]
    )
    ds.set_format("torch")
    loader = DataLoader(
        ds,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        collate_fn=default_data_collator,
        drop_last=False,
    )

    steps_per_epoch = math.ceil(len(loader) / cfg.gradient_accumulation_steps)
    total_steps = steps_per_epoch * cfg.num_train_epochs

    optimizer = build_optimizer(model, cfg)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )
    use_amp = cfg.fp16 and device.type == "cuda"
    try:                                   # torch >= 2.4
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):    # older torch
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = {"loss": [], "epoch_eval": [], "config": cfg.__dict__.copy()}
    global_step = 0
    t0 = time.time()

    for epoch in range(cfg.num_train_epochs):
        model.train()
        running, seen = 0.0, 0
        bar = tqdm(loader, desc=f"epoch {epoch + 1}/{cfg.num_train_epochs}")

        for step, batch in enumerate(bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss / cfg.gradient_accumulation_steps

            scaler.scale(loss).backward()
            running += loss.item() * cfg.gradient_accumulation_steps
            seen += 1

            if (step + 1) % cfg.gradient_accumulation_steps == 0 or (step + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % cfg.logging_steps == 0:
                    avg = running / max(seen, 1)
                    history["loss"].append({"step": global_step, "loss": round(avg, 4),
                                            "lr": scheduler.get_last_lr()[0]})
                    bar.set_postfix(loss=f"{avg:.4f}")
                    running, seen = 0.0, 0

        if eval_examples is not None and eval_features is not None:
            report = evaluate_model(model, eval_examples, eval_features, cfg, device)
            span = report["span_metrics"]
            rank = report["ranking_metrics"]
            history["epoch_eval"].append({"epoch": epoch + 1, **span, **rank})
            print(f"[epoch {epoch + 1}] EM={span['exact_match']} F1={span['f1']} "
                  f"hit@1={rank.get('hit@1')} hit@5={rank.get('hit@5')} "
                  f"MRR={rank.get('MRR_exact')}")

    history["train_seconds"] = round(time.time() - t0, 1)
    save_artifacts(model, tokenizer, cfg)
    print(f"Model saved to: {os.path.abspath(cfg.output_dir)}")
    return history
