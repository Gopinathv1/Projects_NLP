"""Batched inference over features + span and ranking metrics."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import default_data_collator

from .config import QAConfig
from .metrics import confidence_calibration, ranking_metrics, squad_metrics
from .postprocessing import postprocess_ranked_predictions


@torch.no_grad()
def collect_logits(model, features, device, batch_size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """Run the model over validation features and return start/end logits."""
    tensor_features = features.remove_columns(
        [c for c in features.column_names
         if c not in ("input_ids", "attention_mask", "token_type_ids")]
    )
    tensor_features.set_format("torch")
    loader = DataLoader(tensor_features, batch_size=batch_size, collate_fn=default_data_collator)

    model.eval().to(device)
    starts, ends = [], []
    for batch in tqdm(loader, desc="inference", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        starts.append(out.start_logits.float().cpu().numpy())
        ends.append(out.end_logits.float().cpu().numpy())
    return np.concatenate(starts, axis=0), np.concatenate(ends, axis=0)


def references_from_examples(examples) -> Dict[str, List[str]]:
    return {ex_id: list(ans["text"]) for ex_id, ans in zip(examples["id"], examples["answers"])}


def evaluate_model(
    model,
    examples,
    features,
    cfg: QAConfig,
    device=None,
    k_values: Optional[Tuple[int, ...]] = (1, 3, 5),
    top_k: Optional[int] = None,
) -> Dict:
    """End to end: logits -> ranked candidates -> span metrics + ranking metrics."""
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    start_logits, end_logits = collect_logits(model, features, device, cfg.eval_batch_size)

    top_k = top_k or max(k_values or (cfg.top_k,))
    ranked = postprocess_ranked_predictions(examples, features, start_logits, end_logits,
                                            cfg, top_k=top_k)
    refs = references_from_examples(examples)

    top1 = {ex_id: (c[0]["text"] if c and not c[0].get("is_no_answer") else "")
            for ex_id, c in ranked.items()}
    texts = {ex_id: [c["text"] for c in cands if not c.get("is_no_answer")]
             for ex_id, cands in ranked.items()}

    return {
        "span_metrics": squad_metrics(top1, refs),
        "ranking_metrics": ranking_metrics(texts, refs, k_values=k_values or (1,)),
        "calibration": confidence_calibration(ranked, refs),
        "ranked_predictions": ranked,
        "references": refs,
    }
