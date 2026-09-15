"""Helpers for saving/loading checkpoint metadata and validating eval config."""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch


def build_train_checkpoint(
    model,
    iteration: int,
    args,
    text_embeddings=None,
    ontology_embeddings=None,
    model_module: str = "model",
) -> Dict[str, Any]:
    meta = {
        "model_module": model_module,
        "text_embedding_file": getattr(args, "text_embedding_file", None),
        "knowledge_embedding_file": getattr(args, "knowledge_embedding_file", None),
        "text_emb_dim": int(text_embeddings.shape[1]) if text_embeddings is not None else None,
        "domain_emb_dim": int(ontology_embeddings.shape[1]) if ontology_embeddings is not None else None,
        "n_hidden": getattr(args, "n_hidden", None),
        "num_hidden_layers": getattr(args, "num_hidden_layers", None),
        "num_bases": getattr(args, "num_bases", None),
        "w": getattr(args, "w", None),
        "use_ppr": getattr(args, "use_ppr", None),
    }
    return {
        "state_dict": model.state_dict(),
        "iteration": iteration,
        "meta": meta,
    }


def load_checkpoint_bundle(checkpoint_path: str, map_location="cpu") -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint
    return {"state_dict": checkpoint, "iteration": None, "meta": {}}


def infer_checkpoint_specs(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    state_dict = checkpoint["state_dict"]
    meta = checkpoint.get("meta") or {}
    specs = dict(meta)

    if specs.get("text_emb_dim") is None:
        for key in (
            "rgcn.layers.0.norm_text_embeddings.weight",
            "rgcn.layers.0.text_embeddings.weight",
        ):
            if key in state_dict:
                specs["text_emb_dim"] = int(state_dict[key].shape[1])
                break

    if specs.get("domain_emb_dim") is None:
        for key in (
            "rgcn.layers.0.norm_domain_embeddings.weight",
            "rgcn.layers.0.domain_embeddings.weight",
        ):
            if key in state_dict:
                specs["domain_emb_dim"] = int(state_dict[key].shape[1])
                break
        if specs.get("domain_emb_dim") is None and "rgcn.layers.0.poincare_to_euclidean.weight" in state_dict:
            specs["domain_emb_dim"] = int(
                state_dict["rgcn.layers.0.poincare_to_euclidean.weight"].shape[1]
            )

    if specs.get("n_hidden") is None and "relation_weights" in state_dict:
        specs["n_hidden"] = int(state_dict["relation_weights"].shape[1])

    keys = " ".join(state_dict.keys())
    if specs.get("model_module") is None:
        if "poincare_to_euclidean" in keys or "norm_text_embeddings" in keys:
            specs["model_module"] = "model"
        elif "ppr_branch" in keys or "fusion_gate" in keys:
            specs["model_module"] = "model_base4"
        else:
            specs["model_module"] = "model_base4"

    return specs


def validate_eval_against_checkpoint(
    specs: Dict[str, Any],
    text_embeddings,
    ontology_embeddings,
    text_embedding_file: str,
    knowledge_embedding_file: str,
    n_hidden: int,
) -> None:
    errors = []

    ck_text_dim = specs.get("text_emb_dim")
    if text_embeddings is not None and ck_text_dim is not None:
        file_dim = int(text_embeddings.shape[1])
        if file_dim != ck_text_dim:
            errors.append(
                f"text embedding dim mismatch: file '{text_embedding_file}' has {file_dim}, "
                f"checkpoint expects {ck_text_dim}. "
                f"Dung dung file embedding nhu luc train"
                + (f" (goi y: {specs.get('text_embedding_file')})" if specs.get("text_embedding_file") else "")
            )

    ck_domain_dim = specs.get("domain_emb_dim")
    if ontology_embeddings is not None and ck_domain_dim is not None:
        file_dim = int(ontology_embeddings.shape[1])
        if file_dim != ck_domain_dim:
            errors.append(
                f"domain embedding dim mismatch: file '{knowledge_embedding_file}' has {file_dim}, "
                f"checkpoint expects {ck_domain_dim}. "
                f"Dung dung file embedding nhu luc train"
                + (f" (goi y: {specs.get('knowledge_embedding_file')})" if specs.get("knowledge_embedding_file") else "")
            )

    ck_hidden = specs.get("n_hidden")
    if ck_hidden is not None and int(n_hidden) != int(ck_hidden):
        errors.append(
            f"n_hidden mismatch: args={n_hidden}, checkpoint expects {ck_hidden}"
        )

    if errors:
        raise ValueError("\n".join(errors))
