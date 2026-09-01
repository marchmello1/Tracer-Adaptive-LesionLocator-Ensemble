"""Fail-closed primitives for the next-generation click-local edit model.

The model predicts three mutually exclusive voxel classes: KEEP, ADD and
REMOVE.  A deterministic controller enforces the semantics that the network
cannot learn around: ADD is legal only outside immutable champion M0 and near
a positive scribble; REMOVE is legal only inside M0 and near a negative
scribble; everything else is KEEP.
"""

from __future__ import annotations

from dataclasses import dataclass

import cc3d
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


KEEP, ADD, REMOVE = 0, 1, 2


def build_local_support(click_cores: np.ndarray, radius_voxels: float) -> np.ndarray:
    """Dilate two scribble-core channels into deterministic local support tubes."""
    click_cores = np.asarray(click_cores, dtype=bool)
    if click_cores.ndim != 4 or click_cores.shape[0] != 2:
        raise ValueError(f"expected [2,D,H,W] click cores, got {click_cores.shape}")
    if radius_voxels <= 0:
        raise ValueError(f"radius_voxels must be positive, got {radius_voxels}")
    support = np.zeros_like(click_cores, dtype=bool)
    for channel in range(2):
        if np.any(click_cores[channel]):
            support[channel] = distance_transform_edt(~click_cores[channel]) <= radius_voxels
    return support


def build_local_edit_target(
    ground_truth: np.ndarray,
    champion: np.ndarray,
    touched_actions: np.ndarray,
    local_support: np.ndarray,
) -> np.ndarray:
    """Build a KEEP/ADD/REMOVE class map clipped to the click-local support."""
    ground_truth = np.asarray(ground_truth, dtype=bool)
    champion = np.asarray(champion, dtype=bool)
    touched_actions = np.asarray(touched_actions, dtype=bool)
    local_support = np.asarray(local_support, dtype=bool)
    if ground_truth.shape != champion.shape:
        raise ValueError("ground_truth and champion shapes differ")
    if touched_actions.shape != (2, *ground_truth.shape):
        raise ValueError(f"invalid touched_actions shape {touched_actions.shape}")
    if local_support.shape != touched_actions.shape:
        raise ValueError("local_support and touched_actions shapes differ")
    add = touched_actions[0] & local_support[0] & ground_truth & ~champion
    remove = touched_actions[1] & local_support[1] & champion & ~ground_truth
    if np.any(add & remove):
        raise RuntimeError("ADD and REMOVE targets overlap")
    target = np.full(ground_truth.shape, KEEP, dtype=np.int64)
    target[add] = ADD
    target[remove] = REMOVE
    return target


def apply_legal_local_logits(
    logits: torch.Tensor,
    champion: torch.Tensor,
    positive_support: torch.Tensor,
    negative_support: torch.Tensor,
    invalid_logit: float = -30.0,
) -> torch.Tensor:
    """Mask impossible action logits while leaving KEEP available everywhere."""
    if logits.ndim != 5 or logits.shape[1] != 3:
        raise ValueError(f"expected logits [B,3,D,H,W], got {tuple(logits.shape)}")
    champion = champion.bool()
    positive_support = positive_support.bool()
    negative_support = negative_support.bool()
    for name, tensor in (
        ("champion", champion),
        ("positive_support", positive_support),
        ("negative_support", negative_support),
    ):
        if tensor.shape != logits[:, :1].shape:
            raise ValueError(f"{name} shape {tuple(tensor.shape)} does not match logits")
    legal_add = positive_support & ~champion
    legal_remove = negative_support & champion
    constrained = logits.clone()
    constrained[:, 1:2] = torch.where(
        legal_add, constrained[:, 1:2], constrained.new_full((), invalid_logit)
    )
    constrained[:, 2:3] = torch.where(
        legal_remove, constrained[:, 2:3], constrained.new_full((), invalid_logit)
    )
    return constrained


def soft_reconstruct(champion: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
    """Differentiable M1 = M0 + ADD - REMOVE reconstruction."""
    champion = champion.float()
    return champion * (1.0 - probabilities[:, 2:3]) + (1.0 - champion) * probabilities[:, 1:2]


def soft_dice_per_sample(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.float().flatten(1)
    target = target.float().flatten(1)
    numerator = 2.0 * (prediction * target).sum(1) + 1e-5
    denominator = prediction.sum(1) + target.sum(1) + 1e-5
    return numerator / denominator


def local_edit_objective(
    logits: torch.Tensor,
    target_classes: torch.Tensor,
    champion: torch.Tensor,
    ground_truth: torch.Tensor,
    positive_support: torch.Tensor,
    negative_support: torch.Tensor,
    nondegradation_weight: float = 3.0,
    wrong_edit_weight: float = 2.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Optimize local action precision, reconstructed Dice and non-degradation."""
    constrained = apply_legal_local_logits(
        logits, champion, positive_support, negative_support
    )
    probabilities = torch.softmax(constrained.float(), dim=1)
    if target_classes.ndim == 5 and target_classes.shape[1] == 1:
        target_classes = target_classes[:, 0]
    if target_classes.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError(
            f"target shape {tuple(target_classes.shape)} incompatible with {tuple(logits.shape)}"
        )
    target_classes = target_classes.long()
    # Augmentation/interpolation represents support maps as float tensors even
    # though their semantics are binary. Normalize here so the objective is
    # robust to both the raw dataloader and deep-supervision paths.
    local_union = (positive_support.bool() | negative_support.bool())[:, 0]
    voxel_ce = F.cross_entropy(
        constrained, target_classes, reduction="none",
        weight=logits.new_tensor([0.5, 2.0, 2.0]),
    )
    ce = voxel_ce[local_union].mean() if torch.any(local_union) else voxel_ce.sum() * 0

    action_dice_losses = []
    for action_class in (ADD, REMOVE):
        target = target_classes == action_class
        active = target.flatten(1).any(1)
        if torch.any(active):
            action_dice_losses.append(
                1.0 - soft_dice_per_sample(
                    probabilities[active, action_class:action_class + 1],
                    target[active, None],
                ).mean()
            )
    action_dice = (
        torch.stack(action_dice_losses).mean()
        if action_dice_losses else logits.sum() * 0
    )

    reconstructed = soft_reconstruct(champion, probabilities)
    final_dice = soft_dice_per_sample(reconstructed, ground_truth)
    base_dice = soft_dice_per_sample(champion, ground_truth)
    reconstruction_loss = 1.0 - final_dice.mean()
    nondegradation = F.relu(base_dice.detach() - final_dice).mean()

    legal_add = positive_support.bool() & ~champion.bool()
    legal_remove = negative_support.bool() & champion.bool()
    wrong_add = legal_add[:, 0] & (target_classes != ADD)
    wrong_remove = legal_remove[:, 0] & (target_classes != REMOVE)
    wrong_terms = []
    if torch.any(wrong_add):
        wrong_terms.append(probabilities[:, ADD][wrong_add].mean())
    if torch.any(wrong_remove):
        wrong_terms.append(probabilities[:, REMOVE][wrong_remove].mean())
    wrong_edit = torch.stack(wrong_terms).mean() if wrong_terms else logits.sum() * 0

    loss = (
        ce + action_dice + reconstruction_loss
        + nondegradation_weight * nondegradation
        + wrong_edit_weight * wrong_edit
    )
    diagnostics = {
        "ce": ce.detach(),
        "action_dice_loss": action_dice.detach(),
        "reconstruction_loss": reconstruction_loss.detach(),
        "nondegradation_hinge": nondegradation.detach(),
        "wrong_edit": wrong_edit.detach(),
        "base_dice": base_dice.mean().detach(),
        "final_dice": final_dice.mean().detach(),
    }
    return loss, diagnostics


@dataclass(frozen=True)
class LocalEditDecision:
    add: np.ndarray
    remove: np.ndarray
    rejected_oversize_components: int


def _seeded_components(
    candidate: np.ndarray,
    seed: np.ndarray,
    max_action_voxels: int,
) -> tuple[np.ndarray, int]:
    accepted = np.zeros_like(candidate, dtype=bool)
    if not np.any(candidate) or not np.any(seed):
        return accepted, 0
    labels = cc3d.connected_components(candidate.astype(np.uint8, copy=False), connectivity=26)
    component_ids = np.unique(labels[seed])
    component_ids = component_ids[component_ids != 0]
    rejected = 0
    for component_id in component_ids:
        component = labels == component_id
        if int(component.sum()) > max_action_voxels:
            rejected += 1
            continue
        accepted |= component
    return accepted, rejected


def decide_local_edits(
    logits: torch.Tensor,
    champion: torch.Tensor,
    click_cores: torch.Tensor,
    local_support: torch.Tensor,
    probability_threshold: float,
    keep_margin: float,
    max_action_voxels: int,
) -> list[LocalEditDecision]:
    """Fail closed unless a legal, seeded, bounded action defeats KEEP."""
    if click_cores.shape[1] != 2 or local_support.shape[1] != 2:
        raise ValueError("click_cores and local_support must have two action channels")
    constrained = apply_legal_local_logits(
        logits, champion, local_support[:, 0:1], local_support[:, 1:2]
    )
    probabilities = torch.softmax(constrained.float(), dim=1).detach().cpu().numpy()
    champion_np = champion.bool().detach().cpu().numpy()[:, 0]
    cores_np = click_cores.bool().detach().cpu().numpy()
    support_np = local_support.bool().detach().cpu().numpy()
    decisions = []
    for batch_index in range(logits.shape[0]):
        keep = probabilities[batch_index, KEEP]
        add_candidate = (
            (probabilities[batch_index, ADD] >= probability_threshold)
            & (probabilities[batch_index, ADD] >= keep + keep_margin)
            & support_np[batch_index, 0]
            & ~champion_np[batch_index]
        )
        remove_candidate = (
            (probabilities[batch_index, REMOVE] >= probability_threshold)
            & (probabilities[batch_index, REMOVE] >= keep + keep_margin)
            & support_np[batch_index, 1]
            & champion_np[batch_index]
        )
        add, rejected_add = _seeded_components(
            add_candidate, cores_np[batch_index, 0], max_action_voxels
        )
        remove, rejected_remove = _seeded_components(
            remove_candidate, cores_np[batch_index, 1], max_action_voxels
        )
        decisions.append(LocalEditDecision(
            add=add,
            remove=remove,
            rejected_oversize_components=rejected_add + rejected_remove,
        ))
    return decisions
