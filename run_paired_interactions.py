#!/usr/bin/env python3
"""Run the released container through a six-step simulated interaction case."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def next_scribble(prediction_zyx, label_zyx, cumulative, simulate):
    prediction_xyz = prediction_zyx.transpose(2, 1, 0) > 0
    label_xyz = label_zyx.transpose(2, 1, 0) > 0
    over = prediction_xyz & ~label_xyz
    under = ~prediction_xyz & label_xyz
    background = simulate(over, "centerline")
    foreground = simulate(under, "centerline")
    background_points, background_size = (
        (background[0], int(background[-1])) if len(background) == 3 else ([], 0)
    )
    foreground_points, foreground_size = (
        (foreground[0], int(foreground[-1])) if len(foreground) == 3 else ([], 0)
    )
    if background_size <= foreground_size:
        cumulative["tumor"].extend(foreground_points)
        return "tumor", len(foreground_points)
    cumulative["background"].extend(background_points)
    return "background", len(background_points)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--label", type=Path, required=True)
    parser.add_argument("--support-dir", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--run-step-zero", action="store_true")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--psma-max-components", type=int)
    parser.add_argument("--podman-root", type=Path)
    parser.add_argument("--podman-runroot", type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.support_dir))
    from metrics import MetricEvaluator
    from simulate_scribbles import simulate_scribble_from_label, scribbles_to_gc_format

    output = args.case_dir / "output/images/tumor-lesion-segmentation/case.mha"
    clicks_path = args.case_dir / "input/lesion-clicks.json"
    label = sitk.GetArrayFromImage(sitk.ReadImage(str(args.label))) > 0
    cumulative = {"tumor": [], "background": []}
    records = []

    for step in range(args.steps):
        if step:
            prior = sitk.GetArrayFromImage(sitk.ReadImage(str(output))) > 0
            polarity, new_points = next_scribble(
                prior, label, cumulative, simulate_scribble_from_label
            )
            clicks_path.write_text(json.dumps(scribbles_to_gc_format(cumulative)))
        else:
            polarity, new_points = "none", 0
            if args.run_step_zero:
                clicks_path.write_text(json.dumps(scribbles_to_gc_format(cumulative)))

        started = time.monotonic()
        if step or args.run_step_zero:
            environment = os.environ.copy()
            if args.podman_root:
                if not args.podman_runroot:
                    parser.error("--podman-runroot is required with --podman-root")
                command = [
                    "podman", "--root", str(args.podman_root),
                    "--runroot", str(args.podman_runroot), "run", "--rm",
                    "--runtime=/usr/bin/nvidia-container-runtime",
                    "-e", f"NVIDIA_VISIBLE_DEVICES={args.gpu}",
                    "--network=none", "--cap-drop=ALL",
                    "--security-opt=no-new-privileges", "--memory=31g",
                    "--memory-swap=31g", "--shm-size=2g",
                ]
            else:
                command = [
                    "docker", "run", "--rm", "--network=none", "--cap-drop=ALL",
                    "--security-opt=no-new-privileges", "--memory=31g", "--memory-swap=31g",
                    "--shm-size=2g", "--gpus", f"device={args.gpu}",
                ]
            command.extend(
                (
                    [
                        "-e",
                        f"AUTOPET_PSMA_INTERACTION_MAX_COMPONENTS={args.psma_max_components}",
                    ]
                    if args.psma_max_components is not None
                    else []
                )
                + [
                    "-v", f"{args.case_dir / 'input'}:/input:ro",
                    "-v", f"{args.case_dir / 'output'}:/output",
                    "-v", f"{args.model_root}:/opt/ml/model:ro",
                    args.image,
                ]
            )
            log_path = args.case_dir / f"container_step_{step}.log"
            with log_path.open("w") as log:
                subprocess.run(
                    command,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        seconds = time.monotonic() - started

        prediction = sitk.GetArrayFromImage(sitk.ReadImage(str(output))) > 0
        shutil.copy2(output, args.case_dir / f"prediction_step_{step}.mha")
        evaluator = MetricEvaluator(overlap_threshold=0.1, connectivity=18)
        metric = evaluator(prediction, label, f"step_{step}")
        record = {
            "step": step,
            "seconds": seconds,
            "dice": float(metric["dsc"]),
            "f1": float(metric["f1"]),
            "tp": int(metric["tp"]),
            "fp": int(metric["fp"]),
            "fn": int(metric["fn"]),
            "polarity": polarity,
            "new_points": new_points,
            "foreground_points": len(cumulative["tumor"]),
            "background_points": len(cumulative["background"]),
        }
        records.append(record)
        print(json.dumps(record), flush=True)

    x = np.arange(args.steps)
    dice = np.asarray([record["dice"] for record in records])
    f1 = np.asarray([record["f1"] for record in records])
    summary = {
        "normalized_auc_dice": float(np.trapz(dice, x) / (args.steps - 1)),
        "normalized_auc_f1": float(np.trapz(f1, x) / (args.steps - 1)),
        "total_interaction_seconds": float(sum(r["seconds"] for r in records[1:])),
        "records": records,
    }
    (args.case_dir / "interaction_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
