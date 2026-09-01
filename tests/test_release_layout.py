from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_freezes_final_submission_parameters():
    source = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "AUTOPET_FDG_PROBABILITY_THRESHOLD=0.47" in source
    assert "AUTOPET_PSMA_PRUNE_THRESHOLD=0.86" in source
    assert "AUTOPET_FDG_RELAX_BURDEN_COMPONENTS=25" in source
    assert "candidate_runtime.process_rank2_final_interaction" in source


def test_release_contains_required_runtime_and_weight_metadata():
    required = [
        "candidate_runtime/process_rank2_final_interaction.py",
        "candidate_runtime/psma_champion_pruner.py",
        "candidate_runtime/edt_stateless_fusion.py",
        "edt_runner.py",
        "weights/edt_model/plans.json",
        "weights/edt_model/fold_0/checkpoint_final.sha256",
        "LICENSE",
        "NOTICE",
    ]
    assert all((ROOT / relative).is_file() for relative in required)
