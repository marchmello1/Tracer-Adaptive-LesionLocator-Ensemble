import json

import numpy as np
import SimpleITK as sitk

import candidate_runtime.process_rank2_final_interaction as runtime
import public_tracer_router
from candidate_runtime.challenge_io import has_effective_scribbles
from candidate_runtime.initial_prediction import InitialPrediction


def test_empty_clicks_keep_exact_rank2_k0() -> None:
    assert not has_effective_scribbles({"tumor": [], "background": []})


def test_any_valid_click_activates_interaction() -> None:
    assert has_effective_scribbles({"tumor": [[1, 2, 3]], "background": []})
    assert has_effective_scribbles({"tumor": [], "background": [[1, 2, 3]]})


def test_malformed_clicks_do_not_activate_interaction() -> None:
    assert not has_effective_scribbles(
        {"tumor": [None, [1, 2], "bad"], "background": [{"x": 1}]}
    )


def test_runtime_orchestrates_request_and_preserves_output_geometry(
    monkeypatch, tmp_path
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    ct_dir = input_root / "images" / "ct"
    pet_dir = input_root / "images" / "pet"
    ct_dir.mkdir(parents=True)
    pet_dir.mkdir(parents=True)
    reference = sitk.GetImageFromArray(np.zeros((3, 4, 5), dtype=np.float32))
    reference.SetSpacing((2.0, 2.5, 3.0))
    sitk.WriteImage(reference, str(ct_dir / "subject.mha"))
    sitk.WriteImage(reference, str(pet_dir / "subject.mha"))
    (input_root / "lesion-clicks.json").write_text(
        json.dumps({"points": [{"name": "tumor", "point": [4, 3, 2]}]}),
        encoding="utf-8",
    )
    initial_mask = np.zeros((3, 4, 5), dtype=np.uint8)
    initial_mask[1, 1, 1] = 1
    refined_mask = initial_mask.copy()
    refined_mask[2, 3, 4] = 1

    monkeypatch.setenv("AUTOPET_INPUT", str(input_root))
    monkeypatch.setenv("AUTOPET_OUTPUT", str(output_root))
    monkeypatch.setattr(
        public_tracer_router,
        "predict_tracer",
        lambda *args, **kwargs: ("fdg", {"vote": "test"}),
    )
    monkeypatch.setattr(
        runtime,
        "build_initial_prediction",
        lambda *args, **kwargs: InitialPrediction(initial_mask, {"test": 1}, None),
    )
    monkeypatch.setattr(
        runtime,
        "refine_prediction",
        lambda *args, **kwargs: refined_mask,
    )

    runtime.process()

    result = sitk.ReadImage(str(output_root / "subject.mha"))
    np.testing.assert_array_equal(sitk.GetArrayFromImage(result), refined_mask)
    assert result.GetSpacing() == reference.GetSpacing()
