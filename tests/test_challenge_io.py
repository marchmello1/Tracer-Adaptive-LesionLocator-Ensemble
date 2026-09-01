import json

import pytest

from candidate_runtime.challenge_io import has_effective_scribbles, load_request, normalize_prompts


def test_normalize_prompts_accepts_only_named_coordinate_triplets():
    actual = normalize_prompts(
        {
            "points": [
                {"name": "tumor", "point": [1.9, 2, 3]},
                {"name": "background", "point": [4, "5", 6]},
                {"name": "other", "point": [1, 2, 3]},
                {"name": "tumor", "point": [1, 2]},
                None,
            ]
        }
    )
    assert actual == {"tumor": [[1, 2, 3]], "background": [[4, 5, 6]]}


def test_normalize_prompts_rejects_non_document():
    assert normalize_prompts([]) == {"tumor": [], "background": []}


def test_load_request_reports_missing_prompt_document(tmp_path):
    (tmp_path / "images" / "ct").mkdir(parents=True)
    (tmp_path / "images" / "pet").mkdir(parents=True)
    (tmp_path / "images" / "ct" / "case.mha").touch()
    (tmp_path / "images" / "pet" / "case.mha").touch()
    with pytest.raises(ValueError, match="missing lesion-clicks"):
        load_request(tmp_path)


def test_load_request_preserves_case_identifier_and_prompts(tmp_path):
    (tmp_path / "images" / "ct").mkdir(parents=True)
    (tmp_path / "images" / "pet").mkdir(parents=True)
    (tmp_path / "images" / "ct" / "subject.mha").touch()
    (tmp_path / "images" / "pet" / "subject.mha").touch()
    (tmp_path / "lesion-clicks.json").write_text(
        json.dumps({"points": [{"name": "tumor", "point": [1, 2, 3]}]}),
        encoding="utf-8",
    )
    request = load_request(tmp_path)
    assert request.volumes.identifier == "subject"
    assert request.prompts["tumor"] == [[1, 2, 3]]


def test_effective_scribble_detection_ignores_malformed_entries():
    assert has_effective_scribbles({"tumor": [[1, 2, 3]], "background": []})
    assert not has_effective_scribbles({"tumor": [[1, 2]], "background": [None]})
