from extraction import diarize, paths


def test_override_key_only_for_listed_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "CONFIGS", tmp_path)
    assert diarize.load_overrides() == {} and diarize.override_key("C001") == ""
    (tmp_path / "diarization_overrides.yaml").write_text(
        "overrides:\n  C001: {num_speakers: 2, reason: collapse}\n", encoding="utf-8")
    key = diarize.override_key("C001")
    assert key.startswith("ov") and len(key) == 12
    assert diarize.override_key("C002") == ""
    (tmp_path / "diarization_overrides.yaml").write_text(
        "overrides:\n  C001: {pipeline: pyannote/speaker-diarization-3.1, num_speakers: 2, reason: collapse}\n",
        encoding="utf-8")
    assert diarize.override_key("C001") not in ("", key)
