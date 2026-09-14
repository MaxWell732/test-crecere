from extraction import cache
from extraction import config as cfgmod


class _Model:  # stands in for a loaded model object (not JSON serializable)
    pass


def test_run_per_call_writes_stage_meta_with_model_objects(tmp_path, monkeypatch):
    stage_dir = tmp_path / "vad"
    monkeypatch.setitem(cfgmod.STAGES, "vad", {**cfgmod.STAGES["vad"], "dir": stage_dir})

    def setup(ctx):
        ctx.models["vad"] = _Model()
        ctx.models["vad_name"] = "silero-vad test"

    def process(call_id, ctx):
        if call_id == "C002":
            raise ValueError("boom")
        return {"value": 1}

    result = cache.run_per_call("vad", ["C001", "C002"], process, lambda c: stage_dir / f"{c}.json", setup=setup)
    assert result.ok == ["C001"] and "C002" in result.failed
    meta = cache.read_json(stage_dir / "_stage.json")
    assert meta["n_ok"] == 1 and meta["n_failed"] == 1
    assert meta["models"] == {"vad_name": "silero-vad test"}

    again = cache.run_per_call("vad", ["C001"], process, lambda c: stage_dir / f"{c}.json")
    assert again.skipped == ["C001"] and not again.ok


def test_empty_extra_key_keeps_plain_config_key(tmp_path, monkeypatch):
    stage_dir = tmp_path / "diarization"
    monkeypatch.setitem(cfgmod.STAGES, "diarize", {**cfgmod.STAGES["diarize"], "dir": stage_dir})
    first = cache.run_per_call("diarize", ["C001"], lambda c, ctx: {"ok": True}, lambda c: stage_dir / f"{c}.json")
    assert first.ok == ["C001"]
    again = cache.run_per_call("diarize", ["C001"], lambda c, ctx: {"ok": True}, lambda c: stage_dir / f"{c}.json",
                               extra_key=lambda c: "")
    assert again.skipped == ["C001"]
    changed = cache.run_per_call("diarize", ["C001"], lambda c, ctx: {"ok": True}, lambda c: stage_dir / f"{c}.json",
                                 extra_key=lambda c: "ov123")
    assert changed.ok == ["C001"]


def test_regenerated_upstream_file_makes_downstream_stale(tmp_path, monkeypatch):
    import os

    stage_dir = tmp_path / "turns"
    monkeypatch.setitem(cfgmod.STAGES, "turns", {**cfgmod.STAGES["turns"], "dir": stage_dir})
    upstream = tmp_path / "asr" / "C001.json"
    cache.write_json(upstream, {"v": 1})

    def run():
        return cache.run_per_call("turns", ["C001"], lambda c, ctx: {"ok": True}, lambda c: stage_dir / f"{c}.json",
                                  extra_key=lambda c: cache.file_fingerprint(upstream))

    assert run().ok == ["C001"]
    assert run().skipped == ["C001"]
    cache.write_json(upstream, {"v": 2, "regenerated": True})
    st = upstream.stat()
    os.utime(upstream, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    assert run().ok == ["C001"]
