import pytest

from stemforge.orchestrator import RunConfig, load_config


def test_defaults_load():
    cfg = load_config()
    assert isinstance(cfg, RunConfig)
    assert cfg.sample_rate == 44100
    assert cfg.separation.model == "htdemucs_ft"
    assert cfg.separation.stems == ["vocals", "drums", "bass", "other"]
    assert cfg.drums.split.parts[0] == "kick"


def test_dotted_overrides():
    cfg = load_config(overrides={
        "separation.model": "htdemucs_6s",
        "stretch.target_bpm": 120,
        "drums.midi.enabled": True,
    })
    assert cfg.separation.model == "htdemucs_6s"
    assert cfg.stretch.target_bpm == 120
    assert cfg.drums.midi.enabled is True


def test_unknown_key_rejected():
    with pytest.raises(ValueError):
        load_config(overrides={"separation.not_a_field": 1})


def test_roundtrip_to_dict():
    cfg = load_config()
    d = cfg.to_dict()
    assert d["separation"]["model"] == "htdemucs_ft"
    assert set(d) >= {"device", "ingest", "analysis", "separation", "stretch", "midi", "drums", "output"}
