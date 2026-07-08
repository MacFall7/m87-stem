# Tests

Runnable **without a GPU or ML frameworks** — heavy libs are lazily imported and
skipped via `pytest.importorskip`.

- `test_config.py` — config load, dotted overrides, unknown-key rejection
- `test_io.py` — AudioTensor, WAV round-trip, sha256 receipts, slugify
- `test_dsp.py` — GM map, velocity scaling, RMS peak window, open/closed hi-hat
- `test_analysis.py` — regression BPM (clean + jittered), BeatGrid
- `test_stretch.py` — identity no-op; librosa length check (skipped if absent)
- `test_pipeline.py` — end-to-end ingest→manifest with stages disabled (no ML deps)

Golden-file tests (Phase 7): drop reference stems/MIDI under `tests/golden/` and
compare module outputs against them once the model weights are wired locally.

```bash
pip install -e ".[dev]"
pytest
```
