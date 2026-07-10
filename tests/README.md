# Tests

Runnable **without a GPU or ML frameworks** — heavy libs are lazily imported and
skipped via `pytest.importorskip`; the GPU/subprocess paths are mocked. The full
suite is CPU-only and deterministic (seeded RNG, synthetic audio).

```bash
pip install -e ".[dev,ui]"
pytest                       # or: python -m pytest
```

## Core

- `test_config.py` — config load, dotted overrides, unknown-key rejection
- `test_io.py` — AudioTensor, WAV round-trip, sha256 receipts, slugify
- `test_dsp.py` — GM map, velocity scaling, RMS peak window, open/closed hi-hat
- `test_analysis.py` — regression BPM (clean + jittered), BeatGrid
- `test_stretch.py` — identity no-op; librosa length check; match-BPM
- `test_pipeline.py` — end-to-end ingest→manifest with stages disabled
- `test_separation_backends.py` — demucs / uvr / hybrid backend selection
- `test_drum_split.py` — drum teardown (inagoy + uvr), C1 sha256 hash-pin gate
- `test_desktop.py` / `test_cli.py` — launcher + CLI surface

## Hardening

- `test_rcpt.py` — trustworthy manifest: receipt collisions, honest run outcome
- `test_h2v2.py` — concurrency: single-worker queue, per-call scratch, locked loader
- `test_h1.py` / `test_webapp.py` — local server hardening + web API

## Accuracy series

- `test_drum_midi_cymbals.py` — cymbal MIDI on the default path (A1)
- `test_tempo_evidence.py` — honest tempo unknowns + engine fallback chains (A2)
- `test_midi_evidence.py` — Basic Pitch transform ledger, exact reconstruction (A3)
- `test_drum_calibration.py` — per-part velocity, bleed de-dup, tom split, config (A4)
- `test_benchmarks.py` — the `benchmarks/` corpus, metrics, and threshold gate (A5)

The synthetic accuracy corpus and its CI threshold gate live in `benchmarks/`
(see `benchmarks/README.md`).
