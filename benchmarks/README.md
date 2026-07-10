# StemForge accuracy benchmarks (A-series PR-A5)

A synthetic **render-roundtrip** corpus with a known ground truth, a set of
accuracy metrics, and a threshold gate wired into CI. It measures the accuracy
work from PR-A1..A4 (cymbal MIDI, tempo evidence, drum calibration) and fails the
build on a regression.

## Run

```bash
python -m benchmarks.run_benchmarks                     # print a JSON report, exit 1 on a violation
python -m benchmarks.run_benchmarks --report report.json --corpus-dir corpus/
```

## Corpus (`corpus.py`)

Everything is rendered deterministically (seeded `numpy.default_rng`) so runs
reproduce and the fixture SHA-256 is stable:

- **Tempo** — noise-pulse trains at constant **90 / 120 / 140 BPM**.
- **Drums** — a backbeat groove with per-part ground truth (kick / snare / toms),
  quiet **ghost** snares, and a short low-tom fill; plus a **bleed** variant with
  weak phantom onsets in the toms part coincident with each kick.

## Metrics (`metrics.py`)

| Metric | Source | Gated |
|--------|--------|-------|
| `tempo_abs_error_bpm_max` | analysis (A2) | ✅ |
| `half_double_correct` | analysis candidates (A2) | ✅ |
| `drum_f1_min` (kick+snare onset F1) | drum_midi (A1/A4) | ✅ |
| `duplicate_rate_max` (cross-part) | drum_midi de-dup (A4) | ✅ |
| `velocity_rank_corr_min` | drum_midi velocity (A4) | ✅ |
| toms F1, per-fixture detail | drum_midi | reported, not gated (sparse) |

**Skipped (reported, never silently dropped):** melodic onset/pitch F1,
note fragmentation (need `basic-pitch`, absent in the offline harness);
beat alignment (needs a beat-aligned reference); quantization displacement
(covered by the A3 ledger's `max_displacement_s`).

## Thresholds (`thresholds.yaml`)

Bounds with margin over a clean baseline. CI (`.github/workflows/benchmarks.yml`)
runs `run_benchmarks` and fails on any violation. Tighten as accuracy improves.

## Provenance

Every report embeds the **fixture SHA-256**, the drum-midi **config hash**, and
**library/model versions** captured at run time — so a report is traceable to the
exact corpus, config, and dependency set that produced it.
