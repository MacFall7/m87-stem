"""StemForge accuracy benchmarks (A-series PR-A5).

A synthetic render-roundtrip corpus with a known ground truth, a set of accuracy
metrics, and a threshold gate wired into CI. The corpus is generated
deterministically (seeded RNG) so runs are reproducible; every report embeds the
fixture SHA-256, the config hash, and library/model versions as provenance.
"""

from . import corpus, metrics  # noqa: F401
