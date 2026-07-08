"""§5.9 app — local Gradio web UI.

Drag-drop upload, model + threshold controls, target-BPM, per-stem preview, and
download of the full output bundle. A Batch tab processes many files without
reloading models (one shared Pipeline instance). gradio is imported lazily.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .orchestrator import Pipeline, load_config


def _run_single(audio_file, model, do_midi, do_split, do_dmidi, do_stretch,
                target_bpm, onset_threshold, segment, device):
    if not audio_file:
        return "Upload a track first.", {}, [], None, []
    overrides: dict[str, Any] = {
        "separation.model": model,
        "separation.segment": float(segment),
        "device": device,
        "midi.enabled": bool(do_midi),
        "midi.onset_threshold": float(onset_threshold),
        "drums.split.enabled": bool(do_split),
        "drums.midi.enabled": bool(do_dmidi),
        "stretch.enabled": bool(do_stretch) or bool(target_bpm),
    }
    if target_bpm:
        overrides["stretch.target_bpm"] = float(target_bpm)

    cfg = load_config(overrides=overrides)
    manifest = Pipeline(cfg).run(audio_file)

    song = Path(manifest["input"]["filename"]).stem
    out_dir = Path(cfg.output.root) / _slug(song)
    files = [str(p) for p in sorted(out_dir.rglob("*")) if p.is_file()]
    stems = sorted(str(p) for p in (out_dir / "stems").glob("*.wav")) if (out_dir / "stems").is_dir() else []
    stem_names = [Path(s).stem for s in stems]

    bpm = manifest.get("analysis", {}).get("source_bpm", "?")
    status = f"Done · {bpm} BPM · {len(files)} files in `{out_dir}`"
    preview = stems[0] if stems else None
    return status, manifest, files, preview, _dropdown_update(stem_names)


def _preview(stem_name, files):
    if not stem_name or not files:
        return None
    for f in files:
        if Path(f).stem == stem_name:
            return f
    return None


def _run_batch(audio_files, model, do_midi, do_split, do_dmidi, do_stretch, target_bpm, device):
    if not audio_files:
        return "Add files first.", [], []
    overrides: dict[str, Any] = {
        "separation.model": model, "device": device,
        "midi.enabled": bool(do_midi),
        "drums.split.enabled": bool(do_split),
        "drums.midi.enabled": bool(do_dmidi),
        "stretch.enabled": bool(do_stretch) or bool(target_bpm),
    }
    if target_bpm:
        overrides["stretch.target_bpm"] = float(target_bpm)
    cfg = load_config(overrides=overrides)
    pipe = Pipeline(cfg)  # one instance => models load once

    paths = [f.name if hasattr(f, "name") else f for f in audio_files]
    manifests = pipe.run_batch(paths)

    all_files: list[str] = []
    for m in manifests:
        song = _slug(Path(m["input"]["filename"]).stem)
        out_dir = Path(cfg.output.root) / song
        all_files += [str(p) for p in sorted(out_dir.rglob("*")) if p.is_file()]
    return f"Processed {len(manifests)} file(s).", manifests, all_files


def _slug(name: str) -> str:
    from .io_utils import slugify

    return slugify(name)


def _dropdown_update(choices):
    import gradio as gr

    value = choices[0] if choices else None
    return gr.update(choices=choices, value=value)


def build_ui():
    import gradio as gr

    from . import separate as sep_mod

    with gr.Blocks(title="StemForge", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# StemForge\nLocal stem separation · MIDI · BPM stretch · drum decomposition")

        with gr.Tab("Single"):
            with gr.Row():
                with gr.Column(scale=1):
                    audio_in = gr.Audio(label="Input mix", type="filepath")
                    model = gr.Dropdown(sep_mod.available_models(), value="htdemucs_ft", label="Demucs model")
                    with gr.Row():
                        do_midi = gr.Checkbox(label="Melodic MIDI")
                        do_split = gr.Checkbox(label="Drum split")
                        do_dmidi = gr.Checkbox(label="Drum MIDI")
                        do_stretch = gr.Checkbox(label="Stretch")
                    target_bpm = gr.Number(label="Target BPM (blank = off)", value=None)
                    onset = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="MIDI onset threshold")
                    segment = gr.Slider(4, 20, value=10, step=1, label="Segment (VRAM)")
                    device = gr.Dropdown(["auto", "cuda", "cpu"], value="auto", label="Device")
                    run_btn = gr.Button("Run", variant="primary")
                with gr.Column(scale=1):
                    status = gr.Markdown()
                    stem_pick = gr.Dropdown([], label="Preview stem")
                    stem_audio = gr.Audio(label="Stem preview", interactive=False)
                    downloads = gr.Files(label="Download bundle")
                    manifest_out = gr.JSON(label="manifest.json")

            run_btn.click(
                _run_single,
                [audio_in, model, do_midi, do_split, do_dmidi, do_stretch, target_bpm, onset, segment, device],
                [status, manifest_out, downloads, stem_audio, stem_pick],
            )
            stem_pick.change(_preview, [stem_pick, downloads], stem_audio)

        with gr.Tab("Batch"):
            b_files = gr.Files(label="Input tracks", file_types=["audio"])
            with gr.Row():
                b_model = gr.Dropdown(sep_mod.available_models(), value="htdemucs_ft", label="Model")
                b_midi = gr.Checkbox(label="MIDI")
                b_split = gr.Checkbox(label="Drum split")
                b_dmidi = gr.Checkbox(label="Drum MIDI")
                b_stretch = gr.Checkbox(label="Stretch")
                b_bpm = gr.Number(label="Target BPM", value=None)
                b_device = gr.Dropdown(["auto", "cuda", "cpu"], value="auto", label="Device")
            b_run = gr.Button("Run batch", variant="primary")
            b_status = gr.Markdown()
            b_manifest = gr.JSON(label="manifests")
            b_downloads = gr.Files(label="All outputs")
            b_run.click(
                _run_batch,
                [b_files, b_model, b_midi, b_split, b_dmidi, b_stretch, b_bpm, b_device],
                [b_status, b_manifest, b_downloads],
            )

    return demo


if __name__ == "__main__":
    build_ui().launch()
