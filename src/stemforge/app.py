"""§5.9 app — local Gradio web UI.

Drag-drop upload, quality-preset dropdown (Fast / Best / SOTA / Max), progress
feedback, per-stem preview, "open output folder", and download of the full
output bundle. A Batch tab processes many files without reloading models (one
shared Pipeline instance). gradio is imported lazily; Gradio 6 takes the theme
at launch(), not Blocks().
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .orchestrator import Pipeline, load_config, preset_names

# Display labels derived from the single preset source of truth; the handlers
# map back with .lower().
PRESET_CHOICES = [p.upper() if p == "sota" else p.capitalize() for p in preset_names()]
PRESET_INFO = (
    "Fast = htdemucs · Best = htdemucs_ft ×2 shifts · "
    "SOTA = BS-Roformer (vocals + instrumental) · Max = hybrid 4-stem"
)

# Loaded models shared across runs/clicks (keyed by backend:model in Pipeline),
# so switching presets in the UI never reloads weights it already has.
_MODEL_CACHE: dict[str, Any] = {}


def _run_single(audio_file, preset, do_midi, do_split, do_dmidi, do_stretch,
                target_bpm, onset_threshold, segment, device):
    if not audio_file:
        return "Upload a track first.", {}, [], None, [], ""
    overrides: dict[str, Any] = {
        "separation.preset": str(preset).lower(),
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
    manifest = Pipeline(cfg, model_cache=_MODEL_CACHE).run(audio_file)

    song = Path(manifest["input"]["filename"]).stem
    out_dir = Path(cfg.output.root) / _slug(song)
    files = [str(p) for p in sorted(out_dir.rglob("*")) if p.is_file()]
    stems = sorted(str(p) for p in (out_dir / "stems").glob("*.wav")) if (out_dir / "stems").is_dir() else []
    stem_names = [Path(s).stem for s in stems]

    bpm = manifest.get("analysis", {}).get("source_bpm", "?")
    sep = manifest.get("separation", {})
    sep_note = sep.get("model") or sep.get("skipped") or sep.get("error", "?")
    status = f"Done · {bpm} BPM · {sep_note} · {len(files)} files in `{out_dir}`"
    preview = stems[0] if stems else None
    return status, manifest, files, preview, _dropdown_update(stem_names), str(out_dir)


def _preview(stem_name, files):
    if not stem_name or not files:
        return None
    for f in files:
        if Path(f).stem == stem_name:
            return f
    return None


def _open_folder(folder: str) -> str:
    """Open `folder` in the OS file manager; degrade to a message when headless."""
    if not folder or not Path(folder).is_dir():
        return "No output folder yet — run a track first."
    import subprocess
    import sys

    try:
        if sys.platform == "win32":
            os.startfile(folder)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", folder])
        else:
            subprocess.Popen(["xdg-open", folder])
        return f"Asked the OS to open `{folder}`."
    except Exception as e:  # noqa: BLE001 - headless box / no file manager
        return f"Couldn't open a file manager ({e}). Output folder: `{folder}`"


def _run_batch(audio_files, preset, do_midi, do_split, do_dmidi, do_stretch, target_bpm, device):
    if not audio_files:
        return "Add files first.", [], []
    overrides: dict[str, Any] = {
        "separation.preset": str(preset).lower(), "device": device,
        "midi.enabled": bool(do_midi),
        "drums.split.enabled": bool(do_split),
        "drums.midi.enabled": bool(do_dmidi),
        "stretch.enabled": bool(do_stretch) or bool(target_bpm),
    }
    if target_bpm:
        overrides["stretch.target_bpm"] = float(target_bpm)
    cfg = load_config(overrides=overrides)
    pipe = Pipeline(cfg, model_cache=_MODEL_CACHE)  # models load once, shared across runs

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
    return gr.Dropdown(choices=choices, value=value)


def build_ui(theme=None):
    import gradio as gr

    blocks_kwargs = {"title": "StemForge"}
    if theme is not None:  # gradio < 6 only: theme is a Blocks() kwarg there
        blocks_kwargs["theme"] = theme
    with gr.Blocks(**blocks_kwargs) as demo:
        gr.Markdown("# StemForge\nLocal stem separation · MIDI · BPM stretch · drum decomposition")

        with gr.Tab("Single"):
            with gr.Row():
                with gr.Column(scale=1):
                    audio_in = gr.Audio(label="Input mix", type="filepath")
                    preset = gr.Dropdown(
                        PRESET_CHOICES, value="Best", label="Quality preset", info=PRESET_INFO,
                    )
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
                    open_btn = gr.Button("📂 Open output folder")
                    open_status = gr.Markdown()
                    manifest_out = gr.JSON(label="manifest.json")
            out_dir_state = gr.State("")

            def run_single(audio_file, preset, do_midi, do_split, do_dmidi, do_stretch,
                           target_bpm, onset_threshold, segment, device,
                           progress=gr.Progress(track_tqdm=True)):
                progress(0.05, desc="Ingest + analysis…")
                result = _run_single(audio_file, preset, do_midi, do_split, do_dmidi,
                                     do_stretch, target_bpm, onset_threshold, segment, device)
                progress(1.0, desc="Done")
                return result

            run_btn.click(
                run_single,
                [audio_in, preset, do_midi, do_split, do_dmidi, do_stretch, target_bpm, onset, segment, device],
                [status, manifest_out, downloads, stem_audio, stem_pick, out_dir_state],
            )
            stem_pick.change(_preview, [stem_pick, downloads], stem_audio)
            open_btn.click(_open_folder, [out_dir_state], [open_status])

        with gr.Tab("Batch"):
            b_files = gr.Files(label="Input tracks", file_types=["audio"])
            with gr.Row():
                b_preset = gr.Dropdown(PRESET_CHOICES, value="Best", label="Quality preset")
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

            def run_batch(audio_files, preset, do_midi, do_split, do_dmidi, do_stretch,
                          target_bpm, device, progress=gr.Progress(track_tqdm=True)):
                progress(0.05, desc="Processing batch…")
                result = _run_batch(audio_files, preset, do_midi, do_split, do_dmidi,
                                    do_stretch, target_bpm, device)
                progress(1.0, desc="Done")
                return result

            b_run.click(
                run_batch,
                [b_files, b_preset, b_midi, b_split, b_dmidi, b_stretch, b_bpm, b_device],
                [b_status, b_manifest, b_downloads],
            )

    return demo


def launch(host: str = "127.0.0.1", port: int = 7860, share: bool = False,
           open_browser: bool = True):
    """Build + launch the UI. Gradio 6 takes the theme at launch(), not Blocks().

    ``open_browser`` maps to gradio's ``inbrowser`` — it pops the browser once
    the server is up.
    """
    import gradio as gr

    theme = gr.themes.Soft()
    kw = dict(server_name=host, server_port=port, share=share, inbrowser=open_browser)
    if int(gr.__version__.split(".")[0]) >= 6:
        return build_ui().launch(theme=theme, **kw)
    return build_ui(theme=theme).launch(**kw)


if __name__ == "__main__":
    launch()
