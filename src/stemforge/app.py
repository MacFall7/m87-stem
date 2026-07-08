"""§5.9 app — the M87 Space-Tech workstation (Gradio UI).

A workflow-first web app, not a checkbox grid. Four panels:

  1. Extract Stems   — mix -> stems (Fast/Best/SOTA/Max), audition + download.
  2. Drum Teardown   — a drum loop (or the drums stem) -> hit stems + GM drum MIDI.
  3. Melodic -> MIDI — stem/mix -> .mid, monophonic + quantize-to-grid toggles.
  4. Full Teardown   — one drop -> stems + drum hits + MIDI + tempo, grid-aligned.

Styling is a dark "M87 Space-Tech" CSS theme whose colors/fonts are CSS
variables at the top of ``M87_CSS`` (``--m87-bg`` … ``--m87-mono``), so exact
M87 tokens can be dropped in later. gradio is imported lazily; Gradio 6 takes
the theme at ``launch()``, older gradio at ``Blocks()``. All heavy/UVR work runs
in the isolated ``.venv-uvr`` via the pipeline — never in this process.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .orchestrator import Pipeline, load_config, preset_names

# Preset labels derived from the single source of truth; handlers map back with .lower().
PRESET_CHOICES = [p.upper() if p == "sota" else p.capitalize() for p in preset_names()]
DEFAULT_PRESET = "Best"

# Shared loaded-model cache (keyed backend:model in Pipeline) so switching
# panels/presets never reloads weights the process already holds.
_MODEL_CACHE: dict[str, Any] = {}

# Stems we constrain to monophonic when the Melodic panel's toggle is on.
_MELODIC_STEMS = ["bass", "other", "guitar", "piano", "vocals"]


# --------------------------------------------------------------------------- #
# M87 Space-Tech theme — all colors/fonts as CSS variables (swap for real tokens)
# --------------------------------------------------------------------------- #
M87_CSS = """
:root {
  --m87-bg:        #05070d;   /* near-black deep space            */
  --m87-surface:   #0d1220;   /* panel background                 */
  --m87-surface-2: #131a2c;   /* raised surface / inputs          */
  --m87-border:    #1e2740;   /* hairline borders                 */
  --m87-accent:    #22d3ee;   /* primary cyan                     */
  --m87-accent-2:  #a78bfa;   /* secondary violet                 */
  --m87-text:      #e6ecf5;   /* primary text                     */
  --m87-text-dim:  #8b97b0;   /* muted text                       */
  --m87-mono:      'JetBrains Mono','SFMono-Regular',ui-monospace,Menlo,Consolas,monospace;
  --m87-sans:      'Inter',system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
  --m87-radius:    14px;
  --m87-glow:      0 0 0 1px var(--m87-border), 0 8px 30px rgba(0,0,0,.45);
}

.gradio-container, .gradio-container .prose {
  background: radial-gradient(1200px 600px at 20% -10%, rgba(34,211,238,.06) 0, transparent 60%),
              var(--m87-bg) !important;
  color: var(--m87-text) !important;
  font-family: var(--m87-sans) !important;
}
.gradio-container { max-width: 1120px !important; margin: 0 auto !important; }

.m87-header {
  border: 1px solid var(--m87-border);
  border-radius: var(--m87-radius);
  background: linear-gradient(135deg, rgba(34,211,238,.08), rgba(167,139,250,.08)), var(--m87-surface);
  padding: 18px 22px; margin-bottom: 14px; box-shadow: var(--m87-glow);
}
.m87-header h1 {
  font-family: var(--m87-mono) !important; font-weight: 700; letter-spacing: .14em;
  margin: 0; font-size: 1.5rem;
  background: linear-gradient(90deg, var(--m87-accent), var(--m87-accent-2));
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.m87-header .m87-sub { color: var(--m87-text-dim); font-size: .86rem; margin-top: 4px; letter-spacing: .04em; }

h1, h2, h3, .tab-nav button { font-family: var(--m87-mono) !important; letter-spacing: .06em; }

.gradio-container .block, .gradio-container .form,
.gradio-container .panel, .gradio-container fieldset {
  background: var(--m87-surface) !important;
  border: 1px solid var(--m87-border) !important;
  border-radius: var(--m87-radius) !important;
}

.tab-nav { border-bottom: 1px solid var(--m87-border) !important; }
.tab-nav button.selected {
  color: var(--m87-accent) !important;
  border-bottom: 2px solid var(--m87-accent) !important;
}

button.primary, .gradio-container button.primary {
  background: linear-gradient(90deg, var(--m87-accent), var(--m87-accent-2)) !important;
  color: #05070d !important; border: 0 !important; font-weight: 700 !important;
  letter-spacing: .04em; box-shadow: 0 6px 20px rgba(34,211,238,.18) !important;
}
button.secondary { border: 1px solid var(--m87-border) !important; color: var(--m87-text) !important; }

.m87-status {
  font-family: var(--m87-mono) !important; color: var(--m87-accent);
  background: var(--m87-surface-2); border: 1px solid var(--m87-border);
  border-radius: 10px; padding: 8px 12px; min-height: 1.2em;
}
input, textarea, select, .gradio-container .wrap { color: var(--m87-text) !important; }
footer { display: none !important; }
"""


def m87_theme():
    """A Gradio Base theme tuned to the M87 palette (CSS carries the detail)."""
    import gradio as gr

    return gr.themes.Base(
        primary_hue="cyan",
        secondary_hue="purple",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    ).set(
        body_background_fill="#05070d",
        body_text_color="#e6ecf5",
        block_background_fill="#0d1220",
        border_color_primary="#1e2740",
    )


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _slug(name: str) -> str:
    from .io_utils import slugify

    return slugify(name)


def _dropdown_update(choices):
    import gradio as gr

    return gr.Dropdown(choices=choices, value=(choices[0] if choices else None))


def _preview(name, files):
    if not name or not files:
        return None
    for f in files:
        if Path(f).stem == name:
            return f
    return None


def _run(overrides: dict[str, Any], audio_file) -> tuple[dict, Path]:
    cfg = load_config(overrides=overrides)
    manifest = Pipeline(cfg, model_cache=_MODEL_CACHE).run(audio_file)
    out_dir = Path(cfg.output.root) / _slug(Path(manifest["input"]["filename"]).stem)
    return manifest, out_dir


def _glob(out_dir: Path, sub: str, pattern: str) -> list[str]:
    d = out_dir / sub
    return sorted(str(p) for p in d.glob(pattern)) if d.is_dir() else []


def _stage_note(stage: dict, ok: str) -> str:
    if "skipped" in stage:
        return f"skipped — {stage['skipped']}"
    if "error" in stage:
        return f"error — {stage['error']}"
    return ok


def _open_folder(folder: str) -> str:
    """Open `folder` in the OS file manager; degrade to a message when headless."""
    if not folder or not Path(folder).is_dir():
        return "No output yet — run a workflow first."
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
        return f"Couldn't open a file manager ({e}). Output: `{folder}`"


# --------------------------------------------------------------------------- #
# Workflow handlers (return gradio updates; heavy work runs in the pipeline)
# --------------------------------------------------------------------------- #
def extract_stems(audio_file, preset, device, progress=None):
    if not audio_file:
        return "Drop a mix to extract stems.", _dropdown_update([]), None, [], ""
    if progress:
        progress(0.1, desc="Separating…")
    manifest, out_dir = _run({
        "separation.preset": str(preset).lower(),
        "device": device,
        "analysis.enabled": False,
    }, audio_file)
    stems = _glob(out_dir, "stems", "*.wav")
    note = _stage_note(manifest.get("separation", {}), f"{len(stems)} stems")
    status = f"✓ Extract · {note}"
    names = [Path(s).stem for s in stems]
    if progress:
        progress(1.0, desc="Done")
    return status, _dropdown_update(names), (stems[0] if stems else None), stems, str(out_dir)


def drum_teardown(audio_file, device, progress=None):
    if not audio_file:
        return "Drop a drum loop (or a drums stem).", _dropdown_update([]), None, None, [], ""
    if progress:
        progress(0.1, desc="Tearing down the kit…")
    manifest, out_dir = _run({
        "device": device,
        "analysis.enabled": True,          # source BPM for the MIDI + display
        "separation.enabled": False,       # the drop IS the drum material
        "drums.split.enabled": True,
        "drums.split.from_input": True,    # tear down the raw loop itself
        "drums.midi.enabled": True,
    }, audio_file)
    parts = [p for p in _glob(out_dir, "drums", "*.wav")]
    midi = out_dir / "drums" / "drums.mid"
    midi_path = str(midi) if midi.is_file() else None

    ds, dm = manifest.get("drum_split", {}), manifest.get("drum_midi", {})
    if "skipped" in ds or "error" in ds:
        status = f"Drum split {_stage_note(ds, '')}"
    else:
        notes = dm.get("note_count", "?")
        bpm = manifest.get("analysis", {}).get("source_bpm", "?")
        status = f"✓ Teardown · {len(parts)} hit stems · {notes} MIDI notes · {bpm} BPM"
    names = [Path(p).stem for p in parts]
    files = parts + ([midi_path] if midi_path else [])
    if progress:
        progress(1.0, desc="Done")
    return status, _dropdown_update(names), (parts[0] if parts else None), midi_path, files, str(out_dir)


def melodic_midi(audio_file, monophonic, quantize, device, progress=None):
    if not audio_file:
        return "Drop a stem or a mix.", None, [], ""
    if progress:
        progress(0.1, desc="Transcribing…")
    manifest, out_dir = _run({
        "separation.preset": "best",
        "device": device,
        "analysis.enabled": bool(quantize),           # grid only needed to quantize
        "midi.enabled": True,
        "midi.quantize_to_grid": bool(quantize),
        "midi.monophonic_stems": _MELODIC_STEMS if monophonic else [],
    }, audio_file)
    mids = _glob(out_dir, "midi", "*.mid")
    note = _stage_note(manifest.get("midi", {}), f"{len(mids)} MIDI file(s)")
    status = f"✓ Melodic → MIDI · {note}"
    if progress:
        progress(1.0, desc="Done")
    return status, (mids[0] if mids else None), mids, str(out_dir)


def full_teardown(audio_file, preset, device, progress=None):
    if not audio_file:
        return "Drop a track for the full teardown.", {}, [], ""
    if progress:
        progress(0.1, desc="Full teardown…")
    manifest, out_dir = _run({
        "separation.preset": str(preset).lower(),
        "device": device,
        "analysis.enabled": True,
        "midi.enabled": True,
        "drums.split.enabled": True,   # from the separated drums stem
        "drums.midi.enabled": True,
    }, audio_file)
    files = [str(p) for p in sorted(out_dir.rglob("*")) if p.is_file()]
    bpm = manifest.get("analysis", {}).get("source_bpm", "?")
    status = f"✓ Full teardown · {bpm} BPM · {len(files)} files · manifest.json included"
    if progress:
        progress(1.0, desc="Done")
    return status, manifest, files, str(out_dir)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def build_ui(theme=None, css=None):
    import gradio as gr

    # Gradio 6 moved theme + css to launch(); gradio 4/5 take them on Blocks().
    blocks_kwargs: dict[str, Any] = {"title": "M87 · StemForge"}
    if theme is not None:
        blocks_kwargs["theme"] = theme
    if css is not None:
        blocks_kwargs["css"] = css

    with gr.Blocks(**blocks_kwargs) as demo:
        last_out = gr.State("")

        gr.HTML(
            '<div class="m87-header"><h1>M87 · STEMFORGE</h1>'
            '<div class="m87-sub">SPACE-TECH AUDIO WORKSTATION — separation · drum teardown · MIDI</div></div>'
        )
        device = gr.Dropdown(["auto", "cuda", "cpu"], value="auto", label="Device", scale=0)

        # ---- 1. Extract Stems ------------------------------------------------ #
        with gr.Tab("◇ Extract Stems"):
            with gr.Row():
                with gr.Column(scale=1):
                    ex_in = gr.Audio(label="Mix", type="filepath")
                    ex_preset = gr.Dropdown(PRESET_CHOICES, value=DEFAULT_PRESET, label="Quality preset")
                    ex_btn = gr.Button("Extract stems", variant="primary")
                with gr.Column(scale=1):
                    ex_status = gr.Markdown(elem_classes="m87-status")
                    ex_pick = gr.Dropdown([], label="Audition stem")
                    ex_audio = gr.Audio(label="Stem", interactive=False)
                    ex_files = gr.Files(label="Download stems")
            ex_btn.click(
                lambda a, p, d, progress=gr.Progress(track_tqdm=True): extract_stems(a, p, d, progress),
                [ex_in, ex_preset, device], [ex_status, ex_pick, ex_audio, ex_files, last_out],
            )
            ex_pick.change(_preview, [ex_pick, ex_files], ex_audio)

        # ---- 2. Drum Teardown ----------------------------------------------- #
        with gr.Tab("◈ Drum Teardown"):
            with gr.Row():
                with gr.Column(scale=1):
                    dr_in = gr.Audio(label="Drum loop / drums stem", type="filepath")
                    dr_btn = gr.Button("Tear down the kit", variant="primary")
                with gr.Column(scale=1):
                    dr_status = gr.Markdown(elem_classes="m87-status")
                    dr_pick = gr.Dropdown([], label="Audition hit")
                    dr_audio = gr.Audio(label="Hit stem", interactive=False)
                    dr_midi = gr.File(label="Drum MIDI (.mid)")
                    dr_files = gr.Files(label="Download hits + MIDI")
            dr_btn.click(
                lambda a, d, progress=gr.Progress(track_tqdm=True): drum_teardown(a, d, progress),
                [dr_in, device], [dr_status, dr_pick, dr_audio, dr_midi, dr_files, last_out],
            )
            dr_pick.change(_preview, [dr_pick, dr_files], dr_audio)

        # ---- 3. Melodic -> MIDI --------------------------------------------- #
        with gr.Tab("◉ Melodic → MIDI"):
            with gr.Row():
                with gr.Column(scale=1):
                    me_in = gr.Audio(label="Stem or mix", type="filepath")
                    me_mono = gr.Checkbox(label="Monophonic (bass/lead accuracy)", value=False)
                    me_quant = gr.Checkbox(label="Quantize to beat grid", value=False)
                    me_btn = gr.Button("Transcribe → MIDI", variant="primary")
                with gr.Column(scale=1):
                    me_status = gr.Markdown(elem_classes="m87-status")
                    me_first = gr.File(label="MIDI")
                    me_files = gr.Files(label="Download .mid files")
            me_btn.click(
                lambda a, m, q, d, progress=gr.Progress(track_tqdm=True): melodic_midi(a, m, q, d, progress),
                [me_in, me_mono, me_quant, device], [me_status, me_first, me_files, last_out],
            )

        # ---- 4. Full Teardown ----------------------------------------------- #
        with gr.Tab("✦ Full Teardown"):
            with gr.Row():
                with gr.Column(scale=1):
                    fu_in = gr.Audio(label="Track", type="filepath")
                    fu_preset = gr.Dropdown(PRESET_CHOICES, value=DEFAULT_PRESET, label="Quality preset")
                    fu_btn = gr.Button("Full teardown", variant="primary")
                with gr.Column(scale=1):
                    fu_status = gr.Markdown(elem_classes="m87-status")
                    fu_files = gr.Files(label="Grid-aligned bundle")
                    fu_manifest = gr.JSON(label="manifest.json")
            fu_btn.click(
                lambda a, p, d, progress=gr.Progress(track_tqdm=True): full_teardown(a, p, d, progress),
                [fu_in, fu_preset, device], [fu_status, fu_manifest, fu_files, last_out],
            )

        # ---- global footer: open output folder ------------------------------ #
        with gr.Row():
            open_btn = gr.Button("📂 Open output folder", variant="secondary")
            open_status = gr.Markdown(elem_classes="m87-status")
        open_btn.click(_open_folder, [last_out], [open_status])

    return demo


def launch(host: str = "127.0.0.1", port: int = 7860, share: bool = False,
           open_browser: bool = True):
    """Build + launch the UI. Gradio 6 takes the theme at launch(), not Blocks().

    ``open_browser`` maps to gradio's ``inbrowser`` — it pops the browser once
    the server is up (used by ``stemforge ui --open`` and the desktop shortcut).
    """
    import gradio as gr

    theme = m87_theme()
    kw = dict(server_name=host, server_port=port, share=share, inbrowser=open_browser)
    if int(gr.__version__.split(".")[0]) >= 6:
        return build_ui().launch(theme=theme, css=M87_CSS, **kw)
    return build_ui(theme=theme, css=M87_CSS).launch(**kw)


if __name__ == "__main__":
    launch()
