"""§5.9 cli — Typer command line. Scriptable batch counterpart to the Gradio UI.

Examples
--------
    stemforge doctor
    stemforge setup-sota                                 # one-time: isolated audio-separator venv
    stemforge separate song.wav --model htdemucs_ft -o out/
    stemforge separate song.wav --preset sota            # BS-Roformer via the isolated venv
    stemforge run song.wav --preset max --midi --stretch --target-bpm 120
    stemforge match-bpm song.mp3 -t 120                  # stretch a whole file, pitch preserved
    stemforge match-bpm song.mp3 -t 120 -s 140           # override half/double detection
    stemforge ui
"""


from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .orchestrator import Pipeline, load_config, preset_names

app = typer.Typer(add_completion=False, help="StemForge — local audio workstation.")
console = Console()

_PRESET_HELP = (
    "Quality preset: " + " | ".join(preset_names())
    + " (overrides --model; --set separation.* still wins)."
)


# --------------------------------------------------------------------------- #
def _overrides_from_sets(sets: list[str]) -> dict[str, object]:
    out: dict[str, object] = {}
    for item in sets:
        if "=" not in item:
            raise typer.BadParameter(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        out[key.strip()] = _coerce(raw.strip())
    return out


def _coerce(v: str) -> object:
    low = v.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"null", "none"}:
        return None
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def _summarize(manifest: dict) -> None:
    t = Table(title=f"StemForge · {manifest['input']['filename']}", show_header=True)
    t.add_column("stage")
    t.add_column("result", overflow="fold")

    ana = manifest.get("analysis", {})
    if "source_bpm" in ana:
        t.add_row("analysis", f"{ana['engine']} · {ana['source_bpm']} BPM · {ana['num_beats']} beats")
    else:
        t.add_row("analysis", str(ana))

    sep = manifest.get("separation", {})
    if "files" in sep:
        t.add_row("separation", f"{sep['model']} · " + ", ".join(sep["files"]))
    else:
        t.add_row("separation", str(sep))

    for key in ("midi", "drum_split", "drum_midi", "stretch"):
        val = manifest.get(key, {})
        if isinstance(val, dict) and "files" in val:
            t.add_row(key, ", ".join(map(str, (val["files"] if isinstance(val["files"], list) else val["files"].keys()))))
        else:
            t.add_row(key, str(val))
    console.print(t)


# --------------------------------------------------------------------------- #
@app.command()
def doctor() -> None:
    """Phase-0 gate: verify GPU, ffmpeg, rubberband, and imports."""
    import shutil

    t = Table(title="StemForge doctor", show_header=True)
    t.add_column("check")
    t.add_column("status")

    # torch / CUDA
    try:
        import torch

        cuda = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if cuda else "-"
        t.add_row("torch", f"{torch.__version__}")
        t.add_row("CUDA", f"[green]available[/] · {name}" if cuda else "[yellow]CPU only[/]")
    except Exception as e:  # noqa: BLE001
        t.add_row("torch", f"[red]missing[/] ({e})")

    for mod in ("demucs", "beat_this", "onnxruntime", "basic_pitch", "pyrubberband", "fastapi", "uvicorn"):
        try:
            __import__(mod)
            t.add_row(mod, "[green]ok[/]")
        except Exception:  # noqa: BLE001
            t.add_row(mod, "[yellow]not installed[/]")

    # audio-separator lives in its OWN venv (never imported in-process)
    from .separate_uvr import find_cli, resolve_venv_dir, venv_torch_status

    venv_dir = resolve_venv_dir(load_config().separation.uvr_venv)
    cli_path = find_cli(venv_dir)
    t.add_row(
        "audio-separator (isolated venv)",
        f"[green]ok[/] · {cli_path}" if cli_path else "[yellow]not set up — run `stemforge setup-sota`[/]",
    )
    if cli_path:
        ts = venv_torch_status(venv_dir)
        if ts is None:
            t.add_row(".venv-uvr torch", "[yellow]not importable[/]")
        elif ts.get("cuda"):
            t.add_row(".venv-uvr torch", f"{ts.get('version')} · [green]CUDA[/] · {ts.get('name')}")
        else:
            t.add_row(".venv-uvr torch", f"{ts.get('version')} · [yellow]CPU only — re-run `stemforge setup-sota`[/]")

    for name in ("ffmpeg", "rubberband"):
        t.add_row(name, "[green]ok[/]" if shutil.which(name) else "[yellow]not on PATH[/]")

    console.print(t)


@app.command("setup-sota")
def setup_sota(
    venv: Optional[Path] = typer.Option(
        None, "--venv", help="Venv location (default: separation.uvr_venv = <project>/.venv-uvr)."
    ),
) -> None:
    """Create/repair the ISOLATED audio-separator venv (idempotent).

    Installs CUDA torch (cu124 index) + audio-separator into that venv only —
    the main environment's torch/numpy are never touched.
    """
    from .separate_uvr import setup_sota_env

    ok = setup_sota_env(load_config().separation, venv=venv, log=console.print)
    if ok:
        console.print("[green]SOTA separation ready[/] — try: stemforge separate song.wav --preset sota")
    else:
        console.print("[red]setup incomplete[/] — fix the issue above and re-run `stemforge setup-sota`")
        raise typer.Exit(code=1)


@app.command()
def separate(
    input: Path = typer.Argument(..., exists=True, dir_okay=False, help="Audio file."),
    model: str = typer.Option("htdemucs_ft", help="htdemucs_ft | htdemucs_6s | htdemucs"),
    preset: Optional[str] = typer.Option(None, "--preset", help=_PRESET_HELP),
    out: Path = typer.Option(Path("out"), "-o", "--out", help="Output root."),
    segment: float = typer.Option(10.0, help="VRAM control (seconds)."),
    device: str = typer.Option("auto", help="auto | cuda | cpu"),
) -> None:
    """Separation only (Phase 1)."""
    overrides: dict[str, object] = {
        "separation.model": model, "separation.segment": segment,
        "output.root": str(out), "device": device,
        "analysis.enabled": False,
    }
    if preset:  # preset picks the model; drop the --model default so it applies
        overrides["separation.preset"] = preset
        overrides.pop("separation.model")
    cfg = load_config(overrides=overrides)
    manifest = Pipeline(cfg).run(input)
    _summarize(manifest)


@app.command()
def analyze(
    input: Path = typer.Argument(..., exists=True, dir_okay=False),
    engine: str = typer.Option("beat_this", help="beat_this | librosa"),
    device: str = typer.Option("auto"),
) -> None:
    """Tempo / beat / downbeat only."""
    cfg = load_config(overrides={
        "analysis.engine": engine, "device": device,
        "separation.enabled": False,
    })
    manifest = Pipeline(cfg).run(input)
    console.print(manifest.get("analysis", {}))


@app.command("match-bpm")
def match_bpm(
    input: Path = typer.Argument(..., exists=True, dir_okay=False, help="Audio file (any format)."),
    target_bpm: float = typer.Option(..., "-t", "--target-bpm", help="Target BPM (required)."),
    source_bpm: Optional[float] = typer.Option(
        None, "-s", "--source-bpm",
        help="Override the detected source BPM (defeats half/double-tempo errors).",
    ),
    out: Path = typer.Option(Path("out"), "-o", "--out", help="Output root."),
    engine: str = typer.Option("rubberband", help="rubberband | signalsmith | librosa"),
    detect_engine: str = typer.Option("beat_this", "--detect-engine", help="beat_this | librosa"),
    device: str = typer.Option("auto", help="auto | cuda | cpu"),
) -> None:
    """Match a WHOLE file to a target BPM (pitch preserved, no separation)."""
    from . import stretch
    from .io_utils import slugify

    out_dir = Path(out) / slugify(input.stem) / "matched"
    res = stretch.match_bpm_file(
        str(input), target_bpm, out_dir, source_bpm=source_bpm,
        engine=engine, detect_engine=detect_engine, device=device,
    )

    t = Table(title=f"Match BPM · {input.name}", show_header=True)
    t.add_column("field")
    t.add_column("value", overflow="fold")
    if "skipped" in res:
        t.add_row("skipped", f"[yellow]{res['skipped']}[/]")
    elif "error" in res:
        t.add_row("error", f"[red]{res['error']}[/]")
    else:
        src_note = "[cyan]overridden[/]" if res["source_bpm_overridden"] else "detected"
        t.add_row("source BPM", f"{res['source_bpm']} ({src_note})")
        if not res["source_bpm_overridden"]:
            d = res["source_bpm_detected"]
            t.add_row("half / double", f"{round(d/2, 1)} / {round(d*2, 1)} (re-run with -s to override)")
        t.add_row("target BPM", str(res["target_bpm"]))
        t.add_row("ratio", str(res["ratio"]))
        t.add_row("engine", res["engine"])
        t.add_row("output", res["output"])
    console.print(t)
    if "skipped" in res or "error" in res:
        raise typer.Exit(code=1)


@app.command()
def run(
    input: Path = typer.Argument(..., exists=True, dir_okay=False),
    config: Optional[Path] = typer.Option(None, help="Custom YAML config."),
    out: Path = typer.Option(Path("out"), "-o", "--out"),
    model: str = typer.Option("htdemucs_ft"),
    preset: Optional[str] = typer.Option(None, "--preset", help=_PRESET_HELP),
    target_bpm: Optional[float] = typer.Option(None, "--target-bpm"),
    midi: bool = typer.Option(False, "--midi"),
    drum_split: bool = typer.Option(False, "--drum-split"),
    drum_midi: bool = typer.Option(False, "--drum-midi"),
    stretch: bool = typer.Option(False, "--stretch"),
    device: str = typer.Option("auto"),
    segment: float = typer.Option(10.0),
    set_: list[str] = typer.Option([], "--set", help="Dotted override key=value (repeatable)."),
) -> None:
    """Full pipeline (all four capabilities, gated by flags)."""
    overrides: dict[str, object] = {
        "output.root": str(out),
        "separation.model": model,
        "separation.segment": segment,
        "device": device,
        "midi.enabled": midi,
        "drums.split.enabled": drum_split,
        "drums.midi.enabled": drum_midi,
        "stretch.enabled": stretch or target_bpm is not None,
    }
    if preset:  # preset picks the model; drop the --model default so it applies
        overrides["separation.preset"] = preset
        overrides.pop("separation.model")
    if target_bpm is not None:
        overrides["stretch.target_bpm"] = target_bpm
    overrides.update(_overrides_from_sets(set_))
    cfg = load_config(path=config, overrides=overrides)
    manifest = Pipeline(cfg).run(input)
    _summarize(manifest)
    console.print(f"[dim]manifest:[/] {Path(cfg.output.root) / manifest['input']['filename'].rsplit('.',1)[0] / 'manifest.json'}")


@app.command()
def ui(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(7860),
    open_browser: bool = typer.Option(
        True, "--open/--no-open", help="Open the app in a browser once the server is up."
    ),
) -> None:
    """Launch the M87 workstation (bespoke FastAPI web app, served by uvicorn)."""
    from .webapp import launch

    console.print(f"[green]M87 · StemForge[/] → http://{host}:{port}/")
    launch(host=host, port=port, open_browser=open_browser)


@app.command("desktop-shortcut")
def desktop_shortcut() -> None:
    """Create a double-clickable Desktop shortcut that launches the UI.

    Windows -> a .lnk (via PowerShell); macOS -> a .command; Linux -> a
    .desktop entry. Each runs the bundled launcher, which prepends the tool
    dirs (ffmpeg / winget Links) to PATH before starting the UI.
    """
    from .desktop import create_shortcut

    path = create_shortcut(log=console.print)
    if path:
        console.print(f"[green]Desktop shortcut created[/] — double-click {path.name} to launch StemForge.")
    else:
        console.print("[red]could not create a desktop shortcut[/] (see the message above)")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
