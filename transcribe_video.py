#!/usr/bin/env python3
"""Transcribe media files with Whisper and optionally add subtitles to video.

Runtime dependencies:
    - ffmpeg and ffprobe on PATH
    - the OpenAI Whisper CLI (`whisper`) on PATH

The script supports both interactive use and repeatable command-line execution.
It never installs software or overwrites an existing output unless explicitly
requested with --overwrite.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NoReturn, Sequence


LOGGER = logging.getLogger("transcribe_video")
SUPPORTED_MODELS = ("tiny", "base", "small", "medium", "large")
SUBTITLE_MODES = ("auto", "none", "soft", "hard")


class ApplicationError(RuntimeError):
    """Raised for errors that should be reported without a traceback."""


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe an audio or video file with Whisper and optionally "
            "create an MP4 video with subtitles."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="input audio or video file; prompted for when omitted",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="directory for generated files; defaults to the input file directory",
    )
    parser.add_argument(
        "-l",
        "--language",
        help="spoken language passed to Whisper; defaults to Hindi",
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=SUPPORTED_MODELS,
        help="Whisper model to use; defaults to small",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Whisper inference device, for example cpu, cuda, or cuda:0 (default: cpu)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="enable FP16 inference; disabled by default and not supported on CPU",
    )
    parser.add_argument(
        "--subtitle-mode",
        choices=SUBTITLE_MODES,
        default="auto",
        help=(
            "video subtitle mode: auto prompts in interactive use and otherwise "
            "creates only the SRT file (default: auto)"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing generated output files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="include diagnostic details in the console output",
    )
    return parser


def configure_logging(verbose: bool) -> None:
    """Configure concise console logging."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def fail(message: str) -> NoReturn:
    """Raise an expected application error."""
    raise ApplicationError(message)


def prompt_text(message: str, *, default: str | None = None) -> str:
    """Prompt until the user enters a non-empty value."""
    suffix = f" [{default}]" if default is not None else ""
    while True:
        try:
            value = input(f"{message}{suffix}: ").strip().strip('"')
        except EOFError as exc:
            raise ApplicationError(
                "Interactive input is unavailable. Provide all required values "
                "as command-line arguments."
            ) from exc
        if value:
            return value
        if default is not None:
            return default
        print("A value is required.")


def prompt_choice(message: str, choices: dict[str, tuple[str, str]]) -> str:
    """Prompt for one of a small set of numbered choices."""
    print(f"\n{message}")
    for number, (label, _) in choices.items():
        print(f"{number} - {label}")

    while True:
        selection = prompt_text("Enter your choice")
        if selection in choices:
            return choices[selection][1]
        print(f"Invalid choice. Enter one of: {', '.join(choices)}.")


def resolve_input_path(raw_path: Path | None) -> tuple[Path, bool]:
    """Resolve and validate the input file path."""
    interactive = raw_path is None
    candidate = Path(prompt_text("Enter the full path to an input media file")) if interactive else raw_path
    assert candidate is not None

    try:
        resolved = candidate.expanduser().resolve(strict=True)
    except OSError as exc:
        raise ApplicationError(f"Input file does not exist or is inaccessible: {candidate}") from exc

    if not resolved.is_file():
        fail(f"Input path is not a file: {resolved}")
    return resolved, interactive


def resolve_output_dir(raw_path: Path | None, input_path: Path, interactive: bool) -> Path:
    """Resolve the output directory, creating it if needed."""
    if raw_path is None and interactive:
        raw_path = Path(
            prompt_text(
                "Enter an output directory",
                default=str(input_path.parent),
            )
        )

    candidate = raw_path if raw_path is not None else input_path.parent
    try:
        resolved = candidate.expanduser().resolve()
        resolved.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ApplicationError(f"Cannot create or access output directory: {candidate}") from exc

    if not resolved.is_dir():
        fail(f"Output path is not a directory: {resolved}")
    return resolved


def require_executable(name: str) -> str:
    """Return an executable path or fail with an actionable message."""
    executable = shutil.which(name)
    if executable is None:
        fail(
            f"Required executable '{name}' was not found on PATH. "
            "Install it through your approved software distribution process "
            "and retry."
        )
    LOGGER.debug("Using %s: %s", name, executable)
    return executable


def run_command(
    command: Sequence[str],
    *,
    operation: str,
    capture_output: bool = False,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess without invoking a shell."""
    LOGGER.debug("Running command: %r", list(command))
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    try:
        return subprocess.run(
            list(command),
            check=True,
            capture_output=capture_output,
            encoding="utf-8",
            env=environment,
            errors="replace",
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise ApplicationError(f"{operation} failed because an executable was not found.") from exc
    except subprocess.CalledProcessError as exc:
        details = ""
        if capture_output and exc.stderr:
            details = f" Details: {exc.stderr.strip()}"
        raise ApplicationError(
            f"{operation} failed with exit code {exc.returncode}.{details}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        timeout_text = f"{timeout:g}" if timeout is not None else "the configured"
        raise ApplicationError(f"{operation} timed out after {timeout_text} seconds.") from exc
    except OSError as exc:
        raise ApplicationError(f"{operation} failed: {exc}") from exc


def probe_media(ffprobe: str, input_path: Path) -> tuple[bool, bool]:
    """Return whether the input contains audio and video streams."""
    result = run_command(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "json",
            str(input_path),
        ],
        operation="Media inspection",
        capture_output=True,
        timeout=30,
    )

    try:
        payload = json.loads(result.stdout)
        stream_types = {
            stream.get("codec_type")
            for stream in payload.get("streams", [])
            if isinstance(stream, dict)
        }
    except (AttributeError, json.JSONDecodeError, TypeError) as exc:
        raise ApplicationError("Media inspection returned an invalid response.") from exc

    has_audio = "audio" in stream_types
    has_video = "video" in stream_types
    if not has_audio:
        fail(f"Input file does not contain an audio stream: {input_path}")
    return has_audio, has_video


def ensure_output_available(path: Path, overwrite: bool) -> None:
    """Prevent accidental replacement of a generated artifact."""
    if path.exists() and not overwrite:
        fail(f"Output already exists: {path}. Use --overwrite to replace it.")
    if path.exists() and not path.is_file():
        fail(f"Output path exists and is not a file: {path}")


def publish_file(source: Path, destination: Path, overwrite: bool) -> None:
    """Atomically publish a generated file into the output directory."""
    ensure_output_available(destination, overwrite)
    temporary_path: Path | None = None
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)
        shutil.copyfile(source, temporary_path)
        publish_temporary_file(temporary_path, destination, overwrite)
    except OSError as exc:
        raise ApplicationError(f"Could not write output file: {destination}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def publish_temporary_file(temporary_path: Path, destination: Path, overwrite: bool) -> None:
    """Atomically publish a same-directory temporary file."""
    try:
        if overwrite:
            os.replace(temporary_path, destination)
        else:
            os.link(temporary_path, destination)
            temporary_path.unlink()
    except FileExistsError as exc:
        raise ApplicationError(
            f"Output already exists: {destination}. Use --overwrite to replace it."
        ) from exc
    except OSError as exc:
        raise ApplicationError(f"Could not publish output file: {destination}") from exc


def transcribe_media(
    whisper: str,
    input_path: Path,
    output_srt_path: Path,
    *,
    language: str,
    model: str,
    device: str,
    fp16: bool,
    overwrite: bool,
) -> None:
    """Run Whisper and publish the generated SRT file."""
    ensure_output_available(output_srt_path, overwrite)
    LOGGER.info("Starting transcription with Whisper model '%s'.", model)

    with tempfile.TemporaryDirectory(prefix="transcribe_video_") as temporary_dir:
        temporary_path = Path(temporary_dir)
        run_command(
            [
                whisper,
                str(input_path),
                "--language",
                language,
                "--task",
                "transcribe",
                "--device",
                device,
                "--model",
                model,
                "--fp16",
                str(fp16),
                "--output_format",
                "srt",
                "--output_dir",
                str(temporary_path),
            ],
            operation="Transcription",
        )

        expected_path = temporary_path / f"{input_path.stem}.srt"
        candidates = list(temporary_path.glob("*.srt"))
        generated_path = expected_path if expected_path.is_file() else None
        if generated_path is None and len(candidates) == 1:
            generated_path = candidates[0]
        if generated_path is None:
            fail("Whisper completed but did not produce exactly one SRT subtitle file.")

        publish_file(generated_path, output_srt_path, overwrite)

    LOGGER.info("Subtitle file saved to: %s", output_srt_path)


def escape_subtitle_filter_path(path: Path) -> str:
    """Escape a path for FFmpeg's subtitles filter syntax."""
    escaped = path.resolve().as_posix().replace("\\", "\\\\")
    for character in ("'", ":", ",", "[", "]", ";"):
        escaped = escaped.replace(character, f"\\{character}")
    return escaped


def allocate_temporary_output(destination: Path) -> Path:
    """Reserve a unique temporary output name without leaving the file in place."""
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.stem}.",
            suffix=destination.suffix,
            dir=destination.parent,
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)
        temporary_path.unlink()
        return temporary_path
    except OSError as exc:
        raise ApplicationError(
            f"Could not allocate a temporary output file in: {destination.parent}"
        ) from exc


def encode_subtitles(
    ffmpeg: str,
    input_path: Path,
    srt_path: Path,
    output_path: Path,
    *,
    mode: str,
    overwrite: bool,
) -> None:
    """Create an MP4 containing either a soft or hard subtitle track."""
    ensure_output_available(output_path, overwrite)
    temporary_output = allocate_temporary_output(output_path)

    common_arguments = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-nostdin",
        "-i",
        str(input_path),
    ]
    if mode == "soft":
        LOGGER.info("Muxing subtitles as a selectable track.")
        command = [
            *common_arguments,
            "-i",
            str(srt_path),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-map",
            "1:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-c:s",
            "mov_text",
            "-movflags",
            "+faststart",
            "-n",
            str(temporary_output),
        ]
    elif mode == "hard":
        LOGGER.info("Burning subtitles into the video stream.")
        subtitle_filter = f"subtitles=filename='{escape_subtitle_filter_path(srt_path)}'"
        command = [
            *common_arguments,
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-vf",
            subtitle_filter,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            "-n",
            str(temporary_output),
        ]
    else:
        fail(f"Unsupported subtitle mode: {mode}")

    try:
        run_command(command, operation="Subtitle encoding")
        publish_temporary_file(temporary_output, output_path, overwrite)
    finally:
        temporary_output.unlink(missing_ok=True)

    LOGGER.info("Subtitled video saved to: %s", output_path)


def resolve_language(value: str | None, interactive: bool) -> str:
    """Resolve the Whisper language argument."""
    if value is not None:
        language = value.strip()
    elif interactive:
        language = prompt_choice(
            "Select the transcription language:",
            {
                "1": ("Hindi (default)", "Hindi"),
                "2": ("English", "English"),
                "3": ("Other", ""),
            },
        )
        if not language:
            language = prompt_text("Enter the language name")
    else:
        language = "Hindi"

    if not language:
        fail("Language cannot be empty.")
    return language


def resolve_model(value: str | None, interactive: bool) -> str:
    """Resolve the Whisper model selection."""
    if value is not None:
        return value
    if not interactive:
        return "small"
    return prompt_choice(
        "Select a Whisper model:",
        {
            "1": ("small (default, faster)", "small"),
            "2": ("medium (more accurate)", "medium"),
            "3": ("large (most accurate, slower)", "large"),
        },
    )


def resolve_subtitle_mode(value: str, interactive: bool, has_video: bool) -> str:
    """Resolve whether and how subtitles should be added to a video."""
    if not has_video:
        if value in {"soft", "hard"}:
            fail("Subtitle encoding requires an input file with a video stream.")
        return "none"
    if value != "auto":
        return value
    if not interactive:
        return "none"
    return prompt_choice(
        "Select a subtitle output mode:",
        {
            "1": ("SRT file only", "none"),
            "2": ("Soft subtitles (selectable track)", "soft"),
            "3": ("Hard subtitles (burned into video)", "hard"),
        },
    )


def run(args: argparse.Namespace) -> None:
    """Execute the transcription workflow."""
    print("=" * 60)
    print(" VIDEO TRANSCRIPTION AND SUBTITLE ENCODER")
    print("=" * 60)

    input_path, interactive = resolve_input_path(args.input)
    output_dir = resolve_output_dir(args.output_dir, input_path, interactive)
    language = resolve_language(args.language, interactive)
    model = resolve_model(args.model, interactive)

    if not args.device.strip():
        fail("Whisper device cannot be empty.")
    if args.fp16 and args.device.casefold() == "cpu":
        fail("--fp16 cannot be used with the CPU device.")

    ffmpeg = require_executable("ffmpeg")
    ffprobe = require_executable("ffprobe")
    whisper = require_executable("whisper")

    _, has_video = probe_media(ffprobe, input_path)
    subtitle_mode = resolve_subtitle_mode(args.subtitle_mode, interactive, has_video)

    output_srt_path = output_dir / f"{input_path.stem}.srt"
    output_video_path = output_dir / f"{input_path.stem}_subtitled.mp4"
    ensure_output_available(output_srt_path, args.overwrite)
    if subtitle_mode != "none":
        ensure_output_available(output_video_path, args.overwrite)

    transcribe_media(
        whisper,
        input_path,
        output_srt_path,
        language=language,
        model=model,
        device=args.device,
        fp16=args.fp16,
        overwrite=args.overwrite,
    )

    if subtitle_mode != "none":
        encode_subtitles(
            ffmpeg,
            input_path,
            output_srt_path,
            output_video_path,
            mode=subtitle_mode,
            overwrite=args.overwrite,
        )

    print("\n" + "=" * 60)
    print(" Processing complete.")
    print("=" * 60)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run the application, and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    try:
        run(args)
    except KeyboardInterrupt:
        LOGGER.error("Interrupted by user.")
        return 130
    except ApplicationError as exc:
        LOGGER.error("%s", exc)
        return 1
    except Exception:
        LOGGER.exception("Unexpected failure.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
