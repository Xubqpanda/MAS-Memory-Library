# src/tools/mm_tools.py
from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from .base import Tool


def _read_pdf(file_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        return "TextReadError: pypdf is not installed."

    reader = PdfReader(str(file_path))
    texts = []
    for idx, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        texts.append(f"\n# Page {idx + 1}\n{page_text}")
    return "\n".join(texts).strip()


def _read_plain_text(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8", errors="ignore")


class TextInspectorTool(Tool):
    name = "inspect_file_as_text"
    description = "Read local file as text. Supports txt/md/log/json/csv/pdf."

    def run(self, file_path: str, question: str | None = None) -> str:
        path = Path(file_path)
        if not path.exists():
            return f"TextReadError: file not found: {file_path}"
        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                content = _read_pdf(path)
            else:
                content = _read_plain_text(path)
        except Exception as exc:
            return f"TextReadError: {exc}"

        if not content:
            return "TextReadError: empty content."
        if question:
            return (
                f"Question: {question}\n"
                f"Relevant file content (truncated to 12000 chars):\n"
                f"{content[:12000]}"
            )
        return content[:20000]


class VisualInspectorTool(Tool):
    name = "inspect_file_as_image"
    description = "Inspect local image metadata. If vision model is configured later, can be extended."

    def run(self, file_path: str, question: str | None = None) -> str:
        path = Path(file_path)
        if not path.exists():
            return f"ImageReadError: file not found: {file_path}"
        mime, _ = mimetypes.guess_type(str(path))
        if not (mime or "").startswith("image/"):
            return "ImageReadError: not an image file."

        try:
            from PIL import Image
        except Exception:
            return "ImageReadError: Pillow is not installed."

        try:
            with Image.open(path) as img:
                width, height = img.size
                mode = img.mode
                fmt = img.format
        except Exception as exc:
            return f"ImageReadError: {exc}"

        note = f"Question: {question}\n" if question else ""
        return (
            f"{note}Image metadata:\n"
            f"- path: {path}\n"
            f"- format: {fmt}\n"
            f"- mode: {mode}\n"
            f"- size: {width}x{height}\n"
        )


class AudioInspectorTool(Tool):
    name = "inspect_file_as_audio"
    description = "Inspect local audio metadata and optional Whisper transcription."

    def run(self, file_path: str, question: str | None = None) -> str:
        path = Path(file_path)
        if not path.exists():
            return f"AudioReadError: file not found: {file_path}"

        mime, _ = mimetypes.guess_type(str(path))
        if not (mime or "").startswith("audio/"):
            return "AudioReadError: not an audio file."

        # Lightweight default behavior: no forced external API call.
        result = [f"Audio file: {path}", f"MIME: {mime}"]
        if question:
            result.append(f"Question: {question}")
            result.append("Transcription is disabled in default tool runtime.")
        return "\n".join(result)


class EncodeFileBase64Tool(Tool):
    name = "encode_file_base64"
    description = "Read local file and return base64 content."

    def run(self, file_path: str) -> str:
        path = Path(file_path)
        if not path.exists():
            return f"EncodeError: file not found: {file_path}"
        data = base64.b64encode(path.read_bytes()).decode("utf-8")
        return data


def build_default_mm_tools() -> list[Tool]:
    return [
        TextInspectorTool(),
        VisualInspectorTool(),
        AudioInspectorTool(),
        EncodeFileBase64Tool(),
    ]
